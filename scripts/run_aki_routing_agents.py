#!/usr/bin/env python
"""Run AKI routed tumor-board prediction strategies."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.aki import (  # noqa: E402
    AKI_AGENT_ORDER,
    AKIPrediction,
    aggregate_predictions,
    build_prediction_messages,
    normalize_prediction_calls,
    prediction_schema,
)
from src.agents.aki_routing import (  # noqa: E402
    AKI_ROUTING_STRATEGIES,
    AKIRoutingTurn,
    build_handoff_choice_messages,
    format_turns,
    next_round_robin_agent,
    parse_handoff_choice,
    select_by_std,
)
from src.healthbench.modeling import (  # noqa: E402
    default_lens_path,
    load_jlens_model,
    load_lens,
    load_model_tokenizer,
    render_chat_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AKI routing strategies.")
    parser.add_argument("--cohort-csv", required=True)
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8020/v1")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--turn-cap", type=int, default=5)
    parser.add_argument(
        "--strategies",
        default="single_nephrologist,five_agent_independent_mean,arbitrary_round_robin,agent_handoff,jlens_low_std,jlens_high_std",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jlens-hf-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--jlens-dtype", default="float16")
    parser.add_argument("--lens-path", default=None)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def parse_strategy_list(raw: str) -> list[str]:
    strategies = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in strategies if item not in AKI_ROUTING_STRATEGIES]
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}")
    return strategies


def load_existing_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "error" not in row:
                done.add(str(row.get("sample_id")))
    return done


def strip_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def call_prediction(client: OpenAI, model: str, messages: list[dict[str, str]]) -> AKIPrediction:
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "aki_prediction", "schema": prediction_schema()},
    }
    messages = list(messages)
    last_error = ""
    for _ in range(4):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=5000,
            response_format=response_format,
        )
        text = strip_json(response.choices[0].message.content or "")
        try:
            return normalize_prediction_calls(AKIPrediction.model_validate(json.loads(text)))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": "Fix the JSON to exactly match the schema. Error:\n" + last_error[:2500],
                }
            )
    raise ValueError(f"Could not produce valid AKIPrediction: {last_error[:500]}")


def call_text(client: OpenAI, model: str, messages: list[dict[str, str]], max_tokens: int = 64) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def as_optional_float(value: Any) -> float | None:
    try:
        if value != value:
            return None
        return float(value)
    except Exception:
        return None


def serialize_turn(turn: AKIRoutingTurn) -> dict[str, Any]:
    return {
        "speaker": turn.speaker,
        "prediction": turn.prediction.model_dump(),
        "route_metadata": turn.route_metadata or {},
    }


def summarize_jspace(jspace: dict[str, Any], max_tokens_per_layer: int = 10) -> dict[str, Any]:
    return {
        "layers_read": [int(layer) for layer in jspace["layers_read"]],
        "tokens": {
            str(layer): tokens[:max_tokens_per_layer]
            for layer, tokens in jspace["tokens"].items()
        },
    }


def save_vector(vector, out_dir: Path, sample_id: str, name: str) -> str:
    import torch

    vector_dir = out_dir / "vectors" / sample_id
    vector_dir.mkdir(parents=True, exist_ok=True)
    path = vector_dir / f"{name}.pt"
    torch.save(vector.cpu(), path)
    return str(path.relative_to(out_dir))


def load_jlens_stack(args: argparse.Namespace):
    model_cfg = {
        "name": args.jlens_hf_model,
        "dtype": args.jlens_dtype,
        "device_map": "auto",
    }
    hf_model, tokenizer = load_model_tokenizer(model_cfg)
    jlens_model = load_jlens_model(hf_model, tokenizer)
    lens_path = Path(args.lens_path) if args.lens_path else default_lens_path(ROOT, args.jlens_hf_model)
    lens = load_lens(lens_path)
    return jlens_model, lens, tokenizer, str(lens_path)


def score_jlens_candidates(
    *,
    sample_id: str,
    summary: dict,
    turns: list[AKIRoutingTurn],
    eligible_agents: list[str],
    jlens_model,
    lens,
    tokenizer,
    out_dir: Path,
    top_k: int,
) -> dict[str, dict[str, Any]]:
    from src.jspace.extract import extract_jspace
    from src.jspace.metrics import compute_std

    prior_discussion = format_turns(turns)
    scored: dict[str, dict[str, Any]] = {}
    for candidate in eligible_agents:
        messages = build_prediction_messages(
            agent_id=candidate,
            summary=summary,
            prior_discussion=prior_discussion,
        )
        prompt_text = render_chat_prompt(tokenizer, messages)
        jspace = extract_jspace(jlens_model, lens, tokenizer, prompt_text, top_k=top_k)
        std = compute_std(jspace["tokens"])
        scored[candidate] = {
            "std": float(std),
            "next_one_score": 1.0 / (1.0 + max(0.0, float(std))),
            "jspace_summary": summarize_jspace(jspace),
            "vector_path": save_vector(
                jspace["vector"],
                out_dir,
                sample_id,
                f"route_t{len(turns)}_{candidate}",
            ),
        }
    return scored


def run_single_nephrologist(
    *,
    client: OpenAI,
    model: str,
    summary: dict,
) -> tuple[list[AKIRoutingTurn], list[dict[str, Any]]]:
    prediction = call_prediction(
        client,
        model,
        build_prediction_messages(agent_id="nephrologist", summary=summary),
    )
    return [AKIRoutingTurn("nephrologist", prediction)], []


def run_independent_mean(
    *,
    client: OpenAI,
    model: str,
    summary: dict,
) -> tuple[list[AKIRoutingTurn], list[dict[str, Any]]]:
    turns = []
    for agent_id in AKI_AGENT_ORDER:
        prediction = call_prediction(
            client,
            model,
            build_prediction_messages(agent_id=agent_id, summary=summary),
        )
        turns.append(AKIRoutingTurn(agent_id, prediction))
    return turns, []


def run_routed_strategy(
    *,
    strategy: str,
    client: OpenAI,
    model: str,
    summary: dict,
    sample_id: str,
    turn_cap: int,
    jlens_model,
    lens,
    tokenizer,
    out_dir: Path,
    top_k: int,
) -> tuple[list[AKIRoutingTurn], list[dict[str, Any]]]:
    start_agent = "oncologist"
    first = call_prediction(
        client,
        model,
        build_prediction_messages(agent_id=start_agent, summary=summary),
    )
    turns = [AKIRoutingTurn(start_agent, first)]
    route_trace: list[dict[str, Any]] = []

    while len(turns) < min(turn_cap, len(AKI_AGENT_ORDER)):
        current = turns[-1].speaker
        spoken = {turn.speaker for turn in turns}
        eligible = [agent for agent in AKI_AGENT_ORDER if agent not in spoken]
        prior_discussion = format_turns(turns)

        if strategy == "arbitrary_round_robin":
            next_agent = next_round_robin_agent(current, eligible)
            route = {"route_type": "round_robin"}
        elif strategy == "agent_handoff":
            raw = call_text(
                client,
                model,
                build_handoff_choice_messages(
                    current_agent=current,
                    prior_discussion=prior_discussion,
                    eligible_agents=eligible,
                ),
            )
            parsed = parse_handoff_choice(raw, eligible)
            next_agent = parsed or next_round_robin_agent(current, eligible)
            route = {
                "route_type": "agent_handoff",
                "raw_handoff": raw,
                "parsed_handoff": parsed,
                "fallback_used": parsed is None,
            }
        elif strategy in {"jlens_low_std", "jlens_high_std"}:
            if jlens_model is None or lens is None or tokenizer is None:
                raise ValueError(f"{strategy} requires a loaded J-lens stack")
            candidate_scores = score_jlens_candidates(
                sample_id=sample_id,
                summary=summary,
                turns=turns,
                eligible_agents=eligible,
                jlens_model=jlens_model,
                lens=lens,
                tokenizer=tokenizer,
                out_dir=out_dir,
                top_k=top_k,
            )
            mode = "low" if strategy == "jlens_low_std" else "high"
            next_agent, selection_meta = select_by_std(
                current,
                {
                    agent: {
                        "std": values["std"],
                        "next_one_score": values["next_one_score"],
                    }
                    for agent, values in candidate_scores.items()
                },
                mode=mode,
            )
            route = {
                "route_type": strategy,
                "selection_mode": mode,
                "candidate_scores": candidate_scores,
                **selection_meta,
            }
        else:
            raise ValueError(f"Unsupported routed strategy: {strategy}")

        prediction = call_prediction(
            client,
            model,
            build_prediction_messages(
                agent_id=next_agent,
                summary=summary,
                prior_discussion=prior_discussion,
            ),
        )
        route_trace.append(
            {
                "turn_index": len(turns),
                "from": current,
                "to": next_agent,
                **route,
            }
        )
        turns.append(AKIRoutingTurn(next_agent, prediction, route_metadata=route))
    return turns, route_trace


def strategy_to_record(
    *,
    strategy: str,
    turns: list[AKIRoutingTurn],
    route_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate = aggregate_predictions([turn.prediction for turn in turns], strategy=strategy)
    return {
        "aggregate": aggregate.model_dump(),
        "turns": [serialize_turn(turn) for turn in turns],
        "route_trace": route_trace,
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def run_strategy(
    *,
    strategy: str,
    client: OpenAI,
    model: str,
    summary: dict,
    sample_id: str,
    turn_cap: int,
    jlens_model,
    lens,
    tokenizer,
    out_dir: Path,
    top_k: int,
) -> dict[str, Any]:
    if strategy == "single_nephrologist":
        turns, route_trace = run_single_nephrologist(client=client, model=model, summary=summary)
    elif strategy == "five_agent_independent_mean":
        turns, route_trace = run_independent_mean(client=client, model=model, summary=summary)
    else:
        turns, route_trace = run_routed_strategy(
            strategy=strategy,
            client=client,
            model=model,
            summary=summary,
            sample_id=sample_id,
            turn_cap=turn_cap,
            jlens_model=jlens_model,
            lens=lens,
            tokenizer=tokenizer,
            out_dir=out_dir,
            top_k=top_k,
        )
    return strategy_to_record(strategy=strategy, turns=turns, route_trace=route_trace)


def main() -> None:
    import pandas as pd

    args = parse_args()
    strategies = parse_strategy_list(args.strategies)
    out_jsonl = Path(args.out_jsonl)
    out_dir = out_jsonl.parent
    if args.overwrite and out_jsonl.exists():
        out_jsonl.unlink()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    client = OpenAI(base_url=args.base_url, api_key=api_key, timeout=300.0)
    cohort = pd.read_csv(args.cohort_csv)
    if args.limit is not None:
        cohort = cohort.head(args.limit)

    needs_jlens = any(strategy.startswith("jlens_") for strategy in strategies)
    if needs_jlens:
        jlens_model, lens, tokenizer, lens_path = load_jlens_stack(args)
    else:
        jlens_model = lens = tokenizer = None
        lens_path = None

    done = load_existing_sample_ids(out_jsonl)
    for row in cohort.itertuples(index=False):
        sample_id = f"sample_{int(row.person_id)}"
        if sample_id in done:
            continue
        summary_path = Path(args.summary_dir) / f"{sample_id}.json"
        if not summary_path.exists():
            append_jsonl(
                out_jsonl,
                {
                    "sample_id": sample_id,
                    "person_id": int(row.person_id),
                    "error": f"missing summary {summary_path}",
                },
            )
            continue
        summary = json.loads(summary_path.read_text())
        strategy_records = {}
        for strategy in strategies:
            print(f"{sample_id}: running {strategy}", flush=True)
            strategy_records[strategy] = run_strategy(
                strategy=strategy,
                client=client,
                model=args.model,
                summary=summary,
                sample_id=sample_id,
                turn_cap=args.turn_cap,
                jlens_model=jlens_model,
                lens=lens,
                tokenizer=tokenizer,
                out_dir=out_dir,
                top_k=args.top_k,
            )
        append_jsonl(
            out_jsonl,
            {
                "sample_id": sample_id,
                "person_id": int(row.person_id),
                "labels": {
                    "aki_any": as_bool(row.aki_any),
                    "aki_3mo": as_bool(row.aki_3mo),
                    "aki_6mo": as_bool(row.aki_6mo),
                    "aki_evidence": str(row.aki_evidence),
                    "days_to_aki": as_optional_float(row.days_to_aki),
                },
                "model": args.model,
                "jlens_hf_model": args.jlens_hf_model if needs_jlens else None,
                "lens_path": lens_path,
                "turn_cap": args.turn_cap,
                "strategies": strategy_records,
            },
        )
        print(f"Wrote routing predictions for {sample_id}", flush=True)


if __name__ == "__main__":
    main()
