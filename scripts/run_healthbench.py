#!/usr/bin/env python
"""
Run HealthBench Hard freeform debate routing experiment.

Strategies:
  - single_neutral: one neutral response
  - debate_round_robin: capped discussion, fixed next speaker order
  - debate_agent_handoff: current speaker chooses the next speaker by text handoff
  - debate_jlens_next: J-lens scores eligible next speakers and routes to the best

Outputs:
  results/healthbench_hard/<run_name>/responses.jsonl
  results/healthbench_hard/<run_name>/vectors/<case_id>/<route_step_agent>.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.healthbench import (  # noqa: E402
    CLINICIAN_ORDER,
    DebateTurn,
    FREE_TEXT_STRATEGIES,
    build_debate_turn_messages,
    build_handoff_choice_messages,
    build_initial_debate_messages,
    build_neutral_messages,
    jlens_next_score,
    next_round_robin_speaker,
    parse_handoff_choice,
)
from src.data.load_healthbench import (  # noqa: E402
    generation_prompt,
    load_healthbench_hard,
    scoring_payload,
)
from src.healthbench.io import append_jsonl, completed_case_ids, read_jsonl, write_json  # noqa: E402
from src.healthbench.modeling import (  # noqa: E402
    default_lens_path,
    generate_text,
    load_jlens_model,
    load_lens,
    load_model_tokenizer,
    render_chat_prompt,
)


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )


def load_config() -> dict[str, Any]:
    import yaml

    with (ROOT / "configs" / "experiment.yaml").open() as f:
        return yaml.safe_load(f)


def serialize_jspace(jspace: dict[str, Any]) -> dict[str, Any]:
    """Make an extract_jspace result JSON-serializable, excluding vector."""
    return {
        "tokens": {str(layer): tokens for layer, tokens in jspace["tokens"].items()},
        "token_ids": {str(layer): ids for layer, ids in jspace["token_ids"].items()},
        "layers_read": [int(layer) for layer in jspace["layers_read"]],
    }


def save_agent_vector(vector, out_dir: Path, case_id: str, vector_name: str) -> str:
    import torch

    vector_dir = out_dir / "vectors" / case_id
    vector_dir.mkdir(parents=True, exist_ok=True)
    path = vector_dir / f"{vector_name}.pt"
    torch.save(vector.cpu(), path)
    return str(path.relative_to(out_dir))


def turn_to_dict(turn: DebateTurn) -> dict[str, Any]:
    """Serialize a debate turn."""
    return {
        "speaker": turn.speaker,
        "response_text": turn.response_text,
        "handoff_to": turn.handoff_to,
        "route_metadata": turn.route_metadata or {},
    }


def choose_jlens_next_speaker(
    *,
    case_id: str,
    prompt_messages: list[dict[str, str]],
    turns: list[DebateTurn],
    eligible_speakers: list[str],
    jlens_model,
    lens,
    tokenizer,
    out_dir: Path,
    top_k: int,
) -> tuple[str, dict[str, Any]]:
    """Score candidate next speakers with J-lens and return the best route."""
    from src.jspace.extract import extract_jspace
    from src.jspace.metrics import compute_std

    scored: dict[str, dict[str, Any]] = {}
    for candidate in eligible_speakers:
        messages = build_debate_turn_messages(
            prompt_messages,
            speaker=candidate,
            turns=turns,
        )
        prompt_text = render_chat_prompt(tokenizer, messages)
        jspace = extract_jspace(jlens_model, lens, tokenizer, prompt_text, top_k=top_k)
        std = compute_std(jspace["tokens"])
        score = jlens_next_score(std)
        vector_name = f"route_t{len(turns)}_{candidate}"
        vector_path = save_agent_vector(jspace["vector"], out_dir, case_id, vector_name)
        scored[candidate] = {
            "std": round(std, 6),
            "next_one_score": round(score, 6),
            "jspace": serialize_jspace(jspace),
            "vector_path": vector_path,
        }

    selected = max(
        scored,
        key=lambda name: (scored[name]["next_one_score"], -scored[name]["std"], name),
    )
    return selected, {"route_type": "jlens_next_score", "candidate_scores": scored}


def run_debate_strategy(
    *,
    strategy: str,
    prompt_messages: list[dict[str, str]],
    initial_turn: DebateTurn,
    hf_model,
    tokenizer,
    jlens_model,
    lens,
    model_cfg: dict[str, Any],
    out_dir: Path,
    case_id: str,
    discussion_round_cap: int,
    top_k: int,
) -> tuple[str, dict[str, Any]]:
    """Run one capped freeform debate under a routing policy."""
    turns = [initial_turn]
    route_trace: list[dict[str, Any]] = []

    while len(turns) < discussion_round_cap:
        current = turns[-1].speaker
        eligible = [speaker for speaker in CLINICIAN_ORDER if speaker != current]

        if strategy == "debate_round_robin":
            next_speaker = next_round_robin_speaker(current, eligible)
            route = {"route_type": "round_robin"}
        elif strategy == "debate_agent_handoff":
            choice_text = generate_text(
                hf_model,
                tokenizer,
                build_handoff_choice_messages(
                    prompt_messages,
                    current_speaker=current,
                    turns=turns,
                    eligible_speakers=eligible,
                ),
                model_cfg,
            )
            parsed = parse_handoff_choice(choice_text, eligible)
            next_speaker = parsed or next_round_robin_speaker(current, eligible)
            route = {
                "route_type": "agent_handoff",
                "raw_handoff": choice_text,
                "parsed_handoff": parsed,
                "fallback_used": parsed is None,
            }
        elif strategy == "debate_jlens_next":
            next_speaker, route = choose_jlens_next_speaker(
                case_id=case_id,
                prompt_messages=prompt_messages,
                turns=turns,
                eligible_speakers=eligible,
                jlens_model=jlens_model,
                lens=lens,
                tokenizer=tokenizer,
                out_dir=out_dir,
                top_k=top_k,
            )
        else:
            raise ValueError(f"Unknown debate strategy: {strategy}")

        logging.info(
            "%s %s turn %d: %s -> %s",
            case_id,
            strategy,
            len(turns),
            current,
            next_speaker,
        )
        response_text = generate_text(
            hf_model,
            tokenizer,
            build_debate_turn_messages(
                prompt_messages,
                speaker=next_speaker,
                turns=turns,
            ),
            model_cfg,
        )
        route_trace.append(
            {
                "turn_index": len(turns),
                "from": current,
                "to": next_speaker,
                **route,
            }
        )
        turns.append(
            DebateTurn(
                speaker=next_speaker,
                response_text=response_text,
                route_metadata=route,
            )
        )

    return turns[-1].response_text, {
        "strategy": strategy,
        "discussion_round_cap": discussion_round_cap,
        "turns": [turn_to_dict(turn) for turn in turns],
        "route_trace": route_trace,
    }


def run_case(
    case: dict[str, Any],
    hf_model,
    tokenizer,
    jlens_model,
    lens,
    model_cfg: dict[str, Any],
    out_dir: Path,
    top_k: int,
    discussion_round_cap: int,
) -> dict[str, Any]:
    """Run all HealthBench strategies for one case."""
    prompt_messages = generation_prompt(case)

    logging.info("Generating single_neutral for %s", case["case_id"])
    single_neutral = generate_text(
        hf_model,
        tokenizer,
        build_neutral_messages(prompt_messages),
        model_cfg,
    )

    logging.info("Generating shared initial debate turn for %s", case["case_id"])
    initial_response = generate_text(
        hf_model,
        tokenizer,
        build_initial_debate_messages(prompt_messages, speaker=CLINICIAN_ORDER[0]),
        model_cfg,
    )
    initial_turn = DebateTurn(
        speaker=CLINICIAN_ORDER[0],
        response_text=initial_response,
    )

    strategy_outputs = {
        "single_neutral": {
            "response_text": single_neutral,
            "metadata": {"strategy": "single_neutral"},
        }
    }

    for strategy in [
        "debate_round_robin",
        "debate_agent_handoff",
        "debate_jlens_next",
    ]:
        final_text, metadata = run_debate_strategy(
            strategy=strategy,
            prompt_messages=prompt_messages,
            initial_turn=initial_turn,
            hf_model=hf_model,
            tokenizer=tokenizer,
            jlens_model=jlens_model,
            lens=lens,
            model_cfg=model_cfg,
            out_dir=out_dir,
            case_id=case["case_id"],
            discussion_round_cap=discussion_round_cap,
            top_k=top_k,
        )
        strategy_outputs[strategy] = {
            "response_text": final_text,
            "metadata": metadata,
        }

    return {
        **scoring_payload(case),
        "prompt_messages": prompt_messages,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "clinician_order": CLINICIAN_ORDER,
        "discussion_round_cap": discussion_round_cap,
        "strategies": {
            name: strategy_outputs[name]
            for name in FREE_TEXT_STRATEGIES
            if name in strategy_outputs
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HealthBench Hard generation.")
    parser.add_argument("--n-cases", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model-key", type=str, default="healthbench_primary")
    parser.add_argument("--lens-path", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--discussion-round-cap", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    hb_cfg = cfg["benchmarks"]["healthbench_hard"]
    run_cfg = cfg.get("healthbench_run", {})

    n_cases = args.n_cases if args.n_cases is not None else hb_cfg.get("n_cases", 100)
    seed = args.seed if args.seed is not None else cfg["analysis"].get("seed", 42)
    model_key = args.model_key
    model_cfg = cfg["models"].get(model_key) or cfg["models"]["primary"]
    top_k = args.top_k if args.top_k is not None else cfg["jspace"].get("top_k", 50)
    discussion_round_cap = (
        args.discussion_round_cap
        if args.discussion_round_cap is not None
        else run_cfg.get("discussion_round_cap", 5)
    )

    run_name = args.run_name or f"healthbench_hard_n{n_cases}_seed{seed}"
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "healthbench_hard" / run_name
    setup_logging(out_dir)

    responses_path = out_dir / "responses.jsonl"
    if args.overwrite and responses_path.exists():
        responses_path.unlink()

    logging.info("Loading %s HealthBench Hard cases with seed %s", n_cases, seed)
    cases = load_healthbench_hard(
        n_cases=n_cases,
        seed=seed,
        path=args.data_path,
        download_if_missing=run_cfg.get("download_if_missing", True),
    )

    model_cfg = {**model_cfg, **run_cfg.get("generation", {})}
    lens_path = Path(args.lens_path) if args.lens_path else default_lens_path(ROOT, model_cfg["name"])

    write_json(
        out_dir / "run_manifest.json",
        {
            "run_name": run_name,
            "n_cases": n_cases,
            "seed": seed,
            "model_key": model_key,
            "model_cfg": model_cfg,
            "lens_path": str(lens_path),
            "strategies": FREE_TEXT_STRATEGIES,
            "discussion_round_cap": discussion_round_cap,
            "clinician_order": CLINICIAN_ORDER,
            "case_ids": [case["case_id"] for case in cases],
        },
    )

    logging.info("Loading model %s", model_cfg["name"])
    hf_model, tokenizer = load_model_tokenizer(model_cfg)
    jlens_model = load_jlens_model(hf_model, tokenizer)
    logging.info("Loading lens from %s", lens_path)
    lens = load_lens(lens_path)

    done = completed_case_ids(read_jsonl(responses_path))
    logging.info("Resuming with %s completed cases", len(done))

    for idx, case in enumerate(cases, start=1):
        if case["case_id"] in done:
            logging.info("Skipping completed case %s/%s: %s", idx, len(cases), case["case_id"])
            continue
        logging.info("Running case %s/%s: %s", idx, len(cases), case["case_id"])
        record = run_case(
            case=case,
            hf_model=hf_model,
            tokenizer=tokenizer,
            jlens_model=jlens_model,
            lens=lens,
            model_cfg=model_cfg,
            out_dir=out_dir,
            top_k=top_k,
            discussion_round_cap=discussion_round_cap,
        )
        append_jsonl(responses_path, record)

    logging.info("HealthBench generation complete: %s", responses_path)


if __name__ == "__main__":
    main()
