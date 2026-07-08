#!/usr/bin/env python
"""Run structured five-agent AKI predictions from pre-index summaries."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AKI prediction agents.")
    parser.add_argument("--cohort-csv", required=True)
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8020/v1")
    parser.add_argument("--model", default="gpt-oss:120b")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debate-rounds", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add(str(row.get("sample_id")))
    return keys


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
            extra_body={"reasoning_effort": "medium"},
        )
        text = (response.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].removesuffix("```").strip()
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


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def as_optional_float(value):
    try:
        if value != value:
            return None
        return float(value)
    except Exception:
        return None


def main() -> None:
    import pandas as pd

    args = parse_args()
    out_jsonl = Path(args.out_jsonl)
    if args.overwrite and out_jsonl.exists():
        out_jsonl.unlink()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    client = OpenAI(base_url=args.base_url, api_key=api_key, timeout=300.0)
    cohort = pd.read_csv(args.cohort_csv)
    if args.limit is not None:
        cohort = cohort.head(args.limit)

    done = load_existing_keys(out_jsonl)
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
        predictions: list[AKIPrediction] = []
        discussion = ""
        for agent_id in AKI_AGENT_ORDER:
            prediction = call_prediction(
                client,
                args.model,
                build_prediction_messages(
                    agent_id=agent_id,
                    summary=summary,
                    prior_discussion=discussion,
                ),
            )
            predictions.append(prediction)
            discussion += (
                f"\n[{agent_id}] any={prediction.aki_any_probability:.2f}, "
                f"3mo={prediction.aki_3mo_probability:.2f}, "
                f"6mo={prediction.aki_6mo_probability:.2f}; "
                f"{prediction.rationale}"
            )

        # Optional second-pass debate in fixed order.
        for _ in range(args.debate_rounds):
            updated: list[AKIPrediction] = []
            for agent_id in AKI_AGENT_ORDER:
                prediction = call_prediction(
                    client,
                    args.model,
                    build_prediction_messages(
                        agent_id=agent_id,
                        summary=summary,
                        prior_discussion=discussion,
                    ),
                )
                updated.append(prediction)
                discussion += (
                    f"\n[debate:{agent_id}] any={prediction.aki_any_probability:.2f}, "
                    f"3mo={prediction.aki_3mo_probability:.2f}, "
                    f"6mo={prediction.aki_6mo_probability:.2f}; "
                    f"{prediction.rationale}"
                )
            predictions = updated

        aggregate = aggregate_predictions(predictions)
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
                "agent_predictions": [p.model_dump() for p in predictions],
                "aggregate": aggregate.model_dump(),
            },
        )
        print(f"Wrote predictions for {sample_id}")


if __name__ == "__main__":
    main()
