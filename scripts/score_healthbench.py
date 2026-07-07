#!/usr/bin/env python
"""
Score HealthBench Hard strategy responses with rubric-item grader decisions.

Use --fake-grader only for local plumbing tests. Reported scores should use the
OpenAI-backed grader, matching the HealthBench model-graded rubric design.
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.healthbench_scoring import (  # noqa: E402
    DEFAULT_GRADER_MODEL,
    KeywordFakeGrader,
    OpenAIHealthBenchGrader,
    score_response,
)
from src.healthbench.io import append_jsonl, completed_score_keys, read_jsonl, write_json  # noqa: E402


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "score.log"),
        ],
    )


def parse_strategies(raw: str | None, first_record: dict[str, Any]) -> list[str]:
    if raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return sorted(first_record.get("strategies", {}).keys())


def duplicate_case_ids(records: list[dict[str, Any]], rate: float, seed: int) -> set[str]:
    if rate <= 0:
        return set()
    case_ids = [str(record["case_id"]) for record in records]
    n = max(1, round(len(case_ids) * rate))
    rng = random.Random(seed)
    return set(rng.sample(case_ids, min(n, len(case_ids))))


def score_record_strategy(
    record: dict[str, Any],
    strategy: str,
    grader,
    replicate: str,
) -> dict[str, Any]:
    response_text = record["strategies"][strategy]["response_text"]
    scored = score_response(
        prompt_messages=record["prompt_messages"],
        response_text=response_text,
        rubrics=record["rubrics"],
        grader=grader,
    )
    return {
        "case_id": record["case_id"],
        "prompt_id": record["prompt_id"],
        "strategy": strategy,
        "replicate": replicate,
        "score": scored["score"],
        "example_tags": record.get("example_tags", []),
        "strategy_metadata": record["strategies"][strategy].get("metadata", {}),
        "agent_std": record.get("agent_std", {}),
        "cas_to_centroid": record.get("cas_to_centroid", {}),
        "decisions": scored["decisions"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score HealthBench responses.")
    parser.add_argument("--responses-jsonl", required=True)
    parser.add_argument("--out-jsonl", default=None)
    parser.add_argument("--strategies", default=None)
    parser.add_argument("--grader-model", default=DEFAULT_GRADER_MODEL)
    parser.add_argument("--fake-grader", action="store_true")
    parser.add_argument("--duplicate-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    responses_path = Path(args.responses_jsonl)
    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else responses_path.with_name("scores.jsonl")
    setup_logging(out_jsonl.parent)

    records = read_jsonl(responses_path)
    if args.limit_cases is not None:
        records = records[: args.limit_cases]
    if not records:
        raise ValueError(f"No response records found in {responses_path}")

    if args.overwrite and out_jsonl.exists():
        out_jsonl.unlink()

    strategies = parse_strategies(args.strategies, records[0])
    grader = KeywordFakeGrader() if args.fake_grader else OpenAIHealthBenchGrader(args.grader_model)
    duplicate_ids = duplicate_case_ids(records, args.duplicate_rate, args.seed)
    completed = completed_score_keys(read_jsonl(out_jsonl))

    write_json(
        out_jsonl.with_suffix(".manifest.json"),
        {
            "responses_jsonl": str(responses_path),
            "out_jsonl": str(out_jsonl),
            "strategies": strategies,
            "grader_model": "keyword_fake" if args.fake_grader else args.grader_model,
            "duplicate_rate": args.duplicate_rate,
            "duplicate_case_ids": sorted(duplicate_ids),
        },
    )

    for idx, record in enumerate(records, start=1):
        logging.info("Scoring case %s/%s: %s", idx, len(records), record["case_id"])
        replicates = ["primary"] + (["duplicate"] if record["case_id"] in duplicate_ids else [])
        for strategy in strategies:
            if strategy not in record["strategies"]:
                logging.warning("Skipping missing strategy %s for %s", strategy, record["case_id"])
                continue
            for replicate in replicates:
                key = (record["case_id"], strategy, replicate)
                if key in completed:
                    logging.info("Skipping completed score %s", key)
                    continue
                scored = score_record_strategy(record, strategy, grader, replicate)
                append_jsonl(out_jsonl, scored)
                completed.add(key)

    logging.info("Scoring complete: %s", out_jsonl)


if __name__ == "__main__":
    main()
