#!/usr/bin/env python
"""Analyze AKI routed prediction JSONL outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


OUTCOMES = [
    ("aki_any", "aki_any_probability", "aki_any_call"),
    ("aki_3mo", "aki_3mo_probability", "aki_3mo_call"),
    ("aki_6mo", "aki_6mo_probability", "aki_6mo_call"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AKI routing predictions.")
    parser.add_argument("--pred-jsonl", required=True)
    parser.add_argument("--out-csv", required=True)
    return parser.parse_args()


def read_records(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def auroc(labels: list[bool], probs: list[float]) -> float:
    """Compute AUROC by pairwise ranking with tie credit."""
    pos = [p for y, p in zip(labels, probs) if y]
    neg = [p for y, p in zip(labels, probs) if not y]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                ties += 1.0
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def safe_div(num: float, den: float) -> float:
    return num / den if den else float("nan")


def metric_row(strategy: str, outcome: str, prob_col: str, call_col: str, items: list[dict]) -> dict:
    labels = [bool(item["labels"][outcome]) for item in items]
    probs = [float(item["aggregate"][prob_col]) for item in items]
    calls = [str(item["aggregate"][call_col]) for item in items]
    preds = [prob >= 0.5 for prob in probs]
    tp = sum(y and p for y, p in zip(labels, preds))
    tn = sum((not y) and (not p) for y, p in zip(labels, preds))
    fp = sum((not y) and p for y, p in zip(labels, preds))
    fn = sum(y and (not p) for y, p in zip(labels, preds))
    pos_probs = [p for y, p in zip(labels, probs) if y]
    neg_probs = [p for y, p in zip(labels, probs) if not y]
    return {
        "strategy": strategy,
        "outcome": outcome,
        "n": len(items),
        "positives": int(sum(labels)),
        "auroc": auroc(labels, probs),
        "brier": sum((p - float(y)) ** 2 for y, p in zip(labels, probs)) / len(items),
        "accuracy_at_0_5": safe_div(tp + tn, len(items)),
        "sensitivity_at_0_5": safe_div(tp, tp + fn),
        "specificity_at_0_5": safe_div(tn, tn + fp),
        "ppv_at_0_5": safe_div(tp, tp + fp),
        "npv_at_0_5": safe_div(tn, tn + fn),
        "mean_prob_positive": sum(pos_probs) / len(pos_probs) if pos_probs else float("nan"),
        "mean_prob_negative": sum(neg_probs) / len(neg_probs) if neg_probs else float("nan"),
        "yes_calls": calls.count("yes"),
        "uncertain_calls": calls.count("uncertain"),
        "no_calls": calls.count("no"),
    }


def flatten(records: list[dict]) -> dict[str, list[dict]]:
    by_strategy: dict[str, list[dict]] = {}
    for record in records:
        if "error" in record:
            continue
        for strategy, payload in (record.get("strategies") or {}).items():
            by_strategy.setdefault(strategy, []).append(
                {
                    "sample_id": record["sample_id"],
                    "labels": record["labels"],
                    "aggregate": payload["aggregate"],
                }
            )
    return by_strategy


def main() -> None:
    import pandas as pd

    args = parse_args()
    records = read_records(args.pred_jsonl)
    by_strategy = flatten(records)
    rows = []
    for strategy, items in sorted(by_strategy.items()):
        for outcome, prob_col, call_col in OUTCOMES:
            rows.append(metric_row(strategy, outcome, prob_col, call_col, items))
    df = pd.DataFrame(rows)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    primary = df[df["outcome"].eq("aki_any")].sort_values("auroc", ascending=False)
    print("PRIMARY_AKI_ANY")
    print(
        primary[
            [
                "strategy",
                "n",
                "positives",
                "auroc",
                "brier",
                "accuracy_at_0_5",
                "sensitivity_at_0_5",
                "specificity_at_0_5",
                "uncertain_calls",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
