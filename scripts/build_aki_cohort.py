#!/usr/bin/env python
"""Build a balanced post-ICI AKI/no-AKI cohort for pilot experiments."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aki_phenotyping.cohort import (  # noqa: E402
    CohortBuildConfig,
    add_preindex_note_metrics,
    build_balanced_sample,
    load_denominator,
    parse_evidence_list,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced ICI-AKI cohort.")
    parser.add_argument("--denominator-csv", required=True)
    parser.add_argument("--data-dir", default=None, help="irAKI_data root for note metrics.")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--n-per-class", type=int, default=100)
    parser.add_argument("--positive-evidence", default="both")
    parser.add_argument("--control-evidence", default="none")
    parser.add_argument("--min-control-followup-days", type=int, default=180)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-note-metrics",
        action="store_true",
        help="Do not read note parquet files; note strata will default to zero notes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_denominator(args.denominator_csv)
    if args.data_dir and not args.skip_note_metrics:
        frame = add_preindex_note_metrics(frame, args.data_dir, args.lookback_days)
    else:
        frame["preindex_n_notes"] = 0
        frame["preindex_note_chars"] = 0

    config = CohortBuildConfig(
        positive_evidence=parse_evidence_list(args.positive_evidence),
        control_evidence=parse_evidence_list(args.control_evidence),
        min_control_followup_days=args.min_control_followup_days,
        n_per_class=args.n_per_class,
        seed=args.seed,
    )
    sample = build_balanced_sample(frame, config)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_csv, index=False)
    write_manifest(
        out_csv.with_suffix(".manifest.json"),
        input_csv=args.denominator_csv,
        config=config,
        frame=sample,
    )

    print(f"Wrote {len(sample)} rows to {out_csv}")
    print(sample["aki_any"].value_counts(dropna=False).to_string())
    print("3mo:")
    print(sample["aki_3mo"].value_counts(dropna=False).to_string())
    print("6mo:")
    print(sample["aki_6mo"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
