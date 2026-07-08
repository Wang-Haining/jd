#!/usr/bin/env python
"""Build pre-index demographics, Charlson, and medication features for ICI-AKI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aki_phenotyping.baseline_features import (  # noqa: E402
    CHARLSON_COMORBIDITIES,
    build_baseline_features_from_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pre-index ICI baseline features.")
    parser.add_argument("--cohort-csv", required=True, help="Denominator or cohort CSV.")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--data-dir", default=None, help="Root containing structured_data_fixed.")
    parser.add_argument("--diagnosis-path", default=None, help="condition_occurrence path.")
    parser.add_argument("--medication-path", default=None, help="drug_exposure path.")
    parser.add_argument("--person-path", default=None, help="OMOP person path.")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--max-drugs", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = build_baseline_features_from_paths(
        cohort_csv=args.cohort_csv,
        data_dir=args.data_dir,
        diagnosis_path=args.diagnosis_path,
        medication_path=args.medication_path,
        person_path=args.person_path,
        lookback_days=args.lookback_days,
        max_drugs=args.max_drugs,
    )
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_csv, index=False)

    positive_counts = {
        col: int(features[col].sum())
        for col in CHARLSON_COMORBIDITIES
        if col in features.columns
    }
    manifest = {
        "input_cohort_csv": args.cohort_csv,
        "data_dir": args.data_dir,
        "diagnosis_path": args.diagnosis_path,
        "medication_path": args.medication_path,
        "person_path": args.person_path,
        "lookback_days": args.lookback_days,
        "max_drugs": args.max_drugs,
        "n_rows": int(len(features)),
        "n_with_any_charlson_flag": int(features["charlson_comorbidity_count"].gt(0).sum()),
        "n_with_prior_year_drugs": int(features["baseline_drug_count"].gt(0).sum()),
        "charlson_positive_counts": positive_counts,
    }
    out_csv.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote {len(features)} rows to {out_csv}")
    print(f"Any Charlson flag: {manifest['n_with_any_charlson_flag']}")
    print(f"Any prior-year drug: {manifest['n_with_prior_year_drugs']}")


if __name__ == "__main__":
    main()
