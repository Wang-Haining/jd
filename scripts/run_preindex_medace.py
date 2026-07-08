#!/usr/bin/env python
"""Run simplified pre-index MedACE extraction for a balanced AKI cohort."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aki_phenotyping.preindex_medace import run_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-index AKI MedACE extraction.")
    parser.add_argument("--cohort-csv", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8020/v1")
    parser.add_argument("--model", default="gpt-oss:120b")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--schema-mode", choices=["free", "json_schema"], default="json_schema")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    results = run_batch(
        cohort_csv=args.cohort_csv,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        limit=args.limit,
        workers=args.workers,
        overwrite=args.overwrite,
        schema_mode=args.schema_mode,
    )
    ok = sum(1 for item in results if "error" not in item)
    print(f"Completed {ok}/{len(results)} extractions")


if __name__ == "__main__":
    main()
