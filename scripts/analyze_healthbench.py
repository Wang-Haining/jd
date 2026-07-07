#!/usr/bin/env python
"""Generate HealthBench Hard analysis tables and figures."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.healthbench_analysis import (  # noqa: E402
    load_scores,
    paired_differences,
    signal_correlations,
    strategy_summary,
    tag_summary,
    write_analysis_outputs,
)


def save_figures(scores_jsonl: Path, out_dir: Path, baseline: str, seed: int, ci_level: float) -> None:
    import matplotlib.pyplot as plt

    scores = load_scores(scores_jsonl)
    summary = strategy_summary(scores)
    diffs = paired_differences(scores, baseline=baseline, seed=seed, ci_level=ci_level)
    signals = signal_correlations(scores, baseline=baseline)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not summary.empty:
        plt.figure(figsize=(7, 4))
        plt.bar(summary["strategy"], summary["mean"], yerr=summary["se"])
        plt.ylabel("Mean HealthBench score")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "strategy_scores.png", dpi=300)
        plt.close()

    if not diffs.empty:
        plt.figure(figsize=(7, 4))
        yerr = [
            diffs["mean_difference"] - diffs["ci_low"],
            diffs["ci_high"] - diffs["mean_difference"],
        ]
        plt.bar(diffs["strategy"], diffs["mean_difference"], yerr=yerr)
        plt.axhline(0, color="black", linewidth=1)
        plt.ylabel(f"Paired score difference vs {baseline}")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / "paired_differences.png", dpi=300)
        plt.close()

    if not signals.empty:
        plt.figure(figsize=(6, 4))
        plt.bar(signals["signal"], signals["pearson_r"])
        plt.axhline(0, color="black", linewidth=1)
        plt.ylabel("Pearson r with J-lens handoff score gain")
        plt.xticks(rotation=25, ha="right", fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "signal_correlations.png", dpi=300)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze HealthBench scores.")
    parser.add_argument("--scores-jsonl", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--baseline", default="debate_round_robin")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ci-level", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores_jsonl = Path(args.scores_jsonl)
    out_dir = Path(args.out_dir) if args.out_dir else scores_jsonl.parent / "analysis"
    outputs = write_analysis_outputs(
        scores_jsonl=scores_jsonl,
        out_dir=out_dir,
        baseline=args.baseline,
        seed=args.seed,
        ci_level=args.ci_level,
    )
    save_figures(scores_jsonl, out_dir, args.baseline, args.seed, args.ci_level)

    print("Analysis outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    print(f"  figures: {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
