"""Analysis helpers for HealthBench Hard strategy scores."""
from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:
    np = None
    pd = None

from ..healthbench.io import read_jsonl


def load_scores(path: str | Path, replicate: str = "primary") -> pd.DataFrame:
    """Load score JSONL into a DataFrame."""
    _require_deps()
    rows = [
        record
        for record in read_jsonl(path)
        if str(record.get("replicate", "primary")) == replicate
    ]
    flattened = []
    for record in rows:
        route_stats = _route_signal_stats(record.get("strategy_metadata", {}))
        flattened.append(
            {
                "case_id": record["case_id"],
                "prompt_id": record.get("prompt_id"),
                "strategy": record["strategy"],
                "score": record["score"],
                "example_tags": record.get("example_tags", []),
                "agent_std_mean": _mean_or_nan(record.get("agent_std", {}).values()),
                "agent_std_min": _min_or_nan(record.get("agent_std", {}).values()),
                "cas_to_centroid_mean": _mean_or_nan(record.get("cas_to_centroid", {}).values()),
                **route_stats,
            }
        )
    return pd.DataFrame(flattened)


def _mean_or_nan(values) -> float:
    _require_deps()
    vals = [float(v) for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def _min_or_nan(values) -> float:
    _require_deps()
    vals = [float(v) for v in values]
    return float(np.min(vals)) if vals else float("nan")


def _route_signal_stats(metadata: dict) -> dict[str, float]:
    _require_deps()
    route_scores = []
    route_stds = []
    for step in metadata.get("route_trace", []) or []:
        for payload in (step.get("candidate_scores", {}) or {}).values():
            if "next_one_score" in payload:
                route_scores.append(float(payload["next_one_score"]))
            if "std" in payload:
                route_stds.append(float(payload["std"]))
    return {
        "jlens_route_score_mean": float(np.mean(route_scores)) if route_scores else float("nan"),
        "jlens_route_score_max": float(np.max(route_scores)) if route_scores else float("nan"),
        "jlens_route_std_mean": float(np.mean(route_stds)) if route_stds else float("nan"),
        "jlens_route_std_min": float(np.min(route_stds)) if route_stds else float("nan"),
    }


def paired_wide(scores: pd.DataFrame) -> pd.DataFrame:
    """Create case x strategy score matrix."""
    return scores.pivot(index="case_id", columns="strategy", values="score")


def paired_bootstrap_ci(
    values: np.ndarray,
    seed: int = 42,
    n_boot: int = 10000,
    ci_level: float = 0.95,
) -> tuple[float, float]:
    """Paired bootstrap CI for the mean of already-paired differences."""
    _require_deps()
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(clean, size=len(clean), replace=True)
        means.append(float(np.mean(sample)))
    alpha = 1 - ci_level
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def strategy_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize score distributions by strategy."""
    grouped = scores.groupby("strategy")["score"]
    summary = grouped.agg(["count", "mean", "std", "median"]).reset_index()
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    return summary.sort_values("strategy")


def paired_differences(
    scores: pd.DataFrame,
    baseline: str = "debate_round_robin",
    seed: int = 42,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Compute paired mean differences vs a baseline strategy."""
    wide = paired_wide(scores)
    rows = []
    for strategy in sorted(col for col in wide.columns if col != baseline):
        paired = wide[[baseline, strategy]].dropna()
        diff = paired[strategy] - paired[baseline]
        ci_low, ci_high = paired_bootstrap_ci(
            diff.to_numpy(dtype=float),
            seed=seed,
            ci_level=ci_level,
        )
        rows.append(
            {
                "comparison": f"{strategy} - {baseline}",
                "strategy": strategy,
                "baseline": baseline,
                "n_pairs": len(diff),
                "mean_difference": float(diff.mean()) if len(diff) else float("nan"),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def tag_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Exploratory strategy summaries by HealthBench example tag."""
    rows = []
    for _, row in scores.iterrows():
        tags = row["example_tags"] or ["untagged"]
        for tag in tags:
            rows.append({**row.to_dict(), "tag": tag})
    if not rows:
        return pd.DataFrame()
    exploded = pd.DataFrame(rows)
    return (
        exploded.groupby(["tag", "strategy"])["score"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .sort_values(["tag", "strategy"])
    )


def signal_correlations(
    scores: pd.DataFrame,
    baseline: str = "debate_round_robin",
    target: str = "debate_jlens_next",
) -> pd.DataFrame:
    """Correlate run-level J-space signals with paired score changes."""
    wide = paired_wide(scores)
    if baseline not in wide or target not in wide:
        return pd.DataFrame()
    diffs = (wide[target] - wide[baseline]).rename("score_difference").reset_index()
    signal_cols = [
        "case_id",
        "jlens_route_score_mean",
        "jlens_route_score_max",
        "jlens_route_std_mean",
        "jlens_route_std_min",
    ]
    signals = scores[scores["strategy"] == target][signal_cols].drop_duplicates("case_id")
    rows = []
    merged = diffs.merge(signals, on="case_id", how="left")
    for col in [
        "jlens_route_score_mean",
        "jlens_route_score_max",
        "jlens_route_std_mean",
        "jlens_route_std_min",
    ]:
        usable = merged[["score_difference", col]].dropna()
        if len(usable) < 3:
            corr = float("nan")
        else:
            corr = float(usable["score_difference"].corr(usable[col]))
        rows.append({"signal": col, "n": len(usable), "pearson_r": corr})
    return pd.DataFrame(rows)


def write_analysis_outputs(
    scores_jsonl: str | Path,
    out_dir: str | Path,
    baseline: str = "debate_round_robin",
    seed: int = 42,
    ci_level: float = 0.95,
) -> dict[str, str]:
    """Write CSV tables and return output paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores = load_scores(scores_jsonl)

    outputs: dict[str, str] = {}
    tables = {
        "strategy_summary": strategy_summary(scores),
        "paired_differences": paired_differences(scores, baseline, seed, ci_level),
        "tag_summary": tag_summary(scores),
        "signal_correlations": signal_correlations(scores, baseline),
    }
    for name, table in tables.items():
        path = out / f"{name}.csv"
        table.to_csv(path, index=False)
        outputs[name] = str(path)

    return outputs


def _require_deps() -> None:
    if np is None or pd is None:
        raise ModuleNotFoundError(
            "HealthBench analysis requires numpy and pandas. "
            "Install project requirements before running analysis."
        )
