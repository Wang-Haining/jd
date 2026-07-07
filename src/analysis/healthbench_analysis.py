"""Analysis helpers for HealthBench Hard strategy scores."""
from __future__ import annotations

from pathlib import Path
from typing import Any

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
        corr = _safe_corr(usable["score_difference"], usable[col]) if len(usable) >= 3 else float("nan")
        rows.append({"signal": col, "n": len(usable), "pearson_r": corr})
    return pd.DataFrame(rows)


def duplicate_grade_qc(scores_jsonl: str | Path) -> pd.DataFrame:
    """Summarize primary-vs-duplicate grading stability."""
    _require_deps()
    records = read_jsonl(scores_jsonl)
    rows = []
    for record in records:
        rows.append(
            {
                "case_id": record["case_id"],
                "strategy": record["strategy"],
                "replicate": record.get("replicate", "primary"),
                "score": record["score"],
            }
        )
    if not rows:
        return pd.DataFrame([_duplicate_qc_empty_row()])

    frame = pd.DataFrame(rows)
    wide = frame.pivot_table(
        index=["case_id", "strategy"],
        columns="replicate",
        values="score",
        aggfunc="first",
    )
    if "primary" not in wide or "duplicate" not in wide:
        return pd.DataFrame([_duplicate_qc_empty_row()])

    paired = wide[["primary", "duplicate"]].dropna()
    if paired.empty:
        return pd.DataFrame([_duplicate_qc_empty_row()])

    abs_diff = (paired["duplicate"] - paired["primary"]).abs()
    return pd.DataFrame(
        [
            {
                "n_duplicate_pairs": int(len(abs_diff)),
                "mean_abs_difference": float(abs_diff.mean()),
                "median_abs_difference": float(abs_diff.median()),
                "max_abs_difference": float(abs_diff.max()),
            }
        ]
    )


def routing_diagnostics(
    scores_jsonl: str | Path,
    baseline: str = "debate_round_robin",
    target: str = "debate_jlens_next",
    comparator: str = "debate_agent_handoff",
) -> pd.DataFrame:
    """Summarize debate routing behavior from stored strategy metadata."""
    _require_deps()
    records = [
        record
        for record in read_jsonl(scores_jsonl)
        if record.get("replicate", "primary") == "primary"
    ]
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_case.setdefault(str(record["case_id"]), {})[str(record["strategy"])] = record

    target_cases = 0
    baseline_matches = 0
    comparator_matches = 0
    steps = 0
    narrowed_steps = 0
    tie_breakers: dict[str, int] = {}
    selected_to: dict[str, int] = {}
    std_tied_sizes: dict[int, int] = {}

    for strategies in by_case.values():
        if target not in strategies:
            continue
        target_cases += 1
        target_route = _route_to_sequence(strategies[target])
        if baseline in strategies and target_route == _route_to_sequence(strategies[baseline]):
            baseline_matches += 1
        if comparator in strategies and target_route == _route_to_sequence(strategies[comparator]):
            comparator_matches += 1

        for step in _route_trace(strategies[target]):
            steps += 1
            selected = str(step.get("to", "missing"))
            selected_to[selected] = selected_to.get(selected, 0) + 1
            tie_breaker = str(step.get("tie_breaker", "none"))
            tie_breakers[tie_breaker] = tie_breakers.get(tie_breaker, 0) + 1
            candidates = step.get("std_tied_candidates") or step.get("score_tied_candidates") or []
            candidate_count = len(candidates)
            std_tied_sizes[candidate_count] = std_tied_sizes.get(candidate_count, 0) + 1
            candidate_scores = step.get("candidate_scores", {}) or {}
            if candidate_scores and candidate_count < len(candidate_scores):
                narrowed_steps += 1

    rows = [
        {"metric": "n_cases", "value": int(len(by_case))},
        {"metric": "target_cases", "value": int(target_cases)},
        {"metric": "target_route_steps", "value": int(steps)},
        {"metric": f"{target}_equals_{baseline}_cases", "value": int(baseline_matches)},
        {
            "metric": f"{target}_equals_{baseline}_fraction",
            "value": _safe_fraction(baseline_matches, target_cases),
        },
        {"metric": f"{target}_equals_{comparator}_cases", "value": int(comparator_matches)},
        {
            "metric": f"{target}_equals_{comparator}_fraction",
            "value": _safe_fraction(comparator_matches, target_cases),
        },
        {"metric": "target_narrowed_tie_steps", "value": int(narrowed_steps)},
        {
            "metric": "target_narrowed_tie_step_fraction",
            "value": _safe_fraction(narrowed_steps, steps),
        },
    ]
    rows.extend(
        {"metric": f"tie_breaker:{key}", "value": int(value)}
        for key, value in sorted(tie_breakers.items())
    )
    rows.extend(
        {"metric": f"selected_to:{key}", "value": int(value)}
        for key, value in sorted(selected_to.items())
    )
    rows.extend(
        {"metric": f"std_tied_candidate_set_size:{key}", "value": int(value)}
        for key, value in sorted(std_tied_sizes.items())
    )
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
        "duplicate_grade_qc": duplicate_grade_qc(scores_jsonl),
        "routing_diagnostics": routing_diagnostics(scores_jsonl, baseline=baseline),
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


def _safe_corr(left, right) -> float:
    """Pearson correlation, returning NaN for constant vectors."""
    _require_deps()
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return float("nan")
    return float(left.corr(right))


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _route_trace(record: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = record.get("strategy_metadata", {}) or {}
    return metadata.get("route_trace", []) or []


def _route_to_sequence(record: dict[str, Any]) -> list[str | None]:
    return [step.get("to") for step in _route_trace(record)]


def _duplicate_qc_empty_row() -> dict[str, float | int]:
    return {
        "n_duplicate_pairs": 0,
        "mean_abs_difference": float("nan"),
        "median_abs_difference": float("nan"),
        "max_abs_difference": float("nan"),
    }
