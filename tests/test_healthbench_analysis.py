import tempfile
import unittest
import importlib.util
import math
from pathlib import Path

from src.healthbench.io import append_jsonl


HAS_ANALYSIS_DEPS = (
    importlib.util.find_spec("numpy") is not None
    and importlib.util.find_spec("pandas") is not None
)


@unittest.skipUnless(HAS_ANALYSIS_DEPS, "numpy/pandas not installed")
class HealthBenchAnalysisTest(unittest.TestCase):
    def test_paired_difference_primary_comparison(self):
        from src.analysis.healthbench_analysis import load_scores, paired_differences

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.jsonl"
            append_jsonl(path, {"case_id": "a", "strategy": "debate_round_robin", "score": 0.4})
            append_jsonl(path, {"case_id": "a", "strategy": "debate_jlens_next", "score": 0.6})
            append_jsonl(path, {"case_id": "b", "strategy": "debate_round_robin", "score": 0.5})
            append_jsonl(path, {"case_id": "b", "strategy": "debate_jlens_next", "score": 0.7})
            scores = load_scores(path)

        diffs = paired_differences(scores, baseline="debate_round_robin", seed=1, ci_level=0.95)
        row = diffs[diffs["strategy"] == "debate_jlens_next"].iloc[0]

        self.assertEqual(row["n_pairs"], 2)
        self.assertAlmostEqual(row["mean_difference"], 0.2)

    def test_duplicate_grade_qc(self):
        from src.analysis.healthbench_analysis import duplicate_grade_qc

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.jsonl"
            append_jsonl(path, {"case_id": "a", "strategy": "debate_jlens_next", "score": 0.6})
            append_jsonl(
                path,
                {
                    "case_id": "a",
                    "strategy": "debate_jlens_next",
                    "replicate": "duplicate",
                    "score": 0.7,
                },
            )
            append_jsonl(path, {"case_id": "b", "strategy": "debate_jlens_next", "score": 0.4})
            qc = duplicate_grade_qc(path)

        row = qc.iloc[0]
        self.assertEqual(row["n_duplicate_pairs"], 1)
        self.assertAlmostEqual(row["mean_abs_difference"], 0.1)
        self.assertAlmostEqual(row["max_abs_difference"], 0.1)

    def test_routing_diagnostics(self):
        from src.analysis.healthbench_analysis import routing_diagnostics

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.jsonl"
            base_route = [{"from": "generalist", "to": "emergency", "route_type": "round_robin"}]
            target_route = [
                {
                    "from": "generalist",
                    "to": "diagnostician",
                    "route_type": "jlens_next_score",
                    "tie_breaker": "round_robin",
                    "std_tied_candidates": ["diagnostician"],
                    "candidate_scores": {"diagnostician": {}, "emergency": {}},
                }
            ]
            append_jsonl(
                path,
                {
                    "case_id": "a",
                    "strategy": "debate_round_robin",
                    "score": 0.4,
                    "strategy_metadata": {"route_trace": base_route},
                },
            )
            append_jsonl(
                path,
                {
                    "case_id": "a",
                    "strategy": "debate_jlens_next",
                    "score": 0.5,
                    "strategy_metadata": {"route_trace": target_route},
                },
            )
            diag = routing_diagnostics(path)

        metrics = dict(zip(diag["metric"], diag["value"]))
        self.assertEqual(metrics["n_cases"], 1)
        self.assertEqual(metrics["debate_jlens_next_equals_debate_round_robin_cases"], 0)
        self.assertEqual(metrics["target_narrowed_tie_steps"], 1)
        self.assertEqual(metrics["tie_breaker:round_robin"], 1)
        self.assertEqual(metrics["selected_to:diagnostician"], 1)

    def test_signal_correlations_constant_signal_is_nan(self):
        from src.analysis.healthbench_analysis import load_scores, signal_correlations

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.jsonl"
            for idx, target_score in enumerate([0.4, 0.5, 0.6], start=1):
                case_id = f"case-{idx}"
                append_jsonl(
                    path,
                    {"case_id": case_id, "strategy": "debate_round_robin", "score": 0.3},
                )
                append_jsonl(
                    path,
                    {
                        "case_id": case_id,
                        "strategy": "debate_jlens_next",
                        "score": target_score,
                        "strategy_metadata": {
                            "route_trace": [
                                {
                                    "candidate_scores": {
                                        "a": {"next_one_score": 1.0, "std": float(idx)},
                                        "b": {"next_one_score": 1.0, "std": float(idx + 1)},
                                    }
                                }
                            ]
                        },
                    },
                )
            scores = load_scores(path)
            corr = signal_correlations(scores)

        row = corr[corr["signal"] == "jlens_route_score_max"].iloc[0]
        self.assertTrue(math.isnan(row["pearson_r"]))


if __name__ == "__main__":
    unittest.main()
