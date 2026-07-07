import tempfile
import unittest
import importlib.util
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


if __name__ == "__main__":
    unittest.main()
