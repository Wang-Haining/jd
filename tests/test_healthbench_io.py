import tempfile
import unittest
from pathlib import Path

from src.healthbench.io import (
    append_jsonl,
    completed_case_ids,
    completed_score_keys,
    read_jsonl,
)


class HealthBenchIOTest(unittest.TestCase):
    def test_jsonl_roundtrip_and_completed_case_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.jsonl"
            append_jsonl(path, {"case_id": "a", "value": 1})
            append_jsonl(path, {"case_id": "b", "value": 2})
            rows = read_jsonl(path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(completed_case_ids(rows), {"a", "b"})

    def test_completed_score_keys(self):
        rows = [
            {"case_id": "a", "strategy": "debate_round_robin"},
            {
                "case_id": "a",
                "strategy": "debate_jlens_next",
                "replicate": "duplicate",
            },
        ]

        self.assertEqual(
            completed_score_keys(rows),
            {
                ("a", "debate_round_robin", "primary"),
                ("a", "debate_jlens_next", "duplicate"),
            },
        )


if __name__ == "__main__":
    unittest.main()
