import json
import tempfile
import unittest
from pathlib import Path

from src.data.load_healthbench import generation_prompt, load_healthbench_hard


class HealthBenchLoaderTest(unittest.TestCase):
    def test_load_healthbench_hard_normalizes_rows(self):
        row = {
            "prompt_id": "abc123",
            "prompt": [
                {"role": "user", "content": "What should I do about chest pain?"}
            ],
            "rubrics": [
                {
                    "criterion": "Advises urgent evaluation for concerning chest pain.",
                    "points": 10,
                    "tags": ["emergency"],
                }
            ],
            "example_tags": ["emergency"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hard.jsonl"
            path.write_text(json.dumps(row) + "\n")
            cases = load_healthbench_hard(
                n_cases=1,
                seed=42,
                path=path,
                download_if_missing=False,
            )

        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case["case_id"], "healthbench_hard_abc123")
        self.assertEqual(case["prompt_messages"][0]["role"], "user")
        self.assertEqual(case["rubrics"][0]["points"], 10.0)
        self.assertEqual(case["example_tags"], ["emergency"])

    def test_generation_prompt_excludes_rubrics(self):
        case = {
            "prompt_messages": [{"role": "user", "content": "Hello"}],
            "rubrics": [{"criterion": "secret", "points": 1}],
        }
        prompt = generation_prompt(case)

        self.assertEqual(prompt, [{"role": "user", "content": "Hello"}])
        self.assertNotIn("rubrics", prompt[0])


if __name__ == "__main__":
    unittest.main()
