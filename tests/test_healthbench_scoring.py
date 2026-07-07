import unittest

from src.analysis.healthbench_scoring import (
    KeywordFakeGrader,
    calculate_score,
    parse_grader_json,
    score_response,
)


class HealthBenchScoringTest(unittest.TestCase):
    def test_calculate_score_includes_negative_met_criteria(self):
        rubrics = [
            {"criterion": "good", "points": 10},
            {"criterion": "bad", "points": -5},
            {"criterion": "also good", "points": 5},
        ]
        decisions = [
            {"criteria_met": True},
            {"criteria_met": True},
            {"criteria_met": False},
        ]

        self.assertAlmostEqual(calculate_score(rubrics, decisions), 5 / 15)

    def test_parse_grader_json_accepts_fenced_json(self):
        parsed = parse_grader_json(
            '```json\n{"explanation": "ok", "criteria_met": true}\n```'
        )

        self.assertTrue(parsed["criteria_met"])
        self.assertEqual(parsed["explanation"], "ok")

    def test_score_response_stores_decisions(self):
        rubrics = [
            {
                "criterion": "Mentions urgent evaluation",
                "points": 10,
                "tags": ["safety"],
            }
        ]
        scored = score_response(
            prompt_messages=[{"role": "user", "content": "Chest pain"}],
            response_text="You need urgent evaluation now.",
            rubrics=rubrics,
            grader=KeywordFakeGrader(),
        )

        self.assertIn("score", scored)
        self.assertEqual(len(scored["decisions"]), 1)
        self.assertEqual(scored["decisions"][0]["tags"], ["safety"])


if __name__ == "__main__":
    unittest.main()
