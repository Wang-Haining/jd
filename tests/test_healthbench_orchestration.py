import unittest

from src.agents.healthbench import (
    CLINICIAN_ORDER,
    DebateTurn,
    build_debate_turn_messages,
    build_handoff_choice_messages,
    build_initial_debate_messages,
    jlens_next_score,
    next_round_robin_speaker,
    parse_handoff_choice,
)


class HealthBenchOrchestrationTest(unittest.TestCase):
    def test_initial_debate_prompt_keeps_rubrics_out(self):
        messages = build_initial_debate_messages(
            [{"role": "user", "content": "What should I do?"}],
            speaker="generalist",
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("opening turn", messages[-1]["content"])
        self.assertNotIn("rubric", messages[-1]["content"].lower())

    def test_debate_turn_prompt_contains_prior_discussion(self):
        messages = build_debate_turn_messages(
            [{"role": "user", "content": "What now?"}],
            speaker="safety",
            turns=[DebateTurn(speaker="generalist", response_text="Go now.")],
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Go now.", messages[-1]["content"])
        self.assertNotIn("J-lens", messages[-1]["content"])

    def test_handoff_choice_prompt_and_parser(self):
        messages = build_handoff_choice_messages(
            [{"role": "user", "content": "What now?"}],
            current_speaker="generalist",
            turns=[DebateTurn(speaker="generalist", response_text="Go now.")],
            eligible_speakers=["emergency", "safety"],
        )

        self.assertIn("HANDOFF_TO", messages[-1]["content"])
        self.assertEqual(
            parse_handoff_choice("HANDOFF_TO: safety", ["emergency", "safety"]),
            "safety",
        )
        self.assertIsNone(parse_handoff_choice("HANDOFF_TO: generalist", ["safety"]))

    def test_round_robin_skips_current_speaker(self):
        self.assertEqual(
            next_round_robin_speaker("generalist", [s for s in CLINICIAN_ORDER if s != "generalist"]),
            "emergency",
        )

    def test_jlens_next_score_prefers_lower_std(self):
        self.assertGreater(jlens_next_score(0.1), jlens_next_score(1.0))

    def test_clinician_pool_no_more_than_five(self):
        self.assertLessEqual(len(CLINICIAN_ORDER), 5)


if __name__ == "__main__":
    unittest.main()
