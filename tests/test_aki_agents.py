import importlib.util
import unittest


HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class AKIAgentTest(unittest.TestCase):
    def test_aggregate_predictions(self):
        from src.agents.aki import AKIPrediction, aggregate_predictions

        predictions = [
            AKIPrediction(
                sample_id="sample_1",
                person_id=1,
                agent_id="nephrologist",
                aki_any_probability=0.8,
                aki_3mo_probability=0.4,
                aki_6mo_probability=0.7,
                aki_any_call="yes",
                aki_3mo_call="uncertain",
                aki_6mo_call="yes",
                confidence=0.7,
                rationale="CKD and nephrotoxins.",
            ),
            AKIPrediction(
                sample_id="sample_1",
                person_id=1,
                agent_id="pharmacist",
                aki_any_probability=0.6,
                aki_3mo_probability=0.2,
                aki_6mo_probability=0.5,
                aki_any_call="uncertain",
                aki_3mo_call="no",
                aki_6mo_call="uncertain",
                confidence=0.6,
                rationale="Medication risk.",
            ),
        ]

        aggregate = aggregate_predictions(predictions)

        self.assertAlmostEqual(aggregate.aki_any_probability, 0.7)
        self.assertAlmostEqual(aggregate.aki_3mo_probability, 0.3)
        self.assertEqual(aggregate.aki_any_call, "yes")
        self.assertEqual(aggregate.aki_3mo_call, "no")


if __name__ == "__main__":
    unittest.main()
