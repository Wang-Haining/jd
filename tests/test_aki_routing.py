import importlib.util
import unittest


HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None


@unittest.skipUnless(HAS_PYDANTIC, "pydantic not installed")
class AKIRoutingTest(unittest.TestCase):
    def test_parse_handoff_choice(self):
        from src.agents.aki_routing import parse_handoff_choice

        eligible = ["nephrologist", "pharmacist"]

        self.assertEqual(
            parse_handoff_choice("HANDOFF_TO: pharmacist", eligible),
            "pharmacist",
        )
        self.assertIsNone(parse_handoff_choice("HANDOFF_TO: oncologist", eligible))

    def test_select_by_std_low_and_high(self):
        from src.agents.aki_routing import select_by_std

        scores = {
            "nephrologist": {"std": 0.5},
            "pharmacist": {"std": 2.0},
            "hospitalist": {"std": 1.0},
        }

        low, _ = select_by_std("oncologist", scores, mode="low")
        high, _ = select_by_std("oncologist", scores, mode="high")

        self.assertEqual(low, "nephrologist")
        self.assertEqual(high, "pharmacist")


if __name__ == "__main__":
    unittest.main()
