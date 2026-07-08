import importlib.util
import unittest


HAS_PANDAS = importlib.util.find_spec("pandas") is not None


@unittest.skipUnless(HAS_PANDAS, "pandas not installed")
class AKICohortTest(unittest.TestCase):
    def test_labels_and_balanced_sample(self):
        import pandas as pd

        from src.aki_phenotyping.cohort import CohortBuildConfig, build_balanced_sample

        frame = pd.DataFrame(
            [
                {
                    "person_id": 1,
                    "ici_index_date": "2020-01-01",
                    "aki_evidence": "both",
                    "days_to_aki": 30,
                    "followup_days": 200,
                    "age_at_index": 60,
                    "ici_regimen": "pd1_mono",
                    "preindex_n_notes": 10,
                },
                {
                    "person_id": 2,
                    "ici_index_date": "2020-01-01",
                    "aki_evidence": "both",
                    "days_to_aki": 120,
                    "followup_days": 200,
                    "age_at_index": 60,
                    "ici_regimen": "pd1_mono",
                    "preindex_n_notes": 10,
                },
                {
                    "person_id": 3,
                    "ici_index_date": "2020-01-01",
                    "aki_evidence": "none",
                    "days_to_aki": None,
                    "followup_days": 220,
                    "age_at_index": 61,
                    "ici_regimen": "pd1_mono",
                    "preindex_n_notes": 12,
                },
                {
                    "person_id": 4,
                    "ici_index_date": "2020-01-01",
                    "aki_evidence": "none",
                    "days_to_aki": None,
                    "followup_days": 230,
                    "age_at_index": 62,
                    "ici_regimen": "pd1_mono",
                    "preindex_n_notes": 8,
                },
            ]
        )

        sample = build_balanced_sample(
            frame,
            CohortBuildConfig(n_per_class=2, min_control_followup_days=180),
        )

        self.assertEqual(len(sample), 4)
        self.assertEqual(sample["aki_any"].sum(), 2)
        self.assertEqual(sample["aki_3mo"].sum(), 1)
        self.assertEqual(sample["aki_6mo"].sum(), 2)


if __name__ == "__main__":
    unittest.main()
