import importlib.util
import unittest


HAS_PANDAS = importlib.util.find_spec("pandas") is not None


@unittest.skipUnless(HAS_PANDAS, "pandas not installed")
class AKIBaselineFeaturesTest(unittest.TestCase):
    def test_normalize_and_prefix_matching(self):
        from src.aki_phenotyping.baseline_features import (
            CHARLSON_COMORBIDITIES,
            CHARLSON_CODESETS,
            matches_codeset,
            normalize_icd_code,
        )

        self.assertEqual(len(CHARLSON_COMORBIDITIES), 19)
        self.assertEqual(normalize_icd_code("E11.22"), "E1122")
        self.assertEqual(normalize_icd_code("foo^^N18.6"), "N186")
        self.assertTrue(matches_codeset("N18.6", CHARLSON_CODESETS["Renal_Disease_Severe"]))

    def test_charlson_hiv_aids_and_hierarchy(self):
        import pandas as pd

        from src.aki_phenotyping.baseline_features import build_charlson_flags

        cohort = pd.DataFrame(
            [
                {"person_id": 1, "ici_index_date": "2020-01-01"},
                {"person_id": 2, "ici_index_date": "2020-01-01"},
                {"person_id": 3, "ici_index_date": "2020-01-01"},
            ]
        )
        dx = pd.DataFrame(
            [
                {"person_id": 1, "condition_source_value": "B20", "condition_start_date": "2019-04-01"},
                {"person_id": 1, "condition_source_value": "B59", "condition_start_date": "2019-05-01"},
                {"person_id": 2, "condition_source_value": "N18.3", "condition_start_date": "2019-06-01"},
                {"person_id": 2, "condition_source_value": "N18.6", "condition_start_date": "2019-07-01"},
                {"person_id": 3, "condition_source_value": "E11.9", "condition_start_date": "2018-01-01"},
            ]
        )

        flags = build_charlson_flags(dx, cohort, lookback_days=365).set_index("person_id")

        self.assertEqual(flags.loc[1, "AIDS"], 1)
        self.assertEqual(flags.loc[1, "HIV"], 0)
        self.assertEqual(flags.loc[2, "Renal_Disease_Severe"], 1)
        self.assertEqual(flags.loc[2, "Renal_Disease_Mild_Moderate"], 0)
        self.assertEqual(flags.loc[3, "Diabetes_without_Chronic_Complications"], 0)

        all_preindex = build_charlson_flags(dx, cohort).set_index("person_id")
        self.assertEqual(all_preindex.loc[3, "Diabetes_without_Chronic_Complications"], 1)

    def test_medications_and_prompt_format(self):
        import pandas as pd

        from src.aki_phenotyping.baseline_features import (
            build_medication_features,
            format_baseline_features,
        )

        cohort = pd.DataFrame([{"person_id": 1, "ici_index_date": "2020-01-01"}])
        meds = pd.DataFrame(
            [
                {"person_id": 1, "drug_source_value": "Ibuprofen", "drug_exposure_start_date": "2019-06-01"},
                {"person_id": 1, "drug_source_value": " ibuprofen ", "drug_exposure_start_date": "2019-07-01"},
                {"person_id": 1, "drug_source_value": "Lisinopril", "drug_exposure_start_date": "2019-08-01"},
                {"person_id": 1, "drug_source_value": "Vancomycin", "drug_exposure_start_date": "2018-01-01"},
            ]
        )

        features = build_medication_features(meds, cohort, lookback_days=365).iloc[0].to_dict()
        features.update(
            {
                "age_at_index": 70,
                "gender": "Female",
                "race": "White",
                "ethnicity": "Not Hispanic",
                "Renal_Disease_Severe": 1,
                "charlson_comorbidity_count": 1,
            }
        )
        rendered = format_baseline_features(features)

        self.assertEqual(features["baseline_drug_count"], 2)
        self.assertIn("Ibuprofen", features["baseline_drugs"])
        self.assertIn("Lisinopril", rendered)
        self.assertIn("Renal_Disease_Severe", rendered)


if __name__ == "__main__":
    unittest.main()
