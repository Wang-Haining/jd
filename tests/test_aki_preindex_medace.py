import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


HAS_PANDAS = importlib.util.find_spec("pandas") is not None
HAS_OPENAI = importlib.util.find_spec("openai") is not None
HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None
HAS_MEDACE_DEPS = HAS_OPENAI and HAS_PYDANTIC


@unittest.skipUnless(HAS_MEDACE_DEPS, "openai/pydantic not installed")
class AKIPreIndexMedACETest(unittest.TestCase):
    def test_chunk_notes_uses_conservative_prompt_budget(self):
        from src.aki_phenotyping.preindex_medace import build_user_prompt, chunk_notes

        notes = [
            {
                "timestamp": f"2020-01-{idx + 1:02d}",
                "service": "oncology",
                "encounter_id": f"enc_{idx}",
                "note_text": "kidney risk " * 1200,
            }
            for idx in range(8)
        ]

        chunks = chunk_notes(notes, max_chars=25_000)

        self.assertGreater(len(chunks), 1)
        for idx, chunk in enumerate(chunks):
            prompt = build_user_prompt(
                sample_id="sample_1",
                person_id=1,
                demographics={"age_at_index": 70},
                landmark_date="2020-02-01",
                window_start="2019-02-01",
                window_end="2020-02-01",
                notes=chunk,
                chunk_index=idx,
                n_chunks=len(chunks),
                max_note_chars=30_000,
            )
            self.assertLess(len(prompt), 35_000)

    @unittest.skipUnless(HAS_PANDAS, "pandas not installed")
    def test_run_one_patient_records_chunk_errors(self):
        from src.aki_phenotyping.preindex_medace import run_one_patient

        row = SimpleNamespace(person_id=1, ici_index_date="2020-01-01", age_at_index=70)
        notes = [
            {
                "timestamp": "2019-12-01",
                "service": "oncology",
                "encounter_id": "enc_1",
                "note_text": "brief note",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "src.aki_phenotyping.preindex_medace.extract_chunk",
                side_effect=RuntimeError("context exceeded"),
            ):
                result = run_one_patient(
                    client=object(),
                    model="dummy",
                    row=row,
                    notes=notes,
                    out_dir=Path(tmp),
                )

        self.assertEqual(result["error"], "all chunks failed")
        self.assertEqual(result["chunk_errors"][0]["error"], "context exceeded")


if __name__ == "__main__":
    unittest.main()
