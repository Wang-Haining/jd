"""
Load and format MedQA-USMLE for the multi-agent experiment.

Source: HuggingFace GBaker/MedQA-USMLE-4-options
"""
import os
from datasets import load_from_disk, load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data", "medqa", "full")


def load_medqa(split: str = "test", n_cases: int | None = None, seed: int = 42) -> list[dict]:
    """
    Load MedQA-USMLE test cases.

    Returns list of dicts with keys:
        case_id, question, options, gold_answer
    """
    if os.path.exists(DATA_DIR):
        ds = load_from_disk(DATA_DIR)[split]
    else:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)

    if n_cases is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n_cases, len(ds))))

    cases = []
    for i, item in enumerate(ds):
        cases.append({
            "case_id": f"medqa_{i}",
            "question": item["question"],
            "options": item["options"],
            "gold_answer": item["answer_idx"],
            "source": "medqa",
        })
    return cases


def format_prompt(case: dict, system_prompt: str) -> list[dict]:
    """Format a MedQA case as chat messages for an agent."""
    opts = "\n".join(f"{k}: {v}" for k, v in case["options"].items())
    user_msg = (
        f"{case['question']}\n\nOptions:\n{opts}\n\n"
        "Answer with the letter (A, B, C, or D) followed by a brief rationale."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
