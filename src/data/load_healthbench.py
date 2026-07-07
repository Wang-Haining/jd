"""
Load and normalize HealthBench Hard for the J-space orchestration study.

Generators must receive only prompt messages. Rubrics are returned for scoring
and audit trails, but runner code should call generation_prompt(case) before
asking any model to produce a response.
"""
from __future__ import annotations

import json
import os
import random
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "healthbench"
DEFAULT_HEALTHBENCH_HARD_PATH = DATA_DIR / "hard.jsonl"
HEALTHBENCH_HARD_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/"
    "hard_2025-05-08-21-00-10.jsonl"
)


def download_healthbench_hard(
    dest_path: str | os.PathLike[str] = DEFAULT_HEALTHBENCH_HARD_PATH,
    source_url: str = HEALTHBENCH_HARD_URL,
) -> Path:
    """Download HealthBench Hard JSONL to the local data directory."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(source_url) as response:
        dest.write_bytes(response.read())
    return dest


def iter_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def normalize_message(message: dict[str, Any]) -> dict[str, str]:
    """Normalize a HealthBench chat message to role/content strings."""
    role = str(message.get("role", "")).strip()
    content = message.get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", ""))))
            else:
                parts.append(str(item))
        content = "\n".join(part for part in parts if part)
    return {"role": role, "content": str(content)}


def normalize_rubric(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize a HealthBench rubric item."""
    criterion = item.get("criterion", item.get("criteria", ""))
    points = item.get("points", item.get("score", 0))
    tags = item.get("tags", [])
    if tags is None:
        tags = []
    return {
        "criterion": str(criterion),
        "points": float(points),
        "tags": [str(tag) for tag in tags],
    }


def normalize_healthbench_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    """Convert a raw HealthBench row into the project case contract."""
    prompt_messages = row.get("prompt") or row.get("messages")
    if not isinstance(prompt_messages, list):
        raise ValueError(f"HealthBench row {index} has no prompt/messages list")

    raw_rubrics = row.get("rubrics") or row.get("criteria") or []
    if not isinstance(raw_rubrics, list):
        raise ValueError(f"HealthBench row {index} has no rubric list")

    prompt_id = str(row.get("prompt_id") or row.get("id") or f"row_{index:04d}")
    case_id = f"healthbench_hard_{prompt_id}"
    tags = row.get("example_tags") or row.get("tags") or []
    if tags is None:
        tags = []

    return {
        "case_id": case_id,
        "prompt_id": prompt_id,
        "prompt_messages": [normalize_message(message) for message in prompt_messages],
        "rubrics": [normalize_rubric(item) for item in raw_rubrics],
        "example_tags": [str(tag) for tag in tags],
        "source": "healthbench_hard",
    }


def load_healthbench_hard(
    n_cases: int | None = 100,
    seed: int = 42,
    path: str | os.PathLike[str] | None = None,
    download_if_missing: bool = True,
    source_url: str = HEALTHBENCH_HARD_URL,
) -> list[dict[str, Any]]:
    """
    Load HealthBench Hard cases.

    Returns dictionaries with:
        case_id, prompt_id, prompt_messages, rubrics, example_tags, source.
    """
    data_path = Path(path) if path is not None else DEFAULT_HEALTHBENCH_HARD_PATH
    if not data_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"HealthBench Hard not found at {data_path}. "
                "Run runs/03_download_data.sbatch or enable download_if_missing."
            )
        download_healthbench_hard(data_path, source_url=source_url)

    rows = iter_jsonl(data_path)
    cases = [normalize_healthbench_row(row, i) for i, row in enumerate(rows)]

    rng = random.Random(seed)
    rng.shuffle(cases)
    if n_cases is not None:
        cases = cases[: min(n_cases, len(cases))]
    return cases


def generation_prompt(case: dict[str, Any]) -> list[dict[str, str]]:
    """Return only the prompt messages that may be shown to generators."""
    return [normalize_message(message) for message in case["prompt_messages"]]


def scoring_payload(case: dict[str, Any]) -> dict[str, Any]:
    """Return the scoring-only payload for a case."""
    return {
        "case_id": case["case_id"],
        "prompt_id": case["prompt_id"],
        "rubrics": case["rubrics"],
        "example_tags": case.get("example_tags", []),
    }
