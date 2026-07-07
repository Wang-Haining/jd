"""JSONL and checkpoint helpers for HealthBench runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL records. Missing files return an empty list."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with file_path.open() as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {file_path}:{line_no}") from exc
    return rows


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON-serializable record to a JSONL file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write an indented JSON file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def completed_case_ids(records: Iterable[dict[str, Any]]) -> set[str]:
    """Return case IDs already present in a run output file."""
    return {str(record["case_id"]) for record in records if record.get("case_id")}


def completed_score_keys(records: Iterable[dict[str, Any]]) -> set[tuple[str, str, str]]:
    """Return completed scoring keys as (case_id, strategy, replicate)."""
    keys: set[tuple[str, str, str]] = set()
    for record in records:
        case_id = record.get("case_id")
        strategy = record.get("strategy")
        replicate = str(record.get("replicate", "primary"))
        if case_id and strategy:
            keys.add((str(case_id), str(strategy), replicate))
    return keys
