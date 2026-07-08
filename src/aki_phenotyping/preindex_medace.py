"""Pre-index note extraction for post-ICI AKI prediction."""
from __future__ import annotations

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from .baseline_features import format_baseline_features
from .medace_schema import AKIPreIndexSummary, Source

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None


logger = logging.getLogger("aki_preindex_medace")

DEFAULT_TEMPERATURE = 0.1
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_MAX_TOKENS = 6000
DEFAULT_CHUNK_NOTE_CHARS = 45_000
DEFAULT_PROMPT_NOTE_CHARS = 50_000
MAX_SINGLE_NOTE_CHARS = 10_000


def require_pandas() -> None:
    if pd is None:
        raise ModuleNotFoundError("Pre-index MedACE extraction requires pandas/pyarrow.")


def build_system_prompt() -> str:
    """Build a compact schema-as-prompt instruction."""
    schema = json.dumps(AKIPreIndexSummary.model_json_schema(), indent=2)
    return f"""You are a clinical chart abstractor for an ICI-treated cancer cohort.

Task: extract ONLY information documented at or before the ICI index date that
could help predict future post-ICI acute kidney injury (AKI). AKI is defined by
KDIGO stage 1 or higher: serum creatinine increase >=0.3 mg/dL or >=1.5x
baseline. Do not decide whether the future outcome occurred. Do not include
facts after the landmark/index date.

Use concise, auditable facts. Each Fact requires value, evidence, confidence,
observed_date when available, and provenance when the source note is known.
Prefer dates in YYYY-MM-DD. Drug names should be generic when possible.

If the notes contain post-index events despite the requested window, ignore
those facts and set extraction_quality.leakage_check to
"possible_post_index_content". Otherwise set it to "pre_index_only".

Return one JSON object matching this schema, with no markdown and no extra keys:

{schema}
"""


SYSTEM_PROMPT = build_system_prompt()


def build_user_prompt(
    *,
    sample_id: str,
    person_id: int,
    demographics: dict[str, Any],
    landmark_date: str,
    window_start: str,
    window_end: str,
    notes: list[dict[str, Any]],
    baseline_features: str | None = None,
    chunk_index: int | None = None,
    n_chunks: int | None = None,
    max_note_chars: int = DEFAULT_PROMPT_NOTE_CHARS,
) -> str:
    """Pack pre-index notes into a prompt."""
    parts = [
        f"sample_id: {sample_id}",
        f"person_id: {person_id}",
        f"landmark_date: {landmark_date}",
        f"allowed_window: {window_start} to {window_end}",
        "Do not extract any clinical fact after landmark_date.",
    ]
    if n_chunks and n_chunks > 1:
        parts.append(
            f"note_chunk: {chunk_index + 1} of {n_chunks}. Extract from this chunk only."
        )
    if demographics:
        demo = "\n".join(f"- {key}: {value}" for key, value in demographics.items())
        parts.append("demographics:\n" + demo)
    if baseline_features:
        parts.append("structured_pre_index_baseline_features:\n" + baseline_features)

    total = sum(len(part) for part in parts)
    packed = 0
    parts.append("clinical_notes:")
    for idx, note in enumerate(sorted(notes, key=lambda x: x.get("timestamp", "")), start=1):
        text = str(note.get("note_text", "") or "").strip()
        if not text:
            continue
        if len(text) > MAX_SINGLE_NOTE_CHARS:
            text = text[:MAX_SINGLE_NOTE_CHARS] + "\n[NOTE TRUNCATED]"
        entry = (
            f"\n=== NOTE {idx} | note_id: {note.get('encounter_id') or 'unknown'} "
            f"| note_date: {note.get('timestamp') or 'unknown'} "
            f"| note_type: {note.get('service') or 'unknown'} ===\n{text}"
        )
        if total + len(entry) > max_note_chars:
            parts.append(f"\n[TRUNCATED: {len(notes) - packed} notes omitted]")
            break
        parts.append(entry)
        total += len(entry)
        packed += 1
    parts.append(f"\nnotes_packed: {packed}/{len(notes)}")
    return "\n".join(parts)


def chunk_notes(
    notes: list[dict[str, Any]],
    max_chars: int = DEFAULT_CHUNK_NOTE_CHARS,
) -> list[list[dict[str, Any]]]:
    """Split notes into chronological chunks."""
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for note in sorted(notes, key=lambda x: x.get("timestamp", "")):
        text_len = min(len(str(note.get("note_text", "") or "")), MAX_SINGLE_NOTE_CHARS) + 250
        if current and current_chars + text_len > max_chars:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(note)
        current_chars += text_len
    if current:
        chunks.append(current)
    return chunks


def preload_notes(data_dir: str | Path, pids: list[int], cohort):
    """Load notes once and keep notes in the pre-index window."""
    require_pandas()
    data_root = Path(data_dir)
    notes_dir = data_root / "clinical_notes_ICI_101625"
    files = sorted(notes_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No clinical note parquet files found in {notes_dir}")

    wanted = {int(pid) for pid in pids}
    index_dates = {
        int(row.person_id): pd.to_datetime(row.ici_index_date, errors="coerce")
        for row in cohort[["person_id", "ici_index_date"]].itertuples(index=False)
        if int(row.person_id) in wanted
    }
    out: dict[int, list[dict[str, Any]]] = {int(pid): [] for pid in wanted}
    for path in files:
        frame = pd.read_parquet(path)
        frame.columns = [str(col).upper() for col in frame.columns]
        frame["OMOP_PERSON_ID"] = frame["OMOP_PERSON_ID"].astype("int64")
        frame = frame[frame["OMOP_PERSON_ID"].isin(wanted)]
        if frame.empty:
            continue
        frame["PHYSIOLOGIC_TIME"] = pd.to_datetime(frame["PHYSIOLOGIC_TIME"], errors="coerce")
        for note in frame.itertuples(index=False):
            pid = int(getattr(note, "OMOP_PERSON_ID"))
            note_dt = getattr(note, "PHYSIOLOGIC_TIME")
            index_dt = index_dates.get(pid)
            if pd.isna(note_dt) or pd.isna(index_dt):
                continue
            start = index_dt - pd.Timedelta(days=365)
            if not (start <= note_dt <= index_dt):
                continue
            out[pid].append(
                {
                    "timestamp": note_dt.date().isoformat(),
                    "service": str(getattr(note, "SERVICE_NAME", "") or ""),
                    "encounter_id": str(getattr(note, "ENCOUNTER_ID", "") or ""),
                    "note_text": str(getattr(note, "REPORT_TEXT", "") or ""),
                }
            )
    for notes in out.values():
        notes.sort(key=lambda item: item.get("timestamp", ""))
    return out


def _call_with_backoff(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    *,
    response_format: dict | None,
    max_tries: int = 5,
):
    delay = 5.0
    for attempt in range(max_tries):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": DEFAULT_TEMPERATURE,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "extra_body": {"reasoning_effort": DEFAULT_REASONING_EFFORT},
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            transient = any(
                token in str(exc).lower()
                for token in ["429", "timeout", "502", "503", "504", "overloaded"]
            )
            if attempt < max_tries - 1 and transient:
                time.sleep(delay + random.uniform(0, delay * 0.2))
                delay = min(delay * 2, 90)
                continue
            raise


def _strip_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def extract_chunk(
    *,
    client: OpenAI,
    model: str,
    user_prompt: str,
    schema_mode: str = "json_schema",
) -> tuple[AKIPreIndexSummary | None, dict[str, Any]]:
    """Extract one chunk with validation/retry."""
    response_format = None
    if schema_mode == "json_schema":
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "aki_preindex_summary",
                "schema": AKIPreIndexSummary.model_json_schema(),
            },
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt_tokens = completion_tokens = 0
    last_error = None
    for attempt in range(4):
        response = _call_with_backoff(
            client,
            model,
            messages,
            response_format=response_format,
        )
        usage = response.usage
        prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        raw_text = _strip_json(response.choices[0].message.content or "")
        try:
            parsed = json.loads(raw_text)
            return AKIPreIndexSummary.model_validate(parsed), {
                "attempts": attempt + 1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "raw_text": raw_text,
            }
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
            messages.append({"role": "assistant", "content": raw_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Fix the JSON so it exactly matches the schema. "
                        "Return only the corrected JSON object. Error:\n"
                        f"{last_error[:3000]}"
                    ),
                }
            )
    return None, {
        "attempts": 4,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "error": last_error,
    }


def merge_summaries(summaries: list[AKIPreIndexSummary]) -> AKIPreIndexSummary:
    """Merge chunk-level summaries conservatively."""
    if len(summaries) == 1:
        return summaries[0]
    base = summaries[0]
    merged = base.model_dump()
    merged["narrative_summary"] = "\n\n".join(s.narrative_summary for s in summaries)
    merged["sources"] = _merge_sources([source for s in summaries for source in s.sources])
    for block_name in [
        "cancer_ici_context",
        "kidney_baseline",
        "comorbidity_context",
        "medication_risk",
        "acute_illness_context",
    ]:
        merged[block_name] = _merge_block([getattr(s, block_name).model_dump() for s in summaries])
    missing = []
    leakage = "pre_index_only"
    for summary in summaries:
        missing.extend(summary.extraction_quality.key_missing_information)
        if summary.extraction_quality.leakage_check == "possible_post_index_content":
            leakage = "possible_post_index_content"
    merged["extraction_quality"] = {
        "note_coverage_summary": "Merged from multiple pre-index note chunks.",
        "key_missing_information": sorted(set(missing)),
        "leakage_check": leakage,
    }
    return AKIPreIndexSummary.model_validate(merged)


def _merge_sources(sources: list[Source]) -> list[dict[str, str | None]]:
    seen = set()
    out = []
    for source in sources:
        item = source.model_dump()
        key = (item.get("note_id"), item.get("note_date"), item.get("note_type"))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _merge_block(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    assertion_rank = {"present": 3, "possible": 2, "absent": 1, "not_documented": 0}
    merged = dict(blocks[0])
    merged["assertion"] = max(
        (b.get("assertion", "not_documented") for b in blocks),
        key=lambda value: assertion_rank.get(value, 0),
    )
    for block in blocks[1:]:
        for key, value in block.items():
            if key == "assertion" or value in (None, [], {}):
                continue
            if isinstance(value, list):
                merged.setdefault(key, [])
                merged[key].extend(value)
            elif merged.get(key) in (None, [], {}):
                merged[key] = value
    return merged


def run_one_patient(
    *,
    client: OpenAI,
    model: str,
    row,
    notes: list[dict[str, Any]],
    out_dir: Path,
    schema_mode: str = "json_schema",
) -> dict[str, Any]:
    """Run pre-index extraction for one patient."""
    require_pandas()
    person_id = int(row.person_id)
    sample_id = f"sample_{person_id}"
    index_dt = pd.to_datetime(row.ici_index_date)
    window_start = (index_dt - pd.Timedelta(days=365)).date().isoformat()
    window_end = index_dt.date().isoformat()
    landmark = index_dt.date().isoformat()
    out_path = out_dir / f"{sample_id}.json"

    if not notes:
        return {
            "sample_id": sample_id,
            "person_id": person_id,
            "error": "no pre-index notes",
            "n_notes": 0,
            "window_start": window_start,
            "window_end": window_end,
        }

    chunks = chunk_notes(notes)
    summaries: list[AKIPreIndexSummary] = []
    chunk_errors: list[dict[str, Any]] = []
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt_tokens = completion_tokens = attempts = 0
    t0 = time.perf_counter()
    for idx, chunk in enumerate(chunks):
        prompt = build_user_prompt(
            sample_id=sample_id,
            person_id=person_id,
            demographics={
                "age_at_index": getattr(row, "age_at_index", None),
                "ici_regimen": getattr(row, "ici_regimen", None),
            },
            baseline_features=format_baseline_features(row),
            landmark_date=landmark,
            window_start=window_start,
            window_end=window_end,
            notes=chunk,
            chunk_index=idx,
            n_chunks=len(chunks),
        )
        try:
            summary, meta = extract_chunk(
                client=client,
                model=model,
                user_prompt=prompt,
                schema_mode=schema_mode,
            )
        except Exception as exc:
            logger.exception("Chunk extraction failed for %s chunk %s", sample_id, idx + 1)
            summary = None
            meta = {"attempts": 0, "error": str(exc)}
            chunk_errors.append(
                {
                    "chunk_index": idx + 1,
                    "n_chunks": len(chunks),
                    "error": str(exc),
                }
            )
        prompt_tokens += int(meta.get("prompt_tokens", 0))
        completion_tokens += int(meta.get("completion_tokens", 0))
        attempts += int(meta.get("attempts", 0))
        (raw_dir / f"{sample_id}_chunk{idx + 1}of{len(chunks)}.json").write_text(
            meta.get("raw_text", "") or meta.get("error", "")
        )
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        return {
            "sample_id": sample_id,
            "person_id": person_id,
            "error": "all chunks failed",
            "n_notes": len(notes),
            "n_chunks": len(chunks),
            "attempts": attempts,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "chunk_errors": chunk_errors,
        }

    merged = merge_summaries(summaries)
    out_path.write_text(merged.model_dump_json(indent=2, exclude_none=True))
    return {
        "sample_id": sample_id,
        "person_id": person_id,
        "n_notes": len(notes),
        "n_chunks": len(chunks),
        "chunks_ok": len(summaries),
        "chunks_failed": len(chunk_errors),
        "window_start": window_start,
        "window_end": window_end,
        "attempts": attempts,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "out_json": str(out_path),
        "chunk_errors": chunk_errors,
    }


def run_batch(
    *,
    cohort_csv: str | Path,
    data_dir: str | Path,
    out_dir: str | Path,
    base_url: str,
    model: str,
    api_key: str,
    limit: int | None = None,
    workers: int = 1,
    overwrite: bool = False,
    schema_mode: str = "json_schema",
) -> list[dict[str, Any]]:
    """Run a batch of pre-index MedACE extractions."""
    require_pandas()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(cohort_csv)
    if limit is not None:
        cohort = cohort.head(limit)
    pids = [int(pid) for pid in cohort["person_id"].tolist()]
    if not overwrite:
        pids = [pid for pid in pids if not (out / f"sample_{pid}.json").exists()]
        cohort = cohort[cohort["person_id"].astype(int).isin(pids)].copy()
    notes_by_pid = preload_notes(data_dir, pids, cohort)
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=300.0)

    rows = list(cohort.itertuples(index=False))
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for row in rows:
            results.append(
                run_one_patient(
                    client=client,
                    model=model,
                    row=row,
                    notes=notes_by_pid.get(int(row.person_id), []),
                    out_dir=out,
                    schema_mode=schema_mode,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    run_one_patient,
                    client=client,
                    model=model,
                    row=row,
                    notes=notes_by_pid.get(int(row.person_id), []),
                    out_dir=out,
                    schema_mode=schema_mode,
                )
                for row in rows
            ]
            for future in as_completed(futures):
                results.append(future.result())
    (out / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    return results
