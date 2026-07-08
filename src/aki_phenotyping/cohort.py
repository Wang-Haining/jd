"""Balanced cohort construction for post-ICI AKI phenotyping pilots."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - import checked by callers/tests
    pd = None


DEFAULT_POSITIVE_EVIDENCE = ("both",)
DEFAULT_CONTROL_EVIDENCE = ("none",)


@dataclass(frozen=True)
class CohortBuildConfig:
    """Configuration for balanced AKI/no-AKI sampling."""

    positive_evidence: tuple[str, ...] = DEFAULT_POSITIVE_EVIDENCE
    control_evidence: tuple[str, ...] = DEFAULT_CONTROL_EVIDENCE
    min_control_followup_days: int = 180
    n_per_class: int | None = None
    seed: int = 42


def require_pandas() -> None:
    if pd is None:
        raise ModuleNotFoundError("AKI cohort utilities require pandas and pyarrow.")


def load_denominator(path: str | Path):
    """Load the broad ICI denominator with AKI flags."""
    require_pandas()
    return pd.read_csv(path)


def add_aki_labels(frame):
    """Add primary and timing labels from post-ICI AKI fields.

    Primary label:
      - ``aki_any``: any post-ICI AKI evidence in the denominator table.

    Secondary labels:
      - ``aki_3mo``: AKI within 90 days of ICI index.
      - ``aki_6mo``: AKI within 180 days of ICI index.

    Late AKI remains primary-positive but secondary-negative. Controls with
    inadequate follow-up can be excluded later by the sampler.
    """
    require_pandas()
    df = frame.copy()
    evidence = df.get("aki_evidence", "none").fillna("none").astype(str)
    days = pd.to_numeric(df.get("days_to_aki"), errors="coerce")
    df["aki_any"] = evidence.ne("none")
    df["aki_3mo"] = df["aki_any"] & days.le(90)
    df["aki_6mo"] = df["aki_any"] & days.le(180)
    df["aki_late_gt_6mo"] = df["aki_any"] & days.gt(180)
    return df


def add_preindex_note_metrics(frame, data_dir: str | Path, lookback_days: int = 365):
    """Attach pre-index note count/characters for stratified sampling."""
    require_pandas()
    df = frame.copy()
    data_root = Path(data_dir)
    notes_dir = data_root / "clinical_notes_ICI_101625"
    files = sorted(notes_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No clinical note parquet files found in {notes_dir}")

    wanted = set(df["person_id"].astype("int64").tolist())
    index_dates = {
        int(row.person_id): pd.to_datetime(row.ici_index_date, errors="coerce")
        for row in df[["person_id", "ici_index_date"]].itertuples(index=False)
    }

    rows: list[dict] = []
    for path in files:
        notes = pd.read_parquet(path)
        notes.columns = [str(col).upper() for col in notes.columns]
        notes["OMOP_PERSON_ID"] = notes["OMOP_PERSON_ID"].astype("int64")
        notes = notes[notes["OMOP_PERSON_ID"].isin(wanted)]
        if notes.empty:
            continue
        notes["PHYSIOLOGIC_TIME"] = pd.to_datetime(
            notes["PHYSIOLOGIC_TIME"], errors="coerce"
        )
        for note in notes.itertuples(index=False):
            person_id = int(getattr(note, "OMOP_PERSON_ID"))
            index_dt = index_dates.get(person_id)
            note_dt = getattr(note, "PHYSIOLOGIC_TIME")
            if pd.isna(index_dt) or pd.isna(note_dt):
                continue
            start = index_dt - pd.Timedelta(days=lookback_days)
            if not (start <= note_dt <= index_dt):
                continue
            text = str(getattr(note, "REPORT_TEXT", "") or "")
            rows.append(
                {
                    "person_id": person_id,
                    "preindex_n_notes": 1,
                    "preindex_note_chars": len(text),
                    "preindex_first_note": note_dt.date().isoformat(),
                    "preindex_last_note": note_dt.date().isoformat(),
                }
            )

    if not rows:
        df["preindex_n_notes"] = 0
        df["preindex_note_chars"] = 0
        df["preindex_first_note"] = None
        df["preindex_last_note"] = None
        return df

    metrics = pd.DataFrame(rows)
    grouped = (
        metrics.groupby("person_id")
        .agg(
            preindex_n_notes=("preindex_n_notes", "sum"),
            preindex_note_chars=("preindex_note_chars", "sum"),
            preindex_first_note=("preindex_first_note", "min"),
            preindex_last_note=("preindex_last_note", "max"),
        )
        .reset_index()
    )
    df = df.merge(grouped, on="person_id", how="left")
    df["preindex_n_notes"] = df["preindex_n_notes"].fillna(0).astype(int)
    df["preindex_note_chars"] = df["preindex_note_chars"].fillna(0).astype(int)
    return df


def add_sampling_strata(frame):
    """Add coarse strata for age, demographics, regimen, and note volume."""
    require_pandas()
    df = frame.copy()
    age = pd.to_numeric(df.get("age_at_index"), errors="coerce")
    df["age_bin"] = pd.cut(
        age,
        bins=[-1, 49, 64, 74, 200],
        labels=["lt50", "50_64", "65_74", "75plus"],
    ).astype(str)
    df["regimen_bin"] = df.get("ici_regimen", "unknown").fillna("unknown").astype(str)
    notes = pd.to_numeric(df.get("preindex_n_notes", 0), errors="coerce").fillna(0)
    df["note_count_bin"] = pd.cut(
        notes,
        bins=[-1, 0, 5, 20, 100, 1_000_000],
        labels=["0", "1_5", "6_20", "21_100", "gt100"],
    ).astype(str)
    gender = df["gender"] if "gender" in df.columns else pd.Series("unknown", index=df.index)
    race = df["race"] if "race" in df.columns else pd.Series("unknown", index=df.index)
    df["gender_bin"] = _clean_stratum_value(gender).map(_coarse_gender)
    df["race_bin"] = _clean_stratum_value(race).map(_coarse_race)
    df["sample_stratum"] = (
        df["age_bin"]
        + "|"
        + df["gender_bin"]
        + "|"
        + df["race_bin"]
        + "|"
        + df["regimen_bin"]
        + "|"
        + df["note_count_bin"]
    )
    return df


def _clean_stratum_value(values):
    require_pandas()
    return (
        values.fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .replace({"": "unknown", "nan": "unknown", "none": "unknown"})
    )


def _coarse_gender(value: str) -> str:
    if value in {"m", "male", "man"}:
        return "male"
    if value in {"f", "female", "woman"}:
        return "female"
    return "unknown"


def _coarse_race(value: str) -> str:
    if "white" in value or "caucasian" in value:
        return "white"
    if "black" in value or "african" in value:
        return "black"
    if "asian" in value:
        return "asian"
    if value in {"unknown", "nan", "none", "declined", "refused"}:
        return "unknown"
    return "other"


def merge_baseline_features(frame, baseline_features_csv: str | Path):
    """Attach patient-level baseline features before sampling."""
    require_pandas()
    df = frame.copy()
    baseline = pd.read_csv(baseline_features_csv)
    if "person_id" not in baseline.columns:
        raise ValueError("Baseline feature file must contain person_id")
    baseline["person_id"] = pd.to_numeric(baseline["person_id"], errors="coerce").astype("Int64")
    baseline = baseline.dropna(subset=["person_id"]).copy()
    baseline["person_id"] = baseline["person_id"].astype("int64")
    df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["person_id"]).copy()
    df["person_id"] = df["person_id"].astype("int64")
    overlap = [col for col in baseline.columns if col in df.columns and col != "person_id"]
    if overlap:
        df = df.drop(columns=overlap)
    return df.merge(baseline, on="person_id", how="left")


def build_balanced_sample(frame, config: CohortBuildConfig = CohortBuildConfig()):
    """Build a balanced AKI-positive/no-AKI sample with coarse stratum matching."""
    require_pandas()
    df = add_sampling_strata(add_aki_labels(frame))
    df["_aki_source_row"] = range(len(df))
    evidence = df["aki_evidence"].fillna("none").astype(str)
    followup = pd.to_numeric(df.get("followup_days"), errors="coerce")

    cases = df[evidence.isin(config.positive_evidence)].copy()
    controls = df[
        evidence.isin(config.control_evidence)
        & followup.ge(config.min_control_followup_days)
    ].copy()

    rng = config.seed
    case_parts = []
    control_parts = []
    for stratum in sorted(set(cases["sample_stratum"]) | set(controls["sample_stratum"])):
        c = cases[cases["sample_stratum"] == stratum]
        n = controls[controls["sample_stratum"] == stratum]
        take = min(len(c), len(n))
        if take == 0:
            continue
        case_parts.append(c.sample(n=take, random_state=rng))
        control_parts.append(n.sample(n=take, random_state=rng + 1))

    if case_parts:
        sampled_cases = pd.concat(case_parts, ignore_index=True)
        sampled_controls = pd.concat(control_parts, ignore_index=True)
    else:
        take = min(len(cases), len(controls))
        sampled_cases = cases.sample(n=take, random_state=rng)
        sampled_controls = controls.sample(n=take, random_state=rng + 1)

    if config.n_per_class is not None:
        target = min(config.n_per_class, len(cases), len(controls))
        sampled_cases = _top_up_sample(sampled_cases, cases, target, rng + 2)
        sampled_controls = _top_up_sample(sampled_controls, controls, target, rng + 3)
        sampled_cases = sampled_cases.sample(n=target, random_state=rng)
        sampled_controls = sampled_controls.sample(n=target, random_state=rng + 1)

    sampled = pd.concat([sampled_cases, sampled_controls], ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    sampled = sampled.drop(columns=["_aki_source_row"], errors="ignore")
    sampled["label_source"] = sampled["aki_evidence"].map(
        lambda value: "positive_high_confidence" if value in config.positive_evidence else "negative_control"
    )
    return sampled


def _top_up_sample(sampled, pool, target: int, random_state: int):
    """Top up a stratified sample from the remaining pool to reach target size."""
    require_pandas()
    if len(sampled) >= target:
        return sampled
    used = set(sampled.get("_aki_source_row", []))
    remaining = pool[~pool["_aki_source_row"].isin(used)]
    need = min(target - len(sampled), len(remaining))
    if need <= 0:
        return sampled
    top_up = remaining.sample(n=need, random_state=random_state)
    return pd.concat([sampled, top_up], ignore_index=True)


def write_manifest(path: str | Path, *, input_csv: str | Path, config: CohortBuildConfig, frame) -> None:
    """Write a small JSON manifest for a balanced sample."""
    import json

    require_pandas()
    manifest = {
        "input_csv": str(input_csv),
        "positive_evidence": list(config.positive_evidence),
        "control_evidence": list(config.control_evidence),
        "min_control_followup_days": config.min_control_followup_days,
        "n_per_class": config.n_per_class,
        "seed": config.seed,
        "n_rows": int(len(frame)),
        "label_counts": frame["aki_any"].value_counts(dropna=False).to_dict(),
        "aki_3mo_counts": frame["aki_3mo"].value_counts(dropna=False).to_dict(),
        "aki_6mo_counts": frame["aki_6mo"].value_counts(dropna=False).to_dict(),
    }
    Path(path).write_text(json.dumps(manifest, indent=2, default=str))


def parse_evidence_list(raw: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    return tuple(str(part).strip() for part in raw if str(part).strip())
