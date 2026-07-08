"""Pre-index structured baseline features for ICI-AKI prediction.

This module intentionally stays narrow: demographics, Glasheen/NCI Charlson
binary comorbidity flags, and prior-year medication names. Charlson flags use
all diagnosis history before the ICI index by default; medications use the
prior year by default. It produces a
mergeable patient-level CSV for the tumor-board-style AKI prediction pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - import checked by callers/tests
    pd = None


CHARLSON_COMORBIDITIES = [
    "HIV",
    "AIDS",
    "Cerebrovascular_Disease",
    "Congestive_Heart_Failure",
    "Myocardial_Infarction",
    "Peripheral_Vascular_Disease",
    "Chronic_Pulmonary_Disease",
    "Dementia",
    "Liver_Disease_Mild",
    "Liver_Disease_Moderate_Severe",
    "Malignancy",
    "Metastatic_Solid_Tumor",
    "Peptic_Ulcer_Disease",
    "Renal_Disease_Mild_Moderate",
    "Renal_Disease_Severe",
    "Rheumatic_Disease",
    "Hemiplegia_Paraplegia",
    "Diabetes_with_Chronic_Complications",
    "Diabetes_without_Chronic_Complications",
]


CHARLSON_CODESETS: dict[str, dict[str, list[str]]] = {
    "Myocardial_Infarction": {
        "9": ["410", "412"],
        "10": ["I21", "I22", "I252"],
    },
    "Congestive_Heart_Failure": {
        "9": [
            "39891",
            "40201",
            "40211",
            "40291",
            "40401",
            "40403",
            "40411",
            "40413",
            "40491",
            "40493",
            "4254",
            "4255",
            "4256",
            "4257",
            "4258",
            "4259",
            "428",
        ],
        "10": [
            "I099",
            "I110",
            "I130",
            "I132",
            "I255",
            "I420",
            "I425",
            "I426",
            "I427",
            "I428",
            "I429",
            "I43",
            "I50",
            "P290",
        ],
    },
    "Peripheral_Vascular_Disease": {
        "9": [
            "0930",
            "440",
            "441",
            "4431",
            "4432",
            "4433",
            "4434",
            "4435",
            "4436",
            "4437",
            "4438",
            "4439",
            "4471",
            "5571",
            "5579",
            "V434",
        ],
        "10": [
            "I70",
            "I71",
            "I731",
            "I738",
            "I739",
            "I771",
            "I790",
            "I792",
            "K551",
            "K558",
            "K559",
            "Z958",
            "Z959",
        ],
    },
    "Cerebrovascular_Disease": {
        "9": ["36234", "430", "431", "432", "433", "434", "435", "436", "437", "438"],
        "10": ["G45", "G46", "H340", "I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69"],
    },
    "Dementia": {
        "9": ["290", "2941", "3312"],
        "10": ["F00", "F01", "F02", "F03", "F051", "G30", "G311"],
    },
    "Chronic_Pulmonary_Disease": {
        "9": [
            "4168",
            "4169",
            "490",
            "491",
            "492",
            "493",
            "494",
            "495",
            "496",
            "497",
            "498",
            "499",
            "500",
            "501",
            "502",
            "503",
            "504",
            "505",
            "5064",
            "5081",
            "5088",
        ],
        "10": [
            "I278",
            "I279",
            "J40",
            "J41",
            "J42",
            "J43",
            "J44",
            "J45",
            "J46",
            "J47",
            "J60",
            "J61",
            "J62",
            "J63",
            "J64",
            "J65",
            "J66",
            "J67",
            "J684",
            "J701",
            "J703",
        ],
    },
    "Rheumatic_Disease": {
        "9": ["4465", "7100", "7101", "7102", "7103", "7104", "7140", "7141", "7142", "7148", "725"],
        "10": ["M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"],
    },
    "Peptic_Ulcer_Disease": {
        "9": ["531", "532", "533", "534"],
        "10": ["K25", "K26", "K27", "K28"],
    },
    "Liver_Disease_Mild": {
        "9": ["07022", "07023", "07032", "07033", "07044", "07054", "0706", "0709", "570", "571", "5733", "5734", "5738", "5739", "V427"],
        "10": ["B18", "K700", "K701", "K702", "K703", "K709", "K713", "K714", "K715", "K717", "K73", "K74", "K760", "K762", "K763", "K764", "K768", "K769", "Z944"],
    },
    "Liver_Disease_Moderate_Severe": {
        "9": ["4560", "4561", "4562", "5722", "5723", "5724", "5725", "5726", "5727", "5728"],
        "10": ["I850", "I859", "I864", "I982", "K704", "K711", "K721", "K729", "K765", "K766", "K767"],
    },
    "Diabetes_without_Chronic_Complications": {
        "9": ["2500", "2501", "2502", "2503", "2508", "2509"],
        "10": ["E100", "E101", "E106", "E108", "E109", "E110", "E111", "E116", "E118", "E119", "E130", "E131", "E136", "E138", "E139"],
    },
    "Diabetes_with_Chronic_Complications": {
        "9": ["2504", "2505", "2506", "2507"],
        "10": ["E102", "E103", "E104", "E105", "E107", "E112", "E113", "E114", "E115", "E117", "E132", "E133", "E134", "E135", "E137"],
    },
    "Hemiplegia_Paraplegia": {
        "9": ["3341", "342", "343", "3440", "3441", "3442", "3443", "3444", "3445", "3446", "3449"],
        "10": ["G041", "G114", "G801", "G802", "G81", "G82", "G830", "G831", "G832", "G833", "G834", "G839"],
    },
    "Renal_Disease_Mild_Moderate": {
        "9": ["40300", "40310", "40390", "40400", "40401", "40410", "40411", "40490", "40491", "582", "583", "5851", "5852", "5853", "5854", "5859", "V420"],
        "10": ["I129", "I130", "I1310", "N032", "N033", "N034", "N035", "N036", "N037", "N052", "N053", "N054", "N055", "N056", "N057", "N181", "N182", "N183", "N184", "N189", "Z940"],
    },
    "Renal_Disease_Severe": {
        "9": ["40301", "40311", "40391", "40402", "40403", "40412", "40413", "40492", "40493", "5855", "5856", "586", "5880", "V451", "V56"],
        "10": ["I120", "I1311", "I132", "N185", "N186", "N19", "N250", "Z49", "Z992"],
    },
    "HIV": {
        "9": ["042", "043", "044"],
        "10": ["B20", "B21", "B22", "B24"],
    },
    "Metastatic_Solid_Tumor": {
        "9": ["196", "197", "198", "1990"],
        "10": ["C77", "C78", "C79", "C800", "C802"],
    },
    "Malignancy": {
        "9": ["14", "15", "16", "170", "171", "172", "174", "175", "176", "179", "18", "190", "191", "192", "193", "194", "195", "1991", "200", "201", "202", "203", "204", "205", "206", "207", "208", "2386"],
        "10": ["C0", "C1", "C2", "C30", "C31", "C32", "C33", "C34", "C37", "C38", "C39", "C40", "C41", "C43", "C45", "C46", "C47", "C48", "C49", "C50", "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C60", "C61", "C62", "C63", "C76", "C801", "C81", "C82", "C83", "C84", "C85", "C88", "C9"],
    },
}


AIDS_OI_CODESET = {
    "9": [
        "112",
        "180",
        "114",
        "1175",
        "0074",
        "0785",
        "3483",
        "054",
        "115",
        "0072",
        "176",
        "200",
        "201",
        "202",
        "203",
        "204",
        "205",
        "206",
        "207",
        "208",
        "209",
        "031",
        "010",
        "011",
        "012",
        "013",
        "014",
        "015",
        "016",
        "017",
        "018",
        "1363",
        "V1261",
        "0463",
        "0031",
        "130",
        "7994",
    ],
    "10": [
        "B37",
        "C53",
        "B38",
        "B45",
        "A072",
        "B25",
        "G934",
        "B00",
        "B39",
        "A073",
        "C46",
        "C81",
        "C82",
        "C83",
        "C84",
        "C85",
        "C86",
        "C87",
        "C88",
        "C9",
        "C90",
        "C91",
        "C92",
        "C93",
        "C94",
        "C95",
        "C96",
        "A31",
        "A15",
        "A16",
        "A17",
        "A18",
        "A19",
        "B59",
        "Z8701",
        "A812",
        "A021",
        "B58",
        "R64",
    ],
}


HIERARCHY_PAIRS = [
    ("Hemiplegia_Paraplegia", "Cerebrovascular_Disease"),
    ("Liver_Disease_Moderate_Severe", "Liver_Disease_Mild"),
    ("Diabetes_with_Chronic_Complications", "Diabetes_without_Chronic_Complications"),
    ("Renal_Disease_Severe", "Renal_Disease_Mild_Moderate"),
    ("Metastatic_Solid_Tumor", "Malignancy"),
    ("AIDS", "HIV"),
]


PERSON_ALIASES = ("person_id", "omop_person_id", "patid", "patient_id")
INDEX_DATE_ALIASES = ("ici_index_date", "index_date", "earliest_start_date")
DX_CODE_ALIASES = (
    "condition_source_value",
    "dx_code",
    "diagnosis_code",
    "diagnosis_source_value",
    "concept_code",
    "condition_concept_code",
)
DX_DATE_ALIASES = ("condition_start_date", "dx_date", "diagnosis_date", "start_date", "date")
DRUG_NAME_ALIASES = (
    "standard_concept_name",
    "concept_name",
    "drug_name",
    "drug_source_value",
    "drug_concept_name",
    "medication_name",
    "generic_name",
)
DRUG_DATE_ALIASES = (
    "drug_exposure_start_date",
    "drug_era_start_date",
    "start_date",
    "order_date",
    "fill_date",
    "date",
)
DEMOGRAPHIC_COLUMNS = [
    "age_at_index",
    "age",
    "gender",
    "gender_source_value",
    "gender_concept_id",
    "race",
    "race_source_value",
    "ethnicity",
    "ethnicity_source_value",
    "year_of_birth",
]


def require_pandas() -> None:
    if pd is None:
        raise ModuleNotFoundError("AKI baseline feature ETL requires pandas/pyarrow.")


def normalize_icd_code(value: Any) -> str:
    """Normalize an ICD code for prefix matching."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    if "^^" in text:
        parts = [part for part in text.split("^^") if part]
        text = parts[-1] if parts else text
    keep = []
    for char in text.upper():
        if char.isalnum():
            keep.append(char)
    return "".join(keep)


def _flatten_codes(codeset: dict[str, list[str]]) -> tuple[str, ...]:
    prefixes = {normalize_icd_code(code) for codes in codeset.values() for code in codes}
    return tuple(sorted(prefix for prefix in prefixes if prefix))


def matches_codeset(code: Any, codeset: dict[str, list[str]]) -> bool:
    """Return true when a normalized ICD code starts with any codeset prefix."""
    normalized = normalize_icd_code(code)
    if not normalized:
        return False
    return any(normalized.startswith(prefix) for prefix in _flatten_codes(codeset))


def _pick_column(columns: Iterable[str], aliases: Iterable[str], *, required: bool = True) -> str | None:
    low_to_real = {str(col).lower(): str(col) for col in columns}
    for alias in aliases:
        if alias.lower() in low_to_real:
            return low_to_real[alias.lower()]
    if required:
        raise ValueError(f"Could not find any of columns: {', '.join(aliases)}")
    return None


def _read_table(path: str | Path, columns: list[str] | None = None):
    require_pandas()
    p = Path(path)
    wanted = {col.lower() for col in columns or []}

    def parquet_columns(sample: Path) -> list[str]:
        import pyarrow.parquet as pq

        schema = pq.read_schema(sample)
        return [str(name) for name in schema.names]

    if p.is_dir():
        if columns is None:
            return pd.read_parquet(p)
        first = next((item for item in sorted(p.glob("*.parquet")) if not item.name.startswith(".")), None)
        if first is None:
            raise FileNotFoundError(f"No parquet files under {p}")
        available = parquet_columns(first)
        low_to_real = {str(col).lower(): str(col) for col in available}
        keep = [low_to_real[col] for col in wanted if col in low_to_real]
        return pd.read_parquet(p, columns=keep or None)
    if p.suffix.lower() == ".csv":
        if columns is None:
            return pd.read_csv(p)
        return pd.read_csv(p, usecols=lambda col: str(col).lower() in wanted)
    if p.suffix.lower() in {".parquet", ".pq"}:
        if columns is None:
            return pd.read_parquet(p)
        available = parquet_columns(p)
        low_to_real = {str(col).lower(): str(col) for col in available}
        keep = [low_to_real[col] for col in wanted if col in low_to_real]
        return pd.read_parquet(p, columns=keep or None)
    raise ValueError(f"Unsupported table path: {p}")


def _find_omop_table(data_dir: str | Path, table: str) -> Path:
    root = Path(data_dir)
    direct = root / table
    if direct.exists():
        return direct
    nested = root / "structured_data_fixed" / "RDRP-6335_Results_Parquet" / table
    if nested.exists():
        return nested
    matches = sorted(root.glob(f"**/{table}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find OMOP table {table!r} under {root}")


def _prepare_patient_index(cohort):
    require_pandas()
    person_col = _pick_column(cohort.columns, PERSON_ALIASES)
    index_col = _pick_column(cohort.columns, INDEX_DATE_ALIASES)
    out = cohort[[person_col, index_col]].copy()
    out.columns = ["person_id", "ici_index_date"]
    out["person_id"] = pd.to_numeric(out["person_id"], errors="coerce").astype("Int64")
    out["ici_index_date"] = pd.to_datetime(out["ici_index_date"], errors="coerce")
    out = out.dropna(subset=["person_id", "ici_index_date"])
    out["person_id"] = out["person_id"].astype("int64")
    return out.drop_duplicates("person_id")


def build_demographic_features(cohort, person_frame=None):
    """Build age/gender/race/ethnicity rows from denominator plus OMOP person."""
    require_pandas()
    index = _prepare_patient_index(cohort)
    out = index[["person_id"]].copy()

    for col in DEMOGRAPHIC_COLUMNS:
        if col in cohort.columns and col not in out.columns:
            out = out.merge(cohort[["person_id", col]], on="person_id", how="left")

    if person_frame is not None:
        person = person_frame.copy()
        person.columns = [str(col).lower() for col in person.columns]
        if "person_id" in person.columns:
            keep = ["person_id"] + [col for col in DEMOGRAPHIC_COLUMNS if col in person.columns]
            keep = list(dict.fromkeys(keep))
            out = out.merge(person[keep], on="person_id", how="left", suffixes=("", "_person"))

    if "age_at_index" not in out.columns and "age" in out.columns:
        out["age_at_index"] = out["age"]
    if "age_at_index" not in out.columns and "year_of_birth" in out.columns:
        idx = index.set_index("person_id")["ici_index_date"]
        years = pd.to_numeric(out["year_of_birth"], errors="coerce")
        out["age_at_index"] = out["person_id"].map(idx).dt.year - years

    if "gender" not in out.columns:
        if "gender_source_value" in out.columns:
            out["gender"] = out["gender_source_value"]
        elif "gender_concept_id" in out.columns:
            out["gender"] = out["gender_concept_id"].map({8507: "Male", 8532: "Female"})
    if "race" not in out.columns and "race_source_value" in out.columns:
        out["race"] = out["race_source_value"]
    if "ethnicity" not in out.columns and "ethnicity_source_value" in out.columns:
        out["ethnicity"] = out["ethnicity_source_value"]

    for col in ["age_at_index", "gender", "race", "ethnicity"]:
        if col not in out.columns:
            out[col] = None
    return out[["person_id", "age_at_index", "gender", "race", "ethnicity"]]


def _windowed_rows(
    frame,
    patient_index,
    *,
    person_col: str,
    date_col: str,
    lookback_days: int | None,
):
    require_pandas()
    data = frame.copy()
    data[person_col] = pd.to_numeric(data[person_col], errors="coerce")
    data = data.dropna(subset=[person_col])
    data[person_col] = data[person_col].astype("int64")
    index = patient_index[["person_id", "ici_index_date"]].copy()
    data = data.merge(index, left_on=person_col, right_on="person_id", how="inner")
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col, "ici_index_date"])
    before_or_at_index = data[date_col] <= data["ici_index_date"]
    if lookback_days is None:
        return data[before_or_at_index].copy()
    start = data["ici_index_date"] - pd.to_timedelta(lookback_days, unit="D")
    return data[(start <= data[date_col]) & before_or_at_index].copy()


def build_charlson_flags(
    diagnosis_frame,
    patient_index,
    *,
    lookback_days: int | None = None,
    person_col: str | None = None,
    code_col: str | None = None,
    date_col: str | None = None,
):
    """Compute pre-index Charlson binary flags from diagnosis rows.

    ``lookback_days=None`` means all diagnosis history before or at ICI index.
    """
    require_pandas()
    index = _prepare_patient_index(patient_index)
    person_col = person_col or _pick_column(diagnosis_frame.columns, PERSON_ALIASES)
    code_col = code_col or _pick_column(diagnosis_frame.columns, DX_CODE_ALIASES)
    date_col = date_col or _pick_column(diagnosis_frame.columns, DX_DATE_ALIASES)

    out = index[["person_id"]].copy()
    for col in CHARLSON_COMORBIDITIES:
        out[col] = 0
    if diagnosis_frame.empty:
        return out

    dx = _windowed_rows(
        diagnosis_frame[[person_col, code_col, date_col]].copy(),
        index,
        person_col=person_col,
        date_col=date_col,
        lookback_days=lookback_days,
    )
    if dx.empty:
        out["charlson_comorbidity_count"] = 0
        return out

    dx["_icd"] = dx[code_col].map(normalize_icd_code)
    flags = out.set_index("person_id")
    for condition, codeset in CHARLSON_CODESETS.items():
        prefixes = _flatten_codes(codeset)
        matched = dx.loc[
            dx["_icd"].map(lambda code: bool(code) and any(code.startswith(p) for p in prefixes)),
            "person_id",
        ].dropna()
        if not matched.empty:
            flags.loc[flags.index.isin(set(matched.astype("int64"))), condition] = 1

    oi_prefixes = _flatten_codes(AIDS_OI_CODESET)
    oi_pids = set(
        dx.loc[
            dx["_icd"].map(lambda code: bool(code) and any(code.startswith(p) for p in oi_prefixes)),
            "person_id",
        ]
        .dropna()
        .astype("int64")
    )
    hiv_pids = set(flags.index[flags["HIV"].eq(1)])
    aids_pids = hiv_pids & oi_pids
    if aids_pids:
        flags.loc[flags.index.isin(aids_pids), "AIDS"] = 1
        flags.loc[flags.index.isin(aids_pids), "HIV"] = 0

    for severe, mild in HIERARCHY_PAIRS:
        flags.loc[flags[severe].eq(1), mild] = 0

    flags["charlson_comorbidity_count"] = flags[CHARLSON_COMORBIDITIES].sum(axis=1).astype(int)
    return flags.reset_index()


def _normalize_drug_name(value: Any) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).replace("\n", " ").split()).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def build_medication_features(
    medication_frame,
    patient_index,
    *,
    lookback_days: int = 365,
    person_col: str | None = None,
    drug_col: str | None = None,
    date_col: str | None = None,
    max_drugs: int = 80,
):
    """Aggregate unique pre-index medication names into pipe-delimited strings."""
    require_pandas()
    index = _prepare_patient_index(patient_index)
    out = index[["person_id"]].copy()
    out["baseline_drug_count"] = 0
    out["baseline_drugs"] = ""
    if medication_frame.empty:
        return out

    person_col = person_col or _pick_column(medication_frame.columns, PERSON_ALIASES)
    drug_col = drug_col or _pick_column(medication_frame.columns, DRUG_NAME_ALIASES)
    date_col = date_col or _pick_column(medication_frame.columns, DRUG_DATE_ALIASES)

    meds = _windowed_rows(
        medication_frame[[person_col, drug_col, date_col]].copy(),
        index,
        person_col=person_col,
        date_col=date_col,
        lookback_days=lookback_days,
    )
    if meds.empty:
        return out

    meds["_drug"] = meds[drug_col].map(_normalize_drug_name)
    meds = meds[meds["_drug"].ne("")]
    if meds.empty:
        return out

    rows = []
    for pid, group in meds.groupby("person_id"):
        seen: dict[str, str] = {}
        for name in group["_drug"].tolist():
            key = name.lower()
            if key not in seen:
                seen[key] = name
        drugs = [seen[key] for key in sorted(seen)]
        rows.append(
            {
                "person_id": int(pid),
                "baseline_drug_count": len(drugs),
                "baseline_drugs": "|".join(drugs[:max_drugs]),
            }
        )
    return out.drop(columns=["baseline_drug_count", "baseline_drugs"]).merge(
        pd.DataFrame(rows), on="person_id", how="left"
    ).fillna({"baseline_drug_count": 0, "baseline_drugs": ""})


def build_baseline_features(
    *,
    cohort,
    diagnosis_frame=None,
    medication_frame=None,
    person_frame=None,
    charlson_lookback_days: int | None = None,
    drug_lookback_days: int = 365,
    max_drugs: int = 80,
):
    """Combine demographics, Charlson flags, and medication features."""
    require_pandas()
    features = build_demographic_features(cohort, person_frame=person_frame)
    index = _prepare_patient_index(cohort)

    if diagnosis_frame is None:
        charlson = index[["person_id"]].copy()
        for col in CHARLSON_COMORBIDITIES:
            charlson[col] = 0
        charlson["charlson_comorbidity_count"] = 0
    else:
        charlson = build_charlson_flags(
            diagnosis_frame,
            index,
            lookback_days=charlson_lookback_days,
        )

    if medication_frame is None:
        meds = index[["person_id"]].copy()
        meds["baseline_drug_count"] = 0
        meds["baseline_drugs"] = ""
    else:
        meds = build_medication_features(
            medication_frame,
            index,
            lookback_days=drug_lookback_days,
            max_drugs=max_drugs,
        )

    features = features.merge(charlson, on="person_id", how="left")
    features = features.merge(meds, on="person_id", how="left")
    features[CHARLSON_COMORBIDITIES] = features[CHARLSON_COMORBIDITIES].fillna(0).astype("int8")
    features["charlson_comorbidity_count"] = (
        features["charlson_comorbidity_count"].fillna(0).astype(int)
    )
    features["baseline_drug_count"] = features["baseline_drug_count"].fillna(0).astype(int)
    features["baseline_drugs"] = features["baseline_drugs"].fillna("")
    return features


def build_baseline_features_from_paths(
    *,
    cohort_csv: str | Path,
    data_dir: str | Path | None = None,
    diagnosis_path: str | Path | None = None,
    medication_path: str | Path | None = None,
    person_path: str | Path | None = None,
    charlson_lookback_days: int | None = None,
    drug_lookback_days: int = 365,
    max_drugs: int = 80,
):
    """Load known OMOP tables and build patient-level baseline features."""
    require_pandas()
    cohort = pd.read_csv(cohort_csv)
    if data_dir is not None:
        diagnosis_path = diagnosis_path or _find_omop_table(data_dir, "condition_occurrence")
        medication_path = medication_path or _find_omop_table(data_dir, "drug_exposure")
        try:
            person_path = person_path or _find_omop_table(data_dir, "person")
        except FileNotFoundError:
            person_path = person_path

    diagnosis_cols = list(PERSON_ALIASES + DX_CODE_ALIASES + DX_DATE_ALIASES)
    medication_cols = list(PERSON_ALIASES + DRUG_NAME_ALIASES + DRUG_DATE_ALIASES)
    person_cols = list(PERSON_ALIASES + tuple(DEMOGRAPHIC_COLUMNS))

    diagnosis = _read_table(diagnosis_path, columns=diagnosis_cols) if diagnosis_path else None
    medication = _read_table(medication_path, columns=medication_cols) if medication_path else None
    person = _read_table(person_path, columns=person_cols) if person_path else None
    return build_baseline_features(
        cohort=cohort,
        diagnosis_frame=diagnosis,
        medication_frame=medication,
        person_frame=person,
        charlson_lookback_days=charlson_lookback_days,
        drug_lookback_days=drug_lookback_days,
        max_drugs=max_drugs,
    )


def format_baseline_features(row: Any, *, max_drug_chars: int = 2000) -> str:
    """Render merge columns into compact prompt text."""
    get = row.get if hasattr(row, "get") else lambda key, default=None: getattr(row, key, default)
    fields = []
    for label, key in [
        ("age_at_index", "age_at_index"),
        ("gender", "gender"),
        ("race", "race"),
        ("ethnicity", "ethnicity"),
        ("ici_regimen", "ici_regimen"),
    ]:
        value = get(key, None)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            fields.append(f"- {label}: {value}")

    positives = []
    for col in CHARLSON_COMORBIDITIES:
        try:
            present = int(float(get(col, 0) or 0)) == 1
        except (TypeError, ValueError):
            present = False
        if present:
            positives.append(col)
    count = get("charlson_comorbidity_count", len(positives))
    fields.append(f"- charlson_positive_flags: {', '.join(positives) if positives else 'none'}")
    fields.append(f"- charlson_comorbidity_count: {count}")

    drug_count = get("baseline_drug_count", 0)
    drugs = str(get("baseline_drugs", "") or "")
    if len(drugs) > max_drug_chars:
        drugs = drugs[:max_drug_chars] + "...[truncated]"
    fields.append(f"- prior_year_drug_count: {drug_count}")
    fields.append(f"- prior_year_drugs_pipe_delimited: {drugs if drugs else 'none documented'}")
    return "\n".join(fields)
