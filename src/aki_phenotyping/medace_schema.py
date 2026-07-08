"""Simplified pre-index MedACE schema for post-ICI AKI prediction."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Source(BaseModel):
    """Note provenance for an extracted fact."""

    model_config = ConfigDict(extra="forbid")

    note_id: str | None = None
    note_date: str | None = None
    note_type: str | None = None


class Fact(BaseModel):
    """Auditable extracted fact."""

    model_config = ConfigDict(extra="forbid")

    value: str | int | float | bool | list[str] | None
    evidence: str = Field(description="Short verbatim quote or close paraphrase from the note.")
    confidence: float = Field(ge=0.0, le=1.0)
    observed_date: str | None = None
    provenance: Source | None = None


Assertion = Literal["present", "possible", "absent", "not_documented"]


class CancerICIContext(BaseModel):
    """Cancer and ICI-treatment context available at or before ICI start."""

    model_config = ConfigDict(extra="forbid")

    assertion: Assertion = "not_documented"
    cancer_type: Fact | None = None
    cancer_stage_or_extent: Fact | None = None
    ici_drugs: Fact | None = None
    ici_regimen: Fact | None = None
    treatment_intent: Fact | None = None
    prior_cancer_therapy: list[Fact] = Field(default_factory=list)


class KidneyBaseline(BaseModel):
    """Baseline renal reserve and kidney history before the ICI index date."""

    model_config = ConfigDict(extra="forbid")

    assertion: Assertion = "not_documented"
    baseline_creatinine_mg_dl: list[Fact] = Field(default_factory=list)
    baseline_egfr_ml_min_1_73m2: list[Fact] = Field(default_factory=list)
    chronic_kidney_disease: Fact | None = None
    prior_aki: Fact | None = None
    dialysis_or_transplant_history: Fact | None = None
    proteinuria_or_albuminuria: list[Fact] = Field(default_factory=list)
    hematuria_or_abnormal_urinalysis: list[Fact] = Field(default_factory=list)


class ComorbidityContext(BaseModel):
    """Comorbidities that materially affect AKI risk."""

    model_config = ConfigDict(extra="forbid")

    assertion: Assertion = "not_documented"
    diabetes: Fact | None = None
    hypertension: Fact | None = None
    heart_failure_or_cardiovascular_disease: Fact | None = None
    liver_disease: Fact | None = None
    autoimmune_disease: Fact | None = None
    frailty_or_poor_functional_status: Fact | None = None
    other_major_comorbidities: list[Fact] = Field(default_factory=list)


class MedicationRisk(BaseModel):
    """Pre-index medication and exposure risks for AKI."""

    model_config = ConfigDict(extra="forbid")

    assertion: Assertion = "not_documented"
    nephrotoxic_medications: list[Fact] = Field(default_factory=list)
    acei_arb_or_diuretic: list[Fact] = Field(default_factory=list)
    nsaid_or_ppi_exposure: list[Fact] = Field(default_factory=list)
    recent_antibiotics_or_antivirals: list[Fact] = Field(default_factory=list)
    contrast_exposure: list[Fact] = Field(default_factory=list)
    chemotherapy_or_targeted_therapy: list[Fact] = Field(default_factory=list)


class AcuteIllnessContext(BaseModel):
    """Acute illness near ICI start that may raise short-term AKI risk."""

    model_config = ConfigDict(extra="forbid")

    assertion: Assertion = "not_documented"
    infection_or_sepsis: list[Fact] = Field(default_factory=list)
    dehydration_or_poor_intake: list[Fact] = Field(default_factory=list)
    hypotension_or_shock: list[Fact] = Field(default_factory=list)
    hospitalization_or_icu: list[Fact] = Field(default_factory=list)
    obstruction_or_urologic_issue: list[Fact] = Field(default_factory=list)


class ExtractionQuality(BaseModel):
    """Coverage and uncertainty in the available pre-index notes."""

    model_config = ConfigDict(extra="forbid")

    note_coverage_summary: str
    key_missing_information: list[str] = Field(default_factory=list)
    leakage_check: Literal["pre_index_only", "possible_post_index_content"]


class AKIPreIndexSummary(BaseModel):
    """Pre-index structured summary for post-ICI AKI prediction."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    person_id: int
    landmark_date: str
    window_start: str
    window_end: str
    narrative_summary: str
    cancer_ici_context: CancerICIContext = Field(default_factory=CancerICIContext)
    kidney_baseline: KidneyBaseline = Field(default_factory=KidneyBaseline)
    comorbidity_context: ComorbidityContext = Field(default_factory=ComorbidityContext)
    medication_risk: MedicationRisk = Field(default_factory=MedicationRisk)
    acute_illness_context: AcuteIllnessContext = Field(default_factory=AcuteIllnessContext)
    extraction_quality: ExtractionQuality
    sources: list[Source] = Field(default_factory=list)

