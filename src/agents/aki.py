"""Structured five-agent AKI prediction helpers."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


AKI_AGENT_ORDER = [
    "oncologist",
    "nephrologist",
    "pharmacist",
    "hospitalist",
    "critical_care_infectious_risk",
]


AKI_AGENT_PROMPTS = {
    "oncologist": (
        "You are a medical oncologist. Focus on cancer type, disease burden, "
        "ICI regimen, prior therapy, frailty from cancer, and treatment context "
        "that may affect post-ICI AKI risk."
    ),
    "nephrologist": (
        "You are a nephrologist. Focus on baseline kidney reserve, CKD, prior "
        "AKI, creatinine/eGFR trajectory, proteinuria, urinalysis, obstruction, "
        "and KDIGO-relevant risk."
    ),
    "pharmacist": (
        "You are a clinical pharmacist. Focus on nephrotoxic medications, NSAIDs, "
        "PPIs, ACEi/ARB, diuretics, antibiotics, contrast, chemotherapy, renal "
        "dosing, and drug-drug interactions."
    ),
    "hospitalist": (
        "You are a hospitalist. Focus on timeline coherence, comorbidities, "
        "hospitalizations, volume status, documentation gaps, and overall medical "
        "complexity."
    ),
    "critical_care_infectious_risk": (
        "You are an intensivist/infectious disease clinician. Focus on sepsis, "
        "shock, hypotension, ICU exposure, dehydration, severe infection, and "
        "other acute illness risks for early AKI."
    ),
}


class AKIPrediction(BaseModel):
    """Structured prediction for post-ICI AKI outcomes."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    person_id: int
    agent_id: str
    aki_any_probability: float = Field(ge=0.0, le=1.0)
    aki_3mo_probability: float = Field(ge=0.0, le=1.0)
    aki_6mo_probability: float = Field(ge=0.0, le=1.0)
    aki_any_call: Literal["yes", "no", "uncertain"]
    aki_3mo_call: Literal["yes", "no", "uncertain"]
    aki_6mo_call: Literal["yes", "no", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    key_evidence_for_aki: list[str] = Field(default_factory=list)
    key_evidence_against_aki: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    rationale: str


class AKIAggregatePrediction(BaseModel):
    """Mean-probability ensemble over agent predictions."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    person_id: int
    strategy: str
    n_agents: int
    aki_any_probability: float = Field(ge=0.0, le=1.0)
    aki_3mo_probability: float = Field(ge=0.0, le=1.0)
    aki_6mo_probability: float = Field(ge=0.0, le=1.0)
    aki_any_call: Literal["yes", "no", "uncertain"]
    aki_3mo_call: Literal["yes", "no", "uncertain"]
    aki_6mo_call: Literal["yes", "no", "uncertain"]


def prediction_schema() -> dict:
    """Return JSON schema for structured decoding."""
    return AKIPrediction.model_json_schema()


def render_summary_for_prompt(summary: dict) -> str:
    """Compact a pre-index MedACE JSON into prompt text."""
    return json.dumps(summary, ensure_ascii=False, indent=2)[:60_000]


def build_prediction_messages(
    *,
    agent_id: str,
    summary: dict,
    prior_discussion: str = "",
) -> list[dict[str, str]]:
    """Build messages for one structured AKI prediction."""
    system = AKI_AGENT_PROMPTS[agent_id]
    user = (
        "Predict this patient's risk of post-ICI acute kidney injury using only "
        "the pre-index structured note summary below. AKI means KDIGO stage 1 "
        "or higher: serum creatinine increase >=0.3 mg/dL or >=1.5x baseline. "
        "Do not infer that an outcome occurred unless it is supported by "
        "pre-index risk factors; this is a prediction task. The yes/no/uncertain "
        "calls must match the probabilities: yes if probability >=0.65, no if "
        "probability <=0.35, otherwise uncertain.\n\n"
        "Return only JSON matching the requested schema.\n\n"
        f"AGENT_ID: {agent_id}\n\n"
        f"PRE_INDEX_SUMMARY:\n{render_summary_for_prompt(summary)}"
    )
    if prior_discussion:
        user += f"\n\nPRIOR_AGENT_DISCUSSION:\n{prior_discussion[-20_000:]}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_from_probability(probability: float) -> str:
    """Convert a probability into a coarse yes/no/uncertain call."""
    if probability >= 0.65:
        return "yes"
    if probability <= 0.35:
        return "no"
    return "uncertain"


def normalize_prediction_calls(prediction: AKIPrediction) -> AKIPrediction:
    """Make coarse call fields deterministic derivatives of probabilities."""
    data = prediction.model_dump()
    data["aki_any_call"] = call_from_probability(data["aki_any_probability"])
    data["aki_3mo_call"] = call_from_probability(data["aki_3mo_probability"])
    data["aki_6mo_call"] = call_from_probability(data["aki_6mo_probability"])
    return AKIPrediction.model_validate(data)


def aggregate_predictions(
    predictions: list[AKIPrediction],
    *,
    strategy: str = "five_agent_mean",
) -> AKIAggregatePrediction:
    """Aggregate agent predictions by mean probability."""
    if not predictions:
        raise ValueError("No predictions to aggregate")
    any_prob = sum(p.aki_any_probability for p in predictions) / len(predictions)
    p3 = sum(p.aki_3mo_probability for p in predictions) / len(predictions)
    p6 = sum(p.aki_6mo_probability for p in predictions) / len(predictions)
    first = predictions[0]
    return AKIAggregatePrediction(
        sample_id=first.sample_id,
        person_id=first.person_id,
        strategy=strategy,
        n_agents=len(predictions),
        aki_any_probability=any_prob,
        aki_3mo_probability=p3,
        aki_6mo_probability=p6,
        aki_any_call=call_from_probability(any_prob),
        aki_3mo_call=call_from_probability(p3),
        aki_6mo_call=call_from_probability(p6),
    )
