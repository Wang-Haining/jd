"""Routing helpers for AKI tumor-board-style prediction experiments."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .aki import AKI_AGENT_ORDER, AKI_AGENT_PROMPTS, AKIPrediction


AKI_ROUTING_STRATEGIES = [
    "single_nephrologist",
    "five_agent_independent_mean",
    "arbitrary_round_robin",
    "agent_handoff",
    "jlens_low_std",
    "jlens_high_std",
]


@dataclass(frozen=True)
class AKIRoutingTurn:
    """One structured clinician prediction turn in a routed discussion."""

    speaker: str
    prediction: AKIPrediction
    route_metadata: dict[str, Any] | None = None


def prediction_to_discussion_line(prediction: AKIPrediction) -> str:
    """Compact a structured prediction for subsequent clinicians."""
    for_text = "; ".join(prediction.key_evidence_for_aki[:4]) or "none stated"
    against_text = "; ".join(prediction.key_evidence_against_aki[:4]) or "none stated"
    missing = "; ".join(prediction.missing_information[:3]) or "none stated"
    return (
        f"{prediction.agent_id}: any={prediction.aki_any_probability:.2f} "
        f"({prediction.aki_any_call}), 3mo={prediction.aki_3mo_probability:.2f} "
        f"({prediction.aki_3mo_call}), 6mo={prediction.aki_6mo_probability:.2f} "
        f"({prediction.aki_6mo_call}), confidence={prediction.confidence:.2f}. "
        f"For AKI: {for_text}. Against AKI: {against_text}. "
        f"Missing: {missing}. Rationale: {prediction.rationale}"
    )


def format_turns(turns: list[AKIRoutingTurn]) -> str:
    """Render routed prediction turns into prior discussion text."""
    if not turns:
        return ""
    return "\n".join(
        f"[turn {idx} | {turn.speaker}] {prediction_to_discussion_line(turn.prediction)}"
        for idx, turn in enumerate(turns, start=1)
    )


def next_round_robin_agent(current_agent: str, eligible_agents: list[str]) -> str:
    """Choose the next unspoken agent by the fixed clinician order."""
    if not eligible_agents:
        return current_agent
    if current_agent not in AKI_AGENT_ORDER:
        return eligible_agents[0]
    start = AKI_AGENT_ORDER.index(current_agent)
    for offset in range(1, len(AKI_AGENT_ORDER) + 1):
        candidate = AKI_AGENT_ORDER[(start + offset) % len(AKI_AGENT_ORDER)]
        if candidate in eligible_agents:
            return candidate
    return eligible_agents[0]


def build_handoff_choice_messages(
    *,
    current_agent: str,
    prior_discussion: str,
    eligible_agents: list[str],
) -> list[dict[str, str]]:
    """Ask the current clinician to choose the next reviewer."""
    options = ", ".join(eligible_agents)
    request = (
        "Given the AKI risk discussion so far, choose the next clinician who "
        "should review the case. Choose exactly one ID from this list: "
        f"{options}.\n\n"
        f"=== PRIOR DISCUSSION ===\n{prior_discussion or '(No prior discussion.)'}\n\n"
        "Return only this format:\nHANDOFF_TO: <agent_id>"
    )
    return [
        {"role": "system", "content": AKI_AGENT_PROMPTS[current_agent]},
        {"role": "user", "content": request},
    ]


def parse_handoff_choice(text: str, eligible_agents: list[str]) -> str | None:
    """Parse a HANDOFF_TO choice from model text."""
    match = re.search(r"HANDOFF_TO\s*:\s*([a-zA-Z_]+)", text)
    candidate = match.group(1).strip() if match else ""
    if not candidate and text.strip():
        candidate = re.split(r"[\s,.;]+", text.strip())[0]
    return candidate if candidate in eligible_agents else None


def select_by_std(
    current_agent: str,
    candidate_scores: dict[str, dict[str, float]],
    *,
    mode: str,
    epsilon: float = 1e-9,
) -> tuple[str, dict[str, Any]]:
    """Select next agent by lowest or highest J-space STD, with stable ties."""
    if not candidate_scores:
        return current_agent, {"tie_breaker": "none", "std_tied_candidates": []}
    if mode not in {"low", "high"}:
        raise ValueError(f"Unknown STD selection mode: {mode}")

    values = {name: float(score["std"]) for name, score in candidate_scores.items()}
    best_std = min(values.values()) if mode == "low" else max(values.values())
    tied = [name for name, value in values.items() if abs(value - best_std) <= epsilon]
    if len(tied) == 1:
        return tied[0], {"tie_breaker": "std", "std_tied_candidates": tied}
    return next_round_robin_agent(current_agent, tied), {
        "tie_breaker": "round_robin",
        "std_tied_candidates": sorted(tied),
    }
