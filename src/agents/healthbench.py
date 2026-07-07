"""
Freeform HealthBench clinician debate helpers.

No Delphi machinery is used here. A small fixed clinician pool discusses a
case for a capped number of turns. The experiment compares how the next speaker
is chosen: round robin, current-agent handoff, or a J-lens next-speaker score.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


NEUTRAL_SYSTEM_PROMPT = (
    "You are a careful clinical assistant. Respond to the user's last message "
    "with medically accurate, safe, and practical guidance. Ask for urgent care "
    "when appropriate, avoid unsupported certainty, and tailor the answer to the "
    "conversation."
)


CLINICIAN_PROMPTS = {
    "generalist": (
        "You are a general internal medicine clinician. Synthesize the clinical "
        "conversation into safe, practical guidance, identify uncertainty, and "
        "tailor the response to the user's context."
    ),
    "emergency": (
        "You are an emergency medicine clinician. Prioritize time-sensitive red "
        "flags, escalation decisions, immediate safety steps, and clear triage "
        "language without over-explaining."
    ),
    "diagnostician": (
        "You are a diagnostic medicine clinician. Consider the differential, "
        "separate likely from dangerous possibilities, and state what additional "
        "context would change management."
    ),
    "evidence": (
        "You are an evidence-based medicine clinician. Ground advice in "
        "established clinical principles, avoid unsupported certainty, and make "
        "risk/benefit tradeoffs explicit."
    ),
    "safety": (
        "You are a patient-safety clinician. Look for unsafe advice, missing "
        "escalation, harmful reassurance, medication hazards, and communication "
        "that could lead to delay in care."
    ),
}


CLINICIAN_ORDER = [
    "generalist",
    "emergency",
    "diagnostician",
    "evidence",
    "safety",
]


FREE_TEXT_STRATEGIES = [
    "single_neutral",
    "debate_round_robin",
    "debate_agent_handoff",
    "debate_jlens_next",
]


@dataclass(frozen=True)
class DebateTurn:
    """One generated freeform debate turn."""

    speaker: str
    response_text: str
    handoff_to: str | None = None
    route_metadata: dict[str, Any] | None = None


def with_system_prompt(
    prompt_messages: list[dict[str, str]],
    system_prompt: str,
) -> list[dict[str, str]]:
    """Prepend a system prompt while preserving the HealthBench conversation."""
    return [{"role": "system", "content": system_prompt}] + [
        {"role": message["role"], "content": message["content"]}
        for message in prompt_messages
    ]


def build_neutral_messages(prompt_messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build the single-neutral baseline prompt."""
    return with_system_prompt(prompt_messages, NEUTRAL_SYSTEM_PROMPT)


def build_initial_debate_messages(
    prompt_messages: list[dict[str, str]],
    speaker: str = "generalist",
) -> list[dict[str, str]]:
    """Build the first freeform clinician answer."""
    request = (
        "Provide your best clinical response to the user's last message. This is "
        "the opening turn of a short clinician discussion, so be concise, safe, "
        "and clinically specific."
    )
    return (
        with_system_prompt(prompt_messages, CLINICIAN_PROMPTS[speaker])
        + [{"role": "user", "content": request}]
    )


def format_debate_transcript(turns: list[DebateTurn]) -> str:
    """Render prior turns for the next clinician."""
    if not turns:
        return "(No prior discussion.)"
    parts = []
    for idx, turn in enumerate(turns, start=1):
        parts.append(f"Turn {idx} - {turn.speaker}:\n{turn.response_text.strip()}")
    return "\n\n".join(parts)


def build_debate_turn_messages(
    prompt_messages: list[dict[str, str]],
    *,
    speaker: str,
    turns: list[DebateTurn],
) -> list[dict[str, str]]:
    """Build a prompt for a clinician to continue the freeform discussion."""
    latest = turns[-1].response_text if turns else ""
    request = (
        "Continue the clinician discussion below. Review the latest response, "
        "correct clinically important omissions or unsafe wording, and produce "
        "one improved response to the user's last message. Do not mention the "
        "discussion process.\n\n"
        f"=== PRIOR DISCUSSION ===\n{format_debate_transcript(turns)}\n\n"
        f"=== LATEST RESPONSE TO IMPROVE ===\n{latest.strip()}"
    )
    return (
        with_system_prompt(prompt_messages, CLINICIAN_PROMPTS[speaker])
        + [{"role": "user", "content": request}]
    )


def build_handoff_choice_messages(
    prompt_messages: list[dict[str, str]],
    *,
    current_speaker: str,
    turns: list[DebateTurn],
    eligible_speakers: list[str],
) -> list[dict[str, str]]:
    """Ask the current clinician to choose the next speaker."""
    options = ", ".join(eligible_speakers)
    request = (
        "Given the prior discussion, choose the next clinician who should review "
        "and improve the response. Choose exactly one ID from this list: "
        f"{options}.\n\n"
        f"=== PRIOR DISCUSSION ===\n{format_debate_transcript(turns)}\n\n"
        "Return only this format:\nHANDOFF_TO: <clinician_id>"
    )
    return (
        with_system_prompt(prompt_messages, CLINICIAN_PROMPTS[current_speaker])
        + [{"role": "user", "content": request}]
    )


def parse_handoff_choice(text: str, eligible_speakers: list[str]) -> str | None:
    """Parse a HANDOFF_TO choice from model text."""
    match = re.search(r"HANDOFF_TO\s*:\s*([a-zA-Z_]+)", text)
    candidate = match.group(1).strip() if match else text.strip().split()[0] if text.strip() else ""
    return candidate if candidate in eligible_speakers else None


def next_round_robin_speaker(current_speaker: str, eligible_speakers: list[str]) -> str:
    """Return the next speaker in fixed clinician order."""
    if not eligible_speakers:
        return current_speaker
    ordered = [speaker for speaker in CLINICIAN_ORDER if speaker in eligible_speakers]
    if current_speaker not in CLINICIAN_ORDER:
        return ordered[0]
    start = CLINICIAN_ORDER.index(current_speaker)
    for offset in range(1, len(CLINICIAN_ORDER) + 1):
        candidate = CLINICIAN_ORDER[(start + offset) % len(CLINICIAN_ORDER)]
        if candidate in ordered:
            return candidate
    return ordered[0]


def jlens_next_score(std_score: float) -> float:
    """
    Convert candidate next-speaker J-space uncertainty into a routing score.

    Lower STD means the candidate appears internally steadier on the next-turn
    prompt, so score is monotonically decreasing in STD.
    """
    return 1.0 / (1.0 + max(0.0, std_score))
