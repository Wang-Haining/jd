"""
HealthBench rubric scoring helpers.

The scoring contract follows the public HealthBench approach: a grader decides
whether each rubric criterion is met, and the case score is achieved rubric
points divided by total positive rubric points. Negative criteria reduce the
achieved points when met.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol


DEFAULT_GRADER_MODEL = "gpt-4.1-2025-04-14"


class Grader(Protocol):
    """Protocol for a rubric-item grader."""

    def grade(
        self,
        prompt_messages: list[dict[str, str]],
        response_text: str,
        rubric_item: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a dict with criteria_met and explanation."""


def conversation_text(
    prompt_messages: list[dict[str, str]],
    response_text: str,
) -> str:
    """Render the HealthBench conversation plus candidate response."""
    messages = prompt_messages + [{"role": "assistant", "content": response_text}]
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def build_grader_prompt(
    prompt_messages: list[dict[str, str]],
    response_text: str,
    rubric_item: dict[str, Any],
) -> str:
    """Build the rubric-item grading prompt."""
    rubric = f"[{rubric_item['points']}] {rubric_item['criterion']}"
    return (
        "Your job is to evaluate the last assistant response in a clinical "
        "conversation against one rubric item.\n\n"
        "# Conversation\n"
        f"{conversation_text(prompt_messages, response_text)}\n\n"
        "# Rubric item\n"
        f"{rubric}\n\n"
        "# Instructions\n"
        "Return only a JSON object with two fields: explanation and "
        "criteria_met. criteria_met must be true only if the response satisfies "
        "the rubric item. If the item describes undesirable behavior with "
        "negative points, criteria_met should be true when the undesirable "
        "behavior is present and false when it is absent."
    )


def parse_grader_json(text: str) -> dict[str, Any]:
    """Parse a grader JSON response, accepting fenced JSON."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse grader JSON: {text[:200]}") from exc
    if not isinstance(parsed.get("criteria_met"), bool):
        raise ValueError(f"Grader JSON missing boolean criteria_met: {parsed}")
    return {
        "criteria_met": parsed["criteria_met"],
        "explanation": str(parsed.get("explanation", "")),
    }


@dataclass
class OpenAIHealthBenchGrader:
    """OpenAI-backed rubric grader used for HealthBench scoring."""

    model: str = DEFAULT_GRADER_MODEL
    temperature: float = 0.0
    max_output_tokens: int = 768

    def __post_init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI()

    def _responses_create(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text

        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
        return "\n".join(chunks)

    def _chat_completions_create(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        return response.choices[0].message.content or ""

    def grade(
        self,
        prompt_messages: list[dict[str, str]],
        response_text: str,
        rubric_item: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = build_grader_prompt(prompt_messages, response_text, rubric_item)
        if hasattr(self.client, "responses"):
            raw = self._responses_create(prompt)
        else:
            raw = self._chat_completions_create(prompt)
        parsed = parse_grader_json(raw)
        return {
            **parsed,
            "grader_model": self.model,
            "raw_response": raw,
        }


class KeywordFakeGrader:
    """
    Deterministic fake grader for tests and local dry runs.

    A criterion is met if every lowercase token from the criterion appears in
    the response. This is intentionally simple and should never be used for
    reported study scores.
    """

    def grade(
        self,
        prompt_messages: list[dict[str, str]],
        response_text: str,
        rubric_item: dict[str, Any],
    ) -> dict[str, Any]:
        words = [
            word
            for word in re.findall(r"[a-zA-Z]{4,}", rubric_item["criterion"].lower())
            if word not in {"that", "with", "from", "should", "would", "could"}
        ]
        response = response_text.lower()
        criteria_met = bool(words) and all(word in response for word in words[:3])
        return {
            "criteria_met": criteria_met,
            "explanation": "KeywordFakeGrader local dry-run decision.",
            "grader_model": "keyword_fake",
        }


def calculate_score(
    rubrics: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> float | None:
    """Calculate a HealthBench case score from rubric decisions."""
    total_possible = sum(float(item["points"]) for item in rubrics if item["points"] > 0)
    if total_possible <= 0:
        return None
    achieved = sum(
        float(item["points"])
        for item, decision in zip(rubrics, decisions, strict=True)
        if decision["criteria_met"]
    )
    return achieved / total_possible


def score_response(
    prompt_messages: list[dict[str, str]],
    response_text: str,
    rubrics: list[dict[str, Any]],
    grader: Grader,
) -> dict[str, Any]:
    """Grade one response against all rubrics and return score + decisions."""
    decisions = []
    for idx, rubric_item in enumerate(rubrics):
        decision = grader.grade(prompt_messages, response_text, rubric_item)
        decisions.append(
            {
                "rubric_index": idx,
                "criterion": rubric_item["criterion"],
                "points": rubric_item["points"],
                "tags": rubric_item.get("tags", []),
                **decision,
            }
        )
    return {
        "score": calculate_score(rubrics, decisions),
        "decisions": decisions,
    }
