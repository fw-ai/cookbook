"""Simple local grader for the GRPO remote-rollout example."""

from __future__ import annotations

import re
from typing import Any

from eval_protocol.models import EvaluateResult, EvaluationRow, MetricResult

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def extract_answer(text: str) -> str:
    """Extract a final answer from model text."""
    boxed = _BOXED_RE.findall(text or "")
    if boxed:
        return boxed[-1].strip()
    numbers = _NUMBER_RE.findall(text or "")
    if numbers:
        return numbers[-1].strip()
    return (text or "").strip()


def normalize_answer(answer: Any) -> str:
    text = str(answer or "").strip()
    try:
        return str(float(text)).rstrip("0").rstrip(".")
    except ValueError:
        return re.sub(r"\s+", " ", text.lower())


def score_text(text: str, expected: Any) -> tuple[float, str]:
    predicted = extract_answer(text)
    normalized_predicted = normalize_answer(predicted)
    normalized_expected = normalize_answer(expected)
    score = 1.0 if normalized_predicted == normalized_expected else 0.0
    reason = (
        f"predicted={predicted!r} expected={expected!r} "
        f"normalized_predicted={normalized_predicted!r} normalized_expected={normalized_expected!r}"
    )
    return score, reason


def _assistant_text(row: EvaluationRow) -> str:
    for message in reversed(row.messages):
        if getattr(message, "role", None) == "assistant":
            return str(getattr(message, "content", "") or "")
    return ""


def grade_row(row: EvaluationRow, sample_prompt: dict[str, Any]) -> EvaluationRow:
    """Populate ``row.evaluation_result`` from the final assistant answer."""
    expected = (
        sample_prompt.get("answer")
        or sample_prompt.get("ground_truth")
        or sample_prompt.get("expected")
    )
    if expected is None:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            is_score_valid=False,
            reason="sample_prompt missing answer/ground_truth/expected",
        )
        return row

    text = _assistant_text(row)
    score, reason = score_text(text, expected)
    row.evaluation_result = EvaluateResult(
        score=score,
        is_score_valid=True,
        reason=reason,
        metrics={
            "exact_match": MetricResult(
                score=score,
                is_score_valid=True,
                reason=reason,
            )
        },
    )
    return row
