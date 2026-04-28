"""Eval Protocol grader for GSM8K-style math answers.

The ``@evaluation_test`` decorator registers this function for EP's
pytest-based discovery and evaluation runners.  The cookbook recipe
imports the decorated coroutine from :mod:`train` and awaits it
directly (no pytest involved at training time).  Keeping the grader
in its own module lets the same function serve both paths: ``pytest``
evaluation runs and online RL grading.
"""

import re
from typing import Optional

from eval_protocol import EvaluateResult, EvaluationRow, evaluation_test
from eval_protocol.pytest import SingleTurnRolloutProcessor


_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_DIGIT_PATTERN = re.compile(r"(-?\d+)")


def _extract_completion(text: str) -> Optional[str]:
    """Parse ``<answer>N</answer>`` from the model's output."""
    m = _ANSWER_PATTERN.search(text or "")
    if not m:
        return None
    d = _DIGIT_PATTERN.search(m.group(1))
    return d.group(1) if d else None


def _extract_truth(text: str) -> Optional[str]:
    """GSM8K ground truths end in ``#### N``; take the last integer."""
    digits = _DIGIT_PATTERN.findall(text or "")
    return digits[-1] if digits else None


@evaluation_test(
    completion_params=[{"model": "accounts/fireworks/models/qwen3-8b"}],
    input_dataset=[
        "https://raw.githubusercontent.com/eval-protocol/python-sdk/"
        "main/development/gsm8k_sample.jsonl"
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
)
async def test_math_answer_eval(row: EvaluationRow) -> EvaluationRow:
    """Score ``1.0`` when the model's ``<answer>`` matches ``ground_truth``."""
    last = row.last_assistant_message()
    completion = (last.content if last else "") or ""
    predicted = _extract_completion(completion)
    truth = _extract_truth(str(row.ground_truth or ""))
    score = 1.0 if (predicted is not None and predicted == truth) else 0.0
    row.evaluation_result = EvaluateResult(
        score=score,
        reason=f"predicted={predicted!r} truth={truth!r}",
    )
    return row
