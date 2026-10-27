"""Code-repair reward: share of hidden test cases the model's fix passes.

The reply's last fenced code block that defines the entry point (else its last
fenced block) is graded by :func:`sandbox.grade`. Reward is
``passed / num_cases``; no code block, a syntax error, an import-time error,
a timeout or a crash all score 0. Returning the buggy code unchanged earns its
original pass rate, which GRPO's within-group normalization handles: fixes
score above the group mean, regressions below.
"""

from __future__ import annotations

import re
from typing import Any

from training.examples.rl.code_repair.sandbox import grade

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_FENCE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_code(response: str, entry_point: str) -> str | None:
    text = _THINK_BLOCK.sub("", response)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    blocks = _FENCE.findall(text)
    if not blocks:
        return None
    defining = [b for b in blocks if re.search(rf"^\s*def {re.escape(entry_point)}\s*\(", b, re.MULTILINE)]
    return (defining or blocks)[-1]


async def code_repair_reward(response: str, ground_truth: dict[str, Any]) -> float:
    code = extract_code(response, ground_truth["entry_point"])
    if code is None:
        return 0.0
    result = await grade(code, ground_truth["test"], ground_truth["entry_point"])
    return min(1.0, result.passed / ground_truth["num_cases"])
