"""Default reward and filter functions for RL training.

Customers override these by passing their own callables to ``Config``.
"""

from __future__ import annotations

import re
from typing import Optional

from training.utils.rl.losses import PromptGroup


def extract_answer(text: str) -> Optional[str]:
    """Extract a numeric answer from ``<answer>...</answer>`` tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    digits = re.search(r"(-?\d+)", match.group(1))
    return digits.group(1) if digits else None


def default_math_reward(completion: str, row: dict) -> float:
    """Return 1.0 if the model's numeric answer matches the ground truth."""
    predicted = extract_answer(completion)
    truth = extract_answer(str(row.get("ground_truth", "")))
    if predicted is None or truth is None:
        return 0.0
    return 1.0 if predicted == truth else 0.0


def default_variance_filter(pg: PromptGroup) -> bool:
    """Reject groups where all rewards are identical (zero-variance)."""
    return len(set(pg.rewards)) > 1
