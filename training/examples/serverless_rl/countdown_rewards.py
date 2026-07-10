"""Countdown Game reward functions.

The Countdown task: given a target number and a set of source numbers, write an
arithmetic expression (using ``+ - * /`` and each source number exactly once)
that evaluates to the target. The model is asked to reason inside
``<think>...</think>`` and emit its final expression inside
``<answer>...</answer>``.

``composite_reward`` is the signal the RL loop optimizes. It is intentionally
shaped so partial progress is rewarded: emitting a well-formed answer, using the
right numbers, and hitting the target each earn credit. This vendored copy keeps
the example fully standalone -- no monorepo imports.
"""

from __future__ import annotations

import json
import re


def extract_answer(text: str) -> str | None:
    """Extract the equation from the last ``<answer>...</answer>`` tag."""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


def safe_eval_equation(equation: str) -> float | None:
    """Safely evaluate an arithmetic equation with ``+ - * /``, parens, and digits."""
    cleaned = equation.replace(" ", "")
    if not re.match(r"^[\d+\-*/().]+$", cleaned):
        return None
    if "**" in cleaned or "//" in cleaned:
        return None
    try:
        return float(eval(cleaned, {"__builtins__": None}, {}))  # noqa: S307
    except Exception:
        return None


def check_numbers_used(equation: str, numbers: list[int]) -> bool:
    """Check that the equation uses exactly the given numbers."""
    used = sorted(int(x) for x in re.findall(r"\d+", equation))
    return used == sorted(numbers)


def parse_ground_truth(ground_truth: str | dict) -> tuple[list[int], int]:
    """Parse ground_truth into ``(numbers, target)``."""
    if isinstance(ground_truth, str):
        gt = json.loads(ground_truth)
    else:
        gt = ground_truth
    numbers = gt.get("numbers") or gt.get("nums")
    if numbers is None:
        raise KeyError("ground_truth must include 'numbers' or 'nums'")
    return list(numbers), int(gt["target"])


def format_reward(response: str) -> float:
    """Reward Countdown answer format."""
    has_think = bool(re.search(r"<think>", response, re.IGNORECASE))
    has_answer = extract_answer(response) is not None
    if has_think and has_answer:
        return 1.0
    if has_answer:
        return 0.5
    return 0.0


def accuracy_reward(response: str, ground_truth: str | dict) -> float:
    """Return 1.0 when the answer reaches the target using exactly the row numbers."""
    numbers, target = parse_ground_truth(ground_truth)
    equation = extract_answer(response)
    if equation is None:
        return 0.0
    if not check_numbers_used(equation, numbers):
        return 0.0
    result = safe_eval_equation(equation)
    if result is None or abs(result - target) >= 1e-6:
        return 0.0
    return 1.0


def composite_reward(response: str, ground_truth: str | dict) -> float:
    """Composite Countdown reward: format + number usage + correctness.

    - ``+0.1`` for emitting a parseable ``<answer>`` at all.
    - ``+0.2`` for using exactly the source numbers.
    - ``+0.7`` for an expression that evaluates to the target.

    A fully correct answer scores ``1.0``; the intermediate credit gives the RL
    loop a dense-enough signal to climb from a cold start.
    """
    numbers, target = parse_ground_truth(ground_truth)
    score = 0.0

    equation = extract_answer(response)
    if equation is None:
        return score

    score += 0.1

    numbers_valid = check_numbers_used(equation, numbers)
    if numbers_valid:
        score += 0.2

    result = safe_eval_equation(equation)
    if numbers_valid and result is not None and abs(result - target) < 1e-6:
        score += 0.7

    return score
