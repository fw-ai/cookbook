"""GSM8K reward function for the multi-turn message-in recipe.

Mirrors AReaL's ``examples/multi_turn_math/gsm8k_rl_mt.py::gsm8k_reward_fn``:
parse the model's completion (``\\boxed{...}``) and the GSM8K ground-truth
answer string (whose final number follows ``#### ``) with ``math_verify`` and
return ``1.0`` on a verified match, ``0.0`` otherwise.

A pure-regex numeric fallback handles the common case where ``math_verify``'s
LaTeX parsing rejects an otherwise-correct integer answer.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\boxed\s*\{", re.DOTALL)
_GSM8K_GT_RE = re.compile(r"####\s*([\-\+]?[\d,]*\.?\d+)")
_NUMERIC_TOL = 1e-6


def _extract_boxed(text: str) -> str | None:
    """Return the content of the LAST ``\\boxed{...}`` in ``text``.

    Walks brace depth so nested braces (``\\frac{1}{2}``) are preserved.
    """
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    start = last.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1].strip()


def _gsm8k_ground_truth_number(answer: str) -> str | None:
    """Strip the GSM8K chain-of-thought; return the final number after ``#### ``."""
    m = _GSM8K_GT_RE.search(answer)
    if not m:
        return None
    return m.group(1).replace(",", "")


def _try_numeric_match(pred: str, gt: str) -> bool:
    try:
        return abs(float(pred.replace(",", "")) - float(gt.replace(",", ""))) < _NUMERIC_TOL
    except (ValueError, OverflowError):
        return False


def gsm8k_reward(completion: str, answer: str) -> float:
    """Return ``1.0`` if the model's boxed answer matches the GSM8K ground truth.

    The cheap numeric path catches the GSM8K-typical case (integer answers).
    Falls through to ``math_verify`` (a project dep) for fractions / surds /
    LaTeX.  Returns ``0.0`` on any failure -- never raises.
    """
    pred = _extract_boxed(completion)
    if pred is None:
        return 0.0
    gt_num = _gsm8k_ground_truth_number(answer)
    if gt_num is not None and _try_numeric_match(pred, gt_num):
        return 1.0

    try:
        from math_verify import parse as math_parse, verify as math_verify_fn

        pred_parsed = math_parse(f"\\boxed{{{pred}}}")
        gt_parsed = math_parse(answer)
        if pred_parsed and gt_parsed and math_verify_fn(gt_parsed, pred_parsed):
            return 1.0
    except Exception:
        logger.debug("math_verify failed; pred=%r gt=%r", pred, answer, exc_info=True)

    return 0.0
