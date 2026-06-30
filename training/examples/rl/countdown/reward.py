"""Reward helpers for the Countdown arithmetic task."""

from __future__ import annotations

import ast
import operator
import re
from typing import Any

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def extract_answer(text: str) -> str | None:
    """Extract the equation from the last ``<answer>...</answer>`` tag."""
    matches = list(_ANSWER_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def strip_think_blocks(text: str) -> str:
    """Drop reasoning blocks before reward parsing."""
    return _THINK_BLOCK_RE.sub("", text)


def _safe_eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return float(_ALLOWED_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return float(_ALLOWED_UNARY[type(node.op)](_safe_eval_node(node.operand)))
    raise ValueError(f"unsupported arithmetic expression: {type(node).__name__}")


def safe_eval_equation(equation: str) -> float | None:
    """Evaluate arithmetic with digits, parentheses, and ``+ - * /`` only."""
    cleaned = equation.replace(" ", "")
    if not re.fullmatch(r"[\d+\-*/().]+", cleaned):
        return None
    if "**" in cleaned or "//" in cleaned:
        return None
    try:
        return _safe_eval_node(ast.parse(cleaned, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None


def check_numbers_used(equation: str, numbers: list[int]) -> bool:
    """Return true when the equation uses exactly the provided numbers."""
    used = sorted(int(value) for value in re.findall(r"\d+", equation))
    return used == sorted(int(value) for value in numbers)


def parse_ground_truth(row: dict[str, Any]) -> tuple[list[int], int]:
    numbers = row.get("numbers") or row.get("nums")
    if numbers is None:
        raise KeyError("Countdown row must include 'numbers' or 'nums'")
    return [int(value) for value in numbers], int(row["target"])


def composite_reward(completion: str, row: dict[str, Any]) -> float:
    """Score answer format, number usage, and arithmetic correctness."""
    completion = strip_think_blocks(completion)
    equation = extract_answer(completion)
    if equation is None:
        return 0.0

    numbers, target = parse_ground_truth(row)
    score = 0.1
    numbers_valid = check_numbers_used(equation, numbers)
    if numbers_valid:
        score += 0.2

    result = safe_eval_equation(equation)
    if numbers_valid and result is not None and abs(result - target) < 1e-6:
        score += 0.7
    return score
