"""Smoke test: deepmath migrated to pluggable rollout_fn.

Per AC-11 / AC-12 follow-through (Codex R1 review), the legacy
``rl_loop.reward_fn = deepmath_reward`` mutation must be replaced with
the pluggable ``rl_loop.main(..., rollout_fn=...)`` keyword and a
renderer-backed rollout function.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


_TRAIN = Path(__file__).resolve().parent / "train_deepmath.py"


def test_uses_single_turn_renderer_rollout():
    text = _TRAIN.read_text()
    assert "single_turn_renderer_rollout" in text, (
        "deepmath/train_deepmath.py must use single_turn_renderer_rollout"
    )


def test_calls_main_with_rollout_fn_kwarg():
    tree = ast.parse(_TRAIN.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            kwargs = [kw.arg for kw in node.keywords]
            if "rollout_fn" in kwargs:
                return
    raise AssertionError(
        "deepmath/train_deepmath.py must call rl_loop.main(..., rollout_fn=...)"
    )


def test_no_runtime_mutation_of_rl_loop_reward_fn():
    """The legacy entrypoint mutated ``rl_loop.reward_fn`` to inject the
    grader.  That tied the example to recipe-internal naming and is not
    valid under the pluggable rollout_fn surface."""
    text = _TRAIN.read_text()
    assert not re.search(r"\brl_loop\.reward_fn\s*=", text), (
        "deepmath/train_deepmath.py must not mutate rl_loop.reward_fn; "
        "use rl_loop.main(..., rollout_fn=...) instead."
    )
