"""Smoke test: the sync on-policy example must use the sync recipe.

Per AC-11, ``examples/rl/single_turn_sync_on_policy/train.py`` is the
synchronous on-policy deliverable.  It must import ``training.recipes.rl_loop``
directly and must NOT import ``async_rl_loop`` (which would silently
substitute the async recipe).
"""

from __future__ import annotations

import ast
from pathlib import Path


_TRAIN_PATH = Path(__file__).resolve().parent / "train.py"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module)
    return out


def test_imports_sync_recipe():
    mods = _imported_modules(_TRAIN_PATH)
    assert "training.recipes.rl_loop" in mods, (
        "single_turn_sync_on_policy/train.py must import training.recipes.rl_loop"
    )


def test_does_not_import_async_recipe():
    mods = _imported_modules(_TRAIN_PATH)
    assert not any(m.startswith("training.recipes.async_rl_loop") for m in mods), (
        "single_turn_sync_on_policy/train.py must NOT import async_rl_loop; "
        "the sync trainer is the AC-11 deliverable here."
    )


def test_uses_single_turn_renderer_rollout():
    mods = _imported_modules(_TRAIN_PATH)
    assert "training.utils.rl.rollout" in mods


def test_calls_main_with_rollout_fn_kwarg():
    """Ensure the example calls main() with the rollout_fn keyword argument."""
    tree = ast.parse(_TRAIN_PATH.read_text())
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "main":
            kwarg_names = [kw.arg for kw in node.keywords]
            if "rollout_fn" in kwarg_names:
                found = True
                break
    assert found, (
        "main(...) must be called with the rollout_fn keyword argument so the "
        "sync recipe runs the renderer-backed rollout instead of the default."
    )
