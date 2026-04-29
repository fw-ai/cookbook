"""Shared bootstrap for dataloader-cursor tests.
Loads ``utils/dataloader_cursor.py`` and ``utils/rl/train.py`` in isolation, stubbing the SDK.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TESTS_DIR = Path(__file__).parent
_TRAINING_DIR = _TESTS_DIR.parent


@dataclass
class _TestPromptGroup:
    """Minimal stand-in for ``training.utils.rl.losses.PromptGroup`` (only ``.rewards`` is read)."""

    rewards: list[float]


def _stub_fireworks_sdk() -> None:
    """Pass-through stub for ``fireworks.training.sdk.errors.request_with_retries``."""
    if "fireworks.training.sdk.errors" in sys.modules:
        return
    errors = types.ModuleType("fireworks.training.sdk.errors")
    errors.request_with_retries = lambda fn, *args, **kwargs: fn(*args, **kwargs)
    sys.modules.setdefault("fireworks", types.ModuleType("fireworks"))
    sys.modules.setdefault("fireworks.training", types.ModuleType("fireworks.training"))
    sys.modules.setdefault("fireworks.training.sdk", types.ModuleType("fireworks.training.sdk"))
    sys.modules["fireworks.training.sdk.errors"] = errors


def _stub_rl_losses(prompt_group_cls: type) -> None:
    if "training.utils.rl.losses" in sys.modules:
        return
    losses = types.ModuleType("training.utils.rl.losses")
    losses.PromptGroup = prompt_group_cls
    sys.modules["training.utils.rl.losses"] = losses


def _load_module(unique_name: str, relpath: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        unique_name, _TRAINING_DIR / relpath,
    )
    assert spec and spec.loader, f"failed to spec {relpath}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


def load_cursor() -> type:
    """Return ``RawRowCursor`` loaded directly (no SDK deps)."""
    m = _load_module("dataloader_cursor_test_load", "utils/dataloader_cursor.py")
    return m.RawRowCursor


def load_rl_train(
    unique_suffix: str,
    *,
    prompt_group_cls: type = _TestPromptGroup,
) -> tuple[type, Any]:
    """Return ``(TrainStepFns, run_rl_loop)`` loaded under a unique importlib name."""
    _stub_fireworks_sdk()
    _stub_rl_losses(prompt_group_cls)
    m = _load_module(f"rl_train_test_{unique_suffix}", "utils/rl/train.py")
    return m.TrainStepFns, m.run_rl_loop
