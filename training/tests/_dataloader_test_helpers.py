"""Shared bootstrap for dataloader-cursor tests.

The cookbook's training packages depend on the ``fireworks`` SDK and on
``training.utils.rl.losses`` (which transitively pulls heavy deps).
``RLPromptDataset`` and ``run_rl_loop`` themselves need none of that --
the tests load those two modules directly via ``importlib.util`` and
stub the SDK surface they touch.

Centralising this avoids ~30 lines of identical bootstrap in every
cursor test file.
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
    """Minimal stand-in for ``training.utils.rl.losses.PromptGroup``.

    ``run_rl_loop`` only reads ``.rewards`` from prompt groups; tests
    that need richer fields can subclass.
    """

    rewards: list[float]


def _stub_fireworks_sdk() -> None:
    """Register no-op ``fireworks.training.sdk.*`` modules.

    ``training.utils.data`` imports ``request_with_retries`` from the
    SDK's errors submodule. The cursor logic never invokes it, so a
    pass-through stub is enough for unit tests.
    """
    if "fireworks.training.sdk.errors" in sys.modules:
        return
    errors = types.ModuleType("fireworks.training.sdk.errors")
    errors.request_with_retries = lambda fn, *args, **kwargs: fn(*args, **kwargs)
    sys.modules.setdefault("fireworks", types.ModuleType("fireworks"))
    sys.modules.setdefault(
        "fireworks.training", types.ModuleType("fireworks.training"),
    )
    sys.modules.setdefault(
        "fireworks.training.sdk", types.ModuleType("fireworks.training.sdk"),
    )
    sys.modules["fireworks.training.sdk.errors"] = errors


def _stub_rl_losses(prompt_group_cls: type) -> None:
    """Register a stub ``training.utils.rl.losses`` exposing ``PromptGroup``."""
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


def load_dataloader_test_modules(
    unique_suffix: str,
    *,
    prompt_group_cls: type = _TestPromptGroup,
) -> tuple[type, type, Any]:
    """Load ``RLPromptDataset``, ``TrainStepFns``, ``run_rl_loop`` in isolation.

    Each test file passes a unique ``unique_suffix`` so importlib does not
    return the same cached module across files (which would let one
    file's monkeypatches leak into another's).

    Returns ``(RLPromptDataset, TrainStepFns, run_rl_loop)``.
    """
    _stub_fireworks_sdk()
    _stub_rl_losses(prompt_group_cls)

    data_module = _load_module(
        f"training_utils_data_{unique_suffix}", "utils/data.py",
    )
    train_module = _load_module(
        f"training_utils_rl_train_{unique_suffix}", "utils/rl/train.py",
    )
    return (
        data_module.RLPromptDataset,
        train_module.TrainStepFns,
        train_module.run_rl_loop,
    )
