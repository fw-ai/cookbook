"""Smoke tests: verify that every cookbook module imports cleanly.

These tests catch breaking changes in tinker, tinker_cookbook, or
fireworks.training.sdk before they reach users.  Each test is a plain
import -- no API keys, no network, no GPU required.

Run with:
    pytest training/tests/test_smoke_imports.py -v
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest


# ── Recipe modules ──────────────────────────────────────────────────────────

RECIPE_MODULES = [
    "training.recipes.sft_loop",
    "training.recipes.rl_loop",
    "training.recipes.dpo_loop",
    "training.recipes.orpo_loop",
]


@pytest.mark.parametrize("module", RECIPE_MODULES)
def test_recipe_imports(module: str):
    importlib.import_module(module)


# ── Utility modules ─────────────────────────────────────────────────────────

UTIL_MODULES = [
    "training.utils",
    "training.utils.config",
    "training.utils.client",
    "training.utils.data",
    "training.utils.infra",
    "training.utils.losses",
    "training.utils.logging",
    "training.utils.checkpoint_utils",
    "training.utils.timer",
    "training.utils.validation",
    "training.utils.rl",
    "training.utils.rl.cispo",
    "training.utils.rl.common",
    "training.utils.rl.dapo",
    "training.utils.rl.grpo",
    "training.utils.rl.gspo",
    "training.utils.rl.importance_sampling",
    "training.utils.rl.losses",
    "training.utils.rl.metrics",
    "training.utils.rl.pp",
    "training.utils.rl.router_replay",
    "training.utils.rl.train",
]


@pytest.mark.parametrize("module", UTIL_MODULES)
def test_util_imports(module: str):
    importlib.import_module(module)


# ── Example modules ─────────────────────────────────────────────────────────

EXAMPLE_MODULES_WITH_ENV = [
    ("training.examples.deepmath_rl.train_deepmath", "math_verify", {"FIREWORKS_API_KEY": "test"}),
    ("training.examples.text2sql_sft.train_sft", "fireworks", {"FIREWORKS_API_KEY": "test"}),
]

EXAMPLE_MODULES = [
    "training.examples.deepmath_rl.prepare_data",
]


@pytest.mark.parametrize("module", EXAMPLE_MODULES)
def test_example_imports_without_env(module: str):
    importlib.import_module(module)


@pytest.mark.parametrize(
    "module,dep,env",
    EXAMPLE_MODULES_WITH_ENV,
    ids=[m for m, _, _ in EXAMPLE_MODULES_WITH_ENV],
)
def test_example_imports(module: str, dep: str, env: dict):
    pytest.importorskip(dep)
    saved = {}
    for k, v in env.items():
        saved[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        importlib.import_module(module)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── tinker types used by the cookbook ────────────────────────────────────────

TINKER_ATTRS = [
    "Datum",
    "ModelInput",
    "TensorData",
    "AdamParams",
]


@pytest.mark.parametrize("attr", TINKER_ATTRS)
def test_tinker_types(attr: str):
    import tinker

    obj = getattr(tinker, attr, None)
    assert obj is not None, f"tinker.{attr} not found (tinker {tinker.__version__})"


# ── fireworks.training.sdk symbols used by the cookbook ──────────────────────

SDK_IMPORTS = [
    ("fireworks.training.sdk", "DeploymentManager"),
    ("fireworks.training.sdk", "TrainerJobManager"),
    ("fireworks.training.sdk.deployment", "DEFAULT_DELTA_COMPRESSION"),
    ("fireworks.training.sdk.deployment", "DeploymentSampler"),
    ("fireworks.training.sdk.deployment", "DeploymentConfig"),
    ("fireworks.training.sdk.deployment", "DeploymentInfo"),
    ("fireworks.training.sdk.deployment", "DeploymentManager"),
    ("fireworks.training.sdk.weight_syncer", "WeightSyncer"),
    ("fireworks.training.sdk.client", "FiretitanServiceClient"),
    ("fireworks.training.sdk.client", "FiretitanTrainingClient"),
    ("fireworks.training.sdk.trainer", "TrainerJobManager"),
    ("fireworks.training.sdk.trainer", "TrainerServiceEndpoint"),
    ("fireworks.training.sdk.trainer", "TrainingShapeProfile"),
    ("fireworks.training.sdk.errors", "format_sdk_error"),
    ("fireworks.training.sdk.errors", "DOCS_SDK"),
    ("fireworks.training.sdk.errors", "request_with_retries"),
]


@pytest.mark.parametrize("module,attr", SDK_IMPORTS, ids=[f"{m}.{a}" for m, a in SDK_IMPORTS])
def test_sdk_symbols(module: str, attr: str):
    mod = importlib.import_module(module)
    obj = getattr(mod, attr, None)
    assert obj is not None, f"{module}.{attr} not found"


# ── Recipe Config dataclasses instantiate with defaults ─────────────────────


def test_sft_config_defaults():
    from training.recipes.sft_loop import Config

    cfg = Config(log_path="/tmp/test")
    assert cfg.base_model


def test_rl_config_defaults():
    from training.recipes.rl_loop import Config

    cfg = Config(log_path="/tmp/test")
    assert cfg.base_model


def test_dpo_config_defaults():
    from training.recipes.dpo_loop import Config

    cfg = Config(log_path="/tmp/test")
    assert cfg.base_model


def test_orpo_config_defaults():
    from training.recipes.orpo_loop import Config

    cfg = Config(log_path="/tmp/test")
    assert cfg.base_model


# ── Cookbook __all__ consistency ─────────────────────────────────────────────


def test_utils_all_resolvable():
    """Every name in training.utils.__all__ must be importable."""
    import training.utils as utils_pkg

    for name in utils_pkg.__all__:
        assert hasattr(utils_pkg, name), f"training.utils.__all__ lists '{name}' but it cannot be resolved"


def test_utils_rl_all_resolvable():
    """Every name in training.utils.rl.__all__ must be importable."""
    import training.utils.rl as rl_pkg

    for name in rl_pkg.__all__:
        assert hasattr(rl_pkg, name), f"training.utils.rl.__all__ lists '{name}' but it cannot be resolved"


# ── tinker_cookbook (optional — only runs if installed) ──────────────────────

TINKER_COOKBOOK_MODULES = [
    "tinker_cookbook.rl.train",
    "tinker_cookbook.supervised.train",
    "tinker_cookbook.supervised.data",
    "tinker_cookbook.tokenizer_utils",
    "tinker_cookbook.renderers",
]


@pytest.mark.parametrize("module", TINKER_COOKBOOK_MODULES)
def test_tinker_cookbook_imports(module: str):
    pytest.importorskip("tinker_cookbook")
    importlib.import_module(module)
