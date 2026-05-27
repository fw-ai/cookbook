"""Unit tests for ``validate_loss_path``.

The loss-path validator rejects configurations that cannot run on the
requested path before rollouts begin.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from training.utils.rl.losses import (
    LossConfig,
    get_builtin_loss_config,
    validate_loss_path,
)


def _profile() -> SimpleNamespace:
    return SimpleNamespace(
        max_supported_context_length=4096,
        deployment_shape_version="ds-v1",
        accelerator_type="NVIDIA_B200",
        accelerator_count=8,
    )


def test_validate_loss_path_client_is_always_ok():
    """``loss_path='client'`` works for any config -- never raises."""
    validate_loss_path(LossConfig(policy_loss="grpo", loss_path="client"), _profile())


def test_validate_loss_path_rejects_unknown_policy_loss_under_builtin():
    cfg = LossConfig(policy_loss="nonexistent_loss_xyz", loss_path="builtin")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported policy_loss"):
        validate_loss_path(cfg, _profile())


def test_validate_loss_path_builtin_is_eligible():
    """With kl_beta=0, builtin is valid -- no raise."""
    cfg = LossConfig(policy_loss="grpo", loss_path="builtin", kl_beta=0.0)
    validate_loss_path(cfg, _profile())

    kernel, kernel_config = get_builtin_loss_config(cfg)
    assert isinstance(kernel, str)
    assert isinstance(kernel_config, dict)


def test_validate_loss_path_allows_missing_profile():
    """No profile is required for builtin loss validation."""
    cfg = LossConfig(policy_loss="grpo", loss_path="builtin", kl_beta=0.0)
    validate_loss_path(cfg, profile=None)
