"""Unit tests for ``validate_loss_path``.

The PP guard (raises when builtin loss requested with PP > 1) was
previously tested via a heavy-mock recipe test. The logic lives in the
loss-path validator -- exercise it directly here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from training.utils.rl.losses import (
    LossConfig,
    get_builtin_loss_config,
    validate_loss_path,
)


def _profile(*, pp: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_parallelism=pp,
        max_supported_context_length=4096,
        deployment_shape_version="ds-v1",
        accelerator_type="NVIDIA_B200",
        accelerator_count=8,
    )


def test_validate_loss_path_client_is_always_ok():
    """``loss_path='client'`` works for any config -- never raises."""
    validate_loss_path(LossConfig(policy_loss="grpo", loss_path="client"), _profile(pp=4))


def test_validate_loss_path_rejects_unknown_policy_loss_under_builtin():
    cfg = LossConfig(policy_loss="nonexistent_loss_xyz", loss_path="builtin")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported policy_loss"):
        validate_loss_path(cfg, _profile())


def test_validate_loss_path_raises_on_pipeline_parallelism():
    """Builtin loss kernels do not support pipeline parallelism > 1."""
    cfg = LossConfig(policy_loss="grpo", loss_path="builtin")
    with pytest.raises(ValueError, match=r"pipeline_parallelism=4 > 1"):
        validate_loss_path(cfg, _profile(pp=4))


def test_validate_loss_path_pp_eq_1_is_eligible():
    """With PP=1 and kl_beta=0, builtin is valid -- no raise."""
    cfg = LossConfig(policy_loss="grpo", loss_path="builtin", kl_beta=0.0)
    validate_loss_path(cfg, _profile(pp=1))

    kernel, kernel_config = get_builtin_loss_config(cfg)
    assert isinstance(kernel, str)
    assert isinstance(kernel_config, dict)


def test_validate_loss_path_skips_pp_check_without_profile():
    """No profile = no PP info to check -- must not raise on PP grounds."""
    cfg = LossConfig(policy_loss="grpo", loss_path="builtin", kl_beta=0.0)
    validate_loss_path(cfg, profile=None)
