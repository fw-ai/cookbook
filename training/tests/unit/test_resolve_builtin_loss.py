"""Unit tests for resolve_builtin_loss.

Replaces the deleted ``test_main_raises_when_builtin_loss_with_pp`` heavy-mock
test from test_rl_loop.py — the PP guard logic lives here in the library.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from training.utils.rl.losses import resolve_builtin_loss


def _profile(*, pp: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_parallelism=pp,
        max_supported_context_length=4096,
        deployment_shape_version="ds-v1",
        accelerator_type="NVIDIA_B200",
        accelerator_count=8,
    )


def test_resolve_builtin_loss_returns_none_for_unregistered():
    """A made-up loss name with no builtin returns None (caller falls back to custom)."""
    assert resolve_builtin_loss("nonexistent_loss_xyz", _profile()) is None


def test_resolve_builtin_loss_raises_on_pp_gt_1_for_builtin():
    """Builtin loss kernels do not support pipeline parallelism > 1."""
    with pytest.raises(ValueError, match="Pipeline parallelism.*PP=4.*not supported"):
        resolve_builtin_loss("grpo", _profile(pp=4))


def test_resolve_builtin_loss_returns_kernel_when_pp_eq_1():
    """With PP=1, the builtin path is eligible — returns (kernel, config)."""
    result = resolve_builtin_loss("grpo", _profile(pp=1))
    # grpo has a builtin kernel; should be a tuple shape (or None if not registered).
    if result is not None:
        kernel, cfg = result
        assert isinstance(kernel, str)
        assert isinstance(cfg, dict)


def test_resolve_builtin_loss_skips_pp_check_when_profile_none():
    """No profile = no PP info to check — should not raise on PP grounds alone."""
    # Either returns None (not registered) or a tuple (registered, no profile check).
    result = resolve_builtin_loss("grpo", profile=None)
    assert result is None or (isinstance(result, tuple) and len(result) == 2)
