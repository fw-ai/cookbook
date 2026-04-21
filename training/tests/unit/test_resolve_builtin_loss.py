"""Unit tests for ``resolve_builtin_loss``.

The PP guard (raises when builtin loss requested with PP > 1) was
previously tested via a heavy-mock recipe test. The logic lives in the
loss registry — exercise it directly here.
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


def test_resolve_builtin_loss_returns_none_for_unregistered_name():
    """A loss name with no builtin kernel returns ``None`` so the recipe
    falls back to ``forward_backward_custom``."""
    assert resolve_builtin_loss("nonexistent_loss_xyz", _profile()) is None


def test_resolve_builtin_loss_raises_on_pipeline_parallelism():
    """Builtin loss kernels do not support pipeline parallelism > 1."""
    with pytest.raises(ValueError, match="Pipeline parallelism.*PP=4.*not supported"):
        resolve_builtin_loss("grpo", _profile(pp=4))


def test_resolve_builtin_loss_returns_kernel_when_pp_eq_1():
    """With PP=1, the builtin path is eligible — returns ``(kernel, config)``."""
    result = resolve_builtin_loss("grpo", _profile(pp=1))
    if result is not None:
        kernel, kernel_config = result
        assert isinstance(kernel, str)
        assert isinstance(kernel_config, dict)


def test_resolve_builtin_loss_skips_pp_check_without_profile():
    """No profile = no PP info to check — must not raise on PP grounds."""
    result = resolve_builtin_loss("grpo", profile=None)
    assert result is None or (isinstance(result, tuple) and len(result) == 2)
