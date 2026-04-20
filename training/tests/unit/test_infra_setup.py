"""Unit tests for the composable infra builders in utils/rl/infra_setup.

Each builder is tested in isolation with minimal fakes — no end-to-end
recipe wiring. The point is to cover the branching that lives inside
the library, not to verify mocks were called.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from training.utils.config import ConcurrencyConfig
from training.utils.rl.infra_setup import (
    make_concurrency_controller,
    provision_trainer_pair,
    resolve_policy_profile,
    resolve_reference_profile,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _profile(*, pp: int = 1, max_seq: int = 4096, dep_shape: str | None = "ds-v1") -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_parallelism=pp,
        max_supported_context_length=max_seq,
        deployment_shape_version=dep_shape,
        accelerator_type="NVIDIA_B200",
        accelerator_count=1,
    )


# ---------------------------------------------------------------------------
# resolve_policy_profile
# ---------------------------------------------------------------------------


def test_resolve_policy_profile_uses_explicit_shape_id():
    rlor = MagicMock()
    rlor.resolve_training_profile.return_value = _profile(max_seq=8192)

    shape_id, profile = resolve_policy_profile(
        rlor,
        shape_id="accounts/x/trainingShapes/explicit",
        base_model="m",
        lora_rank=0,
        max_seq_len=None,
    )

    assert shape_id == "accounts/x/trainingShapes/explicit"
    assert profile.max_supported_context_length == 8192
    rlor.resolve_training_profile.assert_called_once_with("accounts/x/trainingShapes/explicit")


def test_resolve_policy_profile_auto_selects_when_shape_id_none(monkeypatch):
    rlor = MagicMock()
    rlor.resolve_training_profile.return_value = _profile()
    auto_calls: list[dict] = []

    def fake_auto(rlor_arg, *, base_model, trainer_role, lora_rank, max_seq_len):
        auto_calls.append({
            "base_model": base_model,
            "trainer_role": trainer_role,
            "lora_rank": lora_rank,
            "max_seq_len": max_seq_len,
        })
        return "auto-selected-shape"

    monkeypatch.setattr(
        "training.utils.rl.infra_setup.auto_select_training_shape", fake_auto,
    )

    shape_id, profile = resolve_policy_profile(
        rlor, shape_id=None,
        base_model="qwen3-8b", lora_rank=64, max_seq_len=2048,
    )

    assert shape_id == "auto-selected-shape"
    assert auto_calls == [{
        "base_model": "qwen3-8b",
        "trainer_role": "policy",
        "lora_rank": 64,
        "max_seq_len": 2048,
    }]


# ---------------------------------------------------------------------------
# resolve_reference_profile
# ---------------------------------------------------------------------------


def test_resolve_reference_profile_returns_none_when_kl_disabled():
    """No reference needed if KL is off."""
    rlor = MagicMock()
    shape_id, profile = resolve_reference_profile(
        rlor, shape_id=None,
        base_model="m", lora_rank=0, max_seq_len=4096, kl_beta=0.0,
    )
    assert shape_id is None
    assert profile is None
    rlor.resolve_training_profile.assert_not_called()


def test_resolve_reference_profile_returns_none_for_lora_with_kl():
    """LoRA + KL: policy serves reference via base-only handle, no separate trainer."""
    rlor = MagicMock()
    shape_id, profile = resolve_reference_profile(
        rlor, shape_id=None,
        base_model="m", lora_rank=64, max_seq_len=4096, kl_beta=0.1,
    )
    assert shape_id is None
    assert profile is None
    rlor.resolve_training_profile.assert_not_called()


def test_resolve_reference_profile_auto_selects_when_full_param_with_kl(monkeypatch):
    """Full-param (lora_rank=0) + KL > 0 + no explicit shape → auto-select."""
    rlor = MagicMock()
    rlor.resolve_training_profile.return_value = _profile()
    monkeypatch.setattr(
        "training.utils.rl.infra_setup.auto_select_training_shape",
        lambda rlor_arg, **kw: "auto-ref-shape",
    )

    shape_id, profile = resolve_reference_profile(
        rlor, shape_id=None,
        base_model="m", lora_rank=0, max_seq_len=4096, kl_beta=0.1,
    )

    assert shape_id == "auto-ref-shape"
    assert profile is not None


def test_resolve_reference_profile_uses_explicit_shape_when_set():
    rlor = MagicMock()
    rlor.resolve_training_profile.return_value = _profile()

    shape_id, profile = resolve_reference_profile(
        rlor, shape_id="accounts/x/trainingShapes/explicit-ref",
        base_model="m", lora_rank=0, max_seq_len=4096, kl_beta=0.0,
    )
    # Even with kl_beta=0, an explicit shape is honored.
    assert shape_id == "accounts/x/trainingShapes/explicit-ref"
    assert profile is not None


def test_resolve_reference_profile_skips_auto_when_disabled():
    rlor = MagicMock()
    shape_id, profile = resolve_reference_profile(
        rlor, shape_id=None,
        base_model="m", lora_rank=0, max_seq_len=4096, kl_beta=0.1,
        auto_select_when_kl=False,
    )
    assert shape_id is None
    assert profile is None
    rlor.resolve_training_profile.assert_not_called()


# ---------------------------------------------------------------------------
# provision_trainer_pair
# ---------------------------------------------------------------------------


def test_provision_trainer_pair_skips_reference_when_none(monkeypatch):
    rlor = MagicMock()
    calls: list[dict] = []

    def fake_create(rlor_arg, *, display_name, forward_only=False, **kwargs):
        calls.append({"display_name": display_name, "forward_only": forward_only})
        return SimpleNamespace(
            job_id=f"job-{display_name}",
            job_name=f"jobs/{display_name}",
        )

    monkeypatch.setattr("training.utils.rl.infra_setup.create_trainer_job", fake_create)

    policy_ep, ref_ep = provision_trainer_pair(
        rlor,
        base_model="m",
        infra_config=SimpleNamespace(),
        policy_profile=_profile(),
        ref_profile=None,
        lora_rank=0,
        max_seq_len=4096,
        learning_rate=1e-5,
    )

    assert ref_ep is None
    assert policy_ep.job_id == "job-grpo-policy"
    assert calls == [{"display_name": "grpo-policy", "forward_only": False}]


def test_provision_trainer_pair_provisions_both_with_reference(monkeypatch):
    rlor = MagicMock()
    calls: list[dict] = []

    def fake_create(rlor_arg, *, display_name, forward_only=False, **kwargs):
        calls.append({"display_name": display_name, "forward_only": forward_only})
        return SimpleNamespace(job_id=f"job-{display_name}", job_name=display_name)

    monkeypatch.setattr("training.utils.rl.infra_setup.create_trainer_job", fake_create)

    policy_ep, ref_ep = provision_trainer_pair(
        rlor,
        base_model="m",
        infra_config=SimpleNamespace(),
        policy_profile=_profile(),
        ref_profile=_profile(),
        lora_rank=0,
        max_seq_len=4096,
        learning_rate=1e-5,
    )

    assert policy_ep.job_id == "job-grpo-policy"
    assert ref_ep is not None and ref_ep.job_id == "job-grpo-reference"
    # Both got created; reference is forward_only.
    by_role = {c["display_name"]: c for c in calls}
    assert by_role["grpo-reference"]["forward_only"] is True
    assert by_role["grpo-policy"]["forward_only"] is False


def test_provision_trainer_pair_propagates_failures(monkeypatch):
    rlor = MagicMock()

    def fake_create(rlor_arg, *, display_name, forward_only=False, **kwargs):
        if display_name == "grpo-reference":
            raise RuntimeError("ref creation blew up")
        return SimpleNamespace(job_id="ok", job_name="ok")

    monkeypatch.setattr("training.utils.rl.infra_setup.create_trainer_job", fake_create)

    with pytest.raises(RuntimeError, match="Trainer creation failed"):
        provision_trainer_pair(
            rlor,
            base_model="m",
            infra_config=SimpleNamespace(),
            policy_profile=_profile(),
            ref_profile=_profile(),
            lora_rank=0,
            max_seq_len=4096,
            learning_rate=1e-5,
        )


# ---------------------------------------------------------------------------
# make_concurrency_controller
# ---------------------------------------------------------------------------


def test_make_concurrency_controller_fixed_returns_none():
    cc = ConcurrencyConfig(mode="fixed")
    deploy_mgr = MagicMock()
    deploy_cfg = SimpleNamespace()
    assert make_concurrency_controller(cc, deploy_mgr, deploy_cfg) is None


def test_make_concurrency_controller_adaptive_uses_initial_window(monkeypatch):
    cc = ConcurrencyConfig(
        mode="adaptive", initial_window=24,
        min_window=2, max_window=64, prefill_queue_target=0.7,
    )
    deploy_mgr = MagicMock()
    monkeypatch.setattr(
        "training.utils.rl.infra_setup.get_deployment_gpu_count",
        lambda *a, **kw: 4,
    )

    controller = make_concurrency_controller(cc, deploy_mgr, SimpleNamespace())
    assert controller is not None
    # initial_window respected (not derived from gpu_count).
    assert controller.window_size == 24
    assert controller._max_window == 64
    assert controller._min_window == 2


def test_make_concurrency_controller_adaptive_default_window_from_gpu(monkeypatch):
    cc = ConcurrencyConfig(mode="adaptive", initial_window=None)
    deploy_mgr = MagicMock()
    monkeypatch.setattr(
        "training.utils.rl.infra_setup.get_deployment_gpu_count",
        lambda *a, **kw: 8,
    )
    controller = make_concurrency_controller(cc, deploy_mgr, SimpleNamespace())
    # Default = 8 slots/gpu * 8 gpus = 64.
    assert controller.window_size == 64


def test_make_concurrency_controller_unknown_mode_raises():
    cc = ConcurrencyConfig(mode="bananas")
    with pytest.raises(ValueError, match="Unknown concurrency mode"):
        make_concurrency_controller(cc, MagicMock(), SimpleNamespace())
