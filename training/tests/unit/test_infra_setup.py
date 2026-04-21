"""Unit tests for the bundled setup_infra entry point.

setup_infra owns all the control flow (shape resolution, parallel
trainer provisioning, LoRA shared-reference, deployment, sampler,
weight syncer). These tests exercise each branch in isolation by
patching the SDK boundary and verifying setup_infra wires things
together correctly — particularly the LoRA shared-reference path
that keeps a single trainer alive for both policy and reference.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import training.utils.rl.infra_setup as infra_setup_mod
from training.utils.config import (
    ConcurrencyConfig,
    DeployConfig,
    InfraConfig,
    WeightSyncConfig,
)
from training.utils.rl.infra_setup import Infra, setup_infra


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _profile(*, max_seq: int = 4096, dep_shape: str | None = "ds-v1") -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_parallelism=1,
        max_supported_context_length=max_seq,
        deployment_shape_version=dep_shape,
        accelerator_type="NVIDIA_B200",
        accelerator_count=1,
    )


class _FakeClient:
    """Minimal stand-in for ReconnectableClient that records construction args."""

    instances: list["_FakeClient"] = []

    def __init__(self, rlor_mgr, job_id, base_model, lora_rank=0, **kwargs):
        self.job_id = job_id
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.kwargs = kwargs
        self.inner = object()
        self.base_ref_calls = 0
        _FakeClient.instances.append(self)

    def create_base_reference(self):
        self.base_ref_calls += 1
        return SimpleNamespace(kind="base_shared", policy=self, close=lambda: None)

    def close(self, timeout=5.0):
        pass


def _trainer_ep(job_id: str = "job-x") -> SimpleNamespace:
    return SimpleNamespace(
        job_id=job_id, job_name=f"jobs/{job_id}", base_url="https://t.unit",
    )


def _make_cfg(
    *,
    lora_rank: int = 0,
    training_shape_id: str | None = "shape-policy",
    ref_training_shape_id: str | None = None,
    deployment_id: str | None = "dep-1",
) -> SimpleNamespace:
    return SimpleNamespace(
        base_model="accounts/test/models/m",
        rollout_base_model=None,
        lora_rank=lora_rank,
        max_seq_len=None,
        learning_rate=1e-5,
        step_timeout=0,
        infra=InfraConfig(
            training_shape_id=training_shape_id,
            ref_training_shape_id=ref_training_shape_id,
        ),
        deployment=DeployConfig(deployment_id=deployment_id, tokenizer_model="Q/M"),
        weight_sync=WeightSyncConfig(),
        concurrency=ConcurrencyConfig(mode="fixed"),
        policy_job_id=None,
        reference_job_id=None,
    )


@pytest.fixture
def patch_sdk(monkeypatch):
    """Patch all SDK boundary calls inside infra_setup with fakes."""
    _FakeClient.instances = []

    trainer_calls: list[dict] = []

    def fake_create_trainer(_rlor, *, display_name, forward_only=False, **kwargs):
        job_id = "ref-job" if "reference" in display_name else "policy-job"
        trainer_calls.append({
            "display_name": display_name,
            "forward_only": forward_only,
            "lora_rank": kwargs.get("lora_rank"),
            "job_id_arg": kwargs.get("job_id"),
        })
        return _trainer_ep(job_id)

    monkeypatch.setattr(infra_setup_mod, "create_trainer_job", fake_create_trainer)
    monkeypatch.setattr(infra_setup_mod, "ReconnectableClient", _FakeClient)
    monkeypatch.setattr(
        infra_setup_mod, "setup_or_reattach_deployment",
        lambda *a, **kw: SimpleNamespace(inference_model="accounts/test/models/deployed"),
    )
    monkeypatch.setattr(
        infra_setup_mod, "auto_select_training_shape",
        lambda _rlor, **kw: f"auto-{kw['trainer_role']}",
    )
    monkeypatch.setattr(
        infra_setup_mod, "get_deployment_gpu_count", lambda *a, **kw: 1,
    )
    monkeypatch.setattr(
        infra_setup_mod.transformers.AutoTokenizer,
        "from_pretrained", lambda *a, **kw: object(),
    )

    fake_weight_syncer = MagicMock()
    monkeypatch.setattr(
        infra_setup_mod, "WeightSyncer", lambda **kw: fake_weight_syncer,
    )
    fake_sampler = MagicMock()
    monkeypatch.setattr(
        infra_setup_mod, "DeploymentSampler", lambda **kw: fake_sampler,
    )

    return SimpleNamespace(trainer_calls=trainer_calls)


def _make_mgrs(*, profile=None):
    rlor = MagicMock()
    rlor.resolve_training_profile.return_value = profile or _profile()
    deploy = MagicMock()
    deploy.inference_url = "https://inf.unit"
    deploy.boot_time_s = 1.5
    return rlor, deploy


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_setup_infra_requires_deploy_mgr_when_inference_needed(patch_sdk):
    rlor, _ = _make_mgrs()
    cfg = _make_cfg()
    with pytest.raises(ValueError, match="deploy_mgr is required"):
        setup_infra(
            cfg, rlor_mgr=rlor, deploy_mgr=None,
            needs_reference=False, needs_inference=True,
            api_key="key",
        )


def test_setup_infra_requires_tokenizer_model_for_inference(patch_sdk):
    rlor, deploy = _make_mgrs()
    cfg = _make_cfg()
    cfg.deployment.tokenizer_model = ""
    with pytest.raises(ValueError, match="tokenizer_model"):
        setup_infra(
            cfg, rlor_mgr=rlor, deploy_mgr=deploy,
            needs_reference=False, needs_inference=True,
            api_key="key",
        )


# ---------------------------------------------------------------------------
# RL — full-param + KL: separate reference trainer
# ---------------------------------------------------------------------------


def test_setup_infra_rl_full_param_with_kl_provisions_separate_reference(patch_sdk):
    rlor, deploy = _make_mgrs()
    cfg = _make_cfg(lora_rank=0, ref_training_shape_id="shape-ref")

    infra = setup_infra(
        cfg, rlor_mgr=rlor, deploy_mgr=deploy,
        needs_reference=True, needs_inference=True,
        api_key="key",
    )

    display_names = [c["display_name"] for c in patch_sdk.trainer_calls]
    assert "grpo-policy" in display_names
    assert "grpo-reference" in display_names
    ref_call = next(c for c in patch_sdk.trainer_calls if c["display_name"] == "grpo-reference")
    assert ref_call["forward_only"] is True
    assert ref_call["lora_rank"] == 0  # ref trainers never carry LoRA

    assert infra.policy_job_id == "policy-job"
    assert infra.reference_job_id == "ref-job"
    assert infra.policy is not None
    assert infra.reference is not None
    assert isinstance(infra.reference, _FakeClient)
    # 2 ReconnectableClient constructions: policy + reference (separate)
    assert len(_FakeClient.instances) == 2


# ---------------------------------------------------------------------------
# RL — full-param + no KL: no reference at all
# ---------------------------------------------------------------------------


def test_setup_infra_rl_full_param_no_kl_skips_reference(patch_sdk):
    rlor, deploy = _make_mgrs()
    cfg = _make_cfg(lora_rank=0)

    infra = setup_infra(
        cfg, rlor_mgr=rlor, deploy_mgr=deploy,
        needs_reference=False, needs_inference=True,
        api_key="key",
    )

    display_names = [c["display_name"] for c in patch_sdk.trainer_calls]
    assert display_names == ["grpo-policy"]
    assert infra.reference is None
    assert infra.reference_job_id is None
    assert len(_FakeClient.instances) == 1  # policy only


# ---------------------------------------------------------------------------
# LoRA shared reference — applies to both RL (KL) and DPO
# ---------------------------------------------------------------------------


def test_setup_infra_lora_with_reference_uses_shared_session_rl(patch_sdk):
    """LoRA + needs_reference (RL with kl_beta>0): one trainer, shared session."""
    rlor, deploy = _make_mgrs()
    cfg = _make_cfg(lora_rank=64)

    infra = setup_infra(
        cfg, rlor_mgr=rlor, deploy_mgr=deploy,
        needs_reference=True, needs_inference=True,
        api_key="key",
    )

    # Only the policy trainer — no parallel reference trainer.
    display_names = [c["display_name"] for c in patch_sdk.trainer_calls]
    assert display_names == ["grpo-policy"]
    assert infra.reference_job_id is None
    # Reference is a shared base handle.
    assert infra.reference is not None
    assert getattr(infra.reference, "kind", None) == "base_shared"
    # policy.create_base_reference() was called once.
    policy_inst = next(c for c in _FakeClient.instances if c.job_id == "policy-job")
    assert policy_inst.base_ref_calls == 1


def test_setup_infra_lora_with_reference_uses_shared_session_dpo(patch_sdk):
    """LoRA + needs_reference + needs_inference=False (DPO): one trainer, shared."""
    rlor, _ = _make_mgrs()
    cfg = _make_cfg(lora_rank=64)

    infra = setup_infra(
        cfg, rlor_mgr=rlor, deploy_mgr=None,
        needs_reference=True, needs_inference=False,
        api_key="key",
    )

    display_names = [c["display_name"] for c in patch_sdk.trainer_calls]
    assert display_names == ["dpo-policy"]
    assert infra.reference_job_id is None
    assert infra.reference is not None and getattr(infra.reference, "kind", None) == "base_shared"
    # No sampler / weight_syncer when needs_inference=False.
    assert infra.sampler is None
    assert infra.weight_syncer is None


# ---------------------------------------------------------------------------
# DPO — full-param: separate reference trainer (parallel)
# ---------------------------------------------------------------------------


def test_setup_infra_dpo_full_param_provisions_separate_reference(patch_sdk):
    rlor, _ = _make_mgrs()
    cfg = _make_cfg(lora_rank=0, ref_training_shape_id="shape-ref")

    infra = setup_infra(
        cfg, rlor_mgr=rlor, deploy_mgr=None,
        needs_reference=True, needs_inference=False,
        api_key="key",
    )

    display_names = sorted(c["display_name"] for c in patch_sdk.trainer_calls)
    assert display_names == ["dpo-policy", "dpo-reference"]
    assert infra.reference_job_id == "ref-job"
    assert isinstance(infra.reference, _FakeClient)
    # Reference client created with lora_rank=0 (ref trainers don't carry LoRA).
    ref_inst = next(c for c in _FakeClient.instances if c.job_id == "ref-job")
    assert ref_inst.lora_rank == 0


# ---------------------------------------------------------------------------
# Auto-select shapes
# ---------------------------------------------------------------------------


def test_setup_infra_auto_selects_policy_shape_when_unset(patch_sdk):
    rlor, deploy = _make_mgrs()
    cfg = _make_cfg(training_shape_id=None)

    infra = setup_infra(
        cfg, rlor_mgr=rlor, deploy_mgr=deploy,
        needs_reference=False, needs_inference=True,
        api_key="key",
    )

    assert cfg.infra.training_shape_id == "auto-policy"
    assert infra.policy_profile is not None


def test_setup_infra_auto_selects_reference_shape_for_full_param_with_kl(patch_sdk):
    rlor, deploy = _make_mgrs()
    cfg = _make_cfg(lora_rank=0, ref_training_shape_id=None)

    setup_infra(
        cfg, rlor_mgr=rlor, deploy_mgr=deploy,
        needs_reference=True, needs_inference=True,
        api_key="key",
    )

    assert cfg.infra.ref_training_shape_id == "auto-reference"


def test_setup_infra_does_not_auto_select_ref_for_lora(patch_sdk):
    """LoRA path skips auto-select — uses shared session instead."""
    rlor, deploy = _make_mgrs()
    cfg = _make_cfg(lora_rank=64, ref_training_shape_id=None)

    setup_infra(
        cfg, rlor_mgr=rlor, deploy_mgr=deploy,
        needs_reference=True, needs_inference=True,
        api_key="key",
    )

    assert cfg.infra.ref_training_shape_id is None


# ---------------------------------------------------------------------------
# Closeables + boot metrics
# ---------------------------------------------------------------------------


def test_setup_infra_returns_closeables_for_caller_to_register(patch_sdk):
    rlor, deploy = _make_mgrs()
    cfg = _make_cfg(lora_rank=64)
    infra = setup_infra(
        cfg, rlor_mgr=rlor, deploy_mgr=deploy,
        needs_reference=True, needs_inference=True,
        api_key="key",
    )
    # Policy + base-shared reference both close-able and registered.
    assert len(infra.closeables) == 2


def test_setup_infra_records_boot_metrics(patch_sdk):
    rlor, deploy = _make_mgrs()
    cfg = _make_cfg()
    infra = setup_infra(
        cfg, rlor_mgr=rlor, deploy_mgr=deploy,
        needs_reference=False, needs_inference=True,
        api_key="key",
    )
    assert "infra/total_boot_time" in infra.boot_metrics
    assert infra.boot_metrics["infra/deploy_boot_time"] == 1.5
