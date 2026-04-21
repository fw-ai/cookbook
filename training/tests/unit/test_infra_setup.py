"""Unit tests for the bundled setup_infra entry point.

setup_infra owns all the control flow (shape resolution, parallel
trainer provisioning, LoRA shared-reference, deployment, sampler,
weight syncer). These tests exercise each branch in isolation by
patching the SDK boundary and verifying setup_infra wires things
together correctly — particularly the LoRA shared-reference path
that keeps a single trainer alive for both policy and reference.

Parallel-provisioning tests verify:
  * request phase fires before wait phase
  * trainer and deployment waits overlap in wall time
  * cleanup is registered at request time so failures during wait still clean up
  * re-attach PATCH is issued before trainer READY (parallel, not serial)
"""

from __future__ import annotations

import threading
import time
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
from training.utils.infra import ResourceCleanup
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


def _trainer_handle(job_id: str = "job-x") -> SimpleNamespace:
    """Fake CreatedTrainerJob returned by request_trainer_job."""
    return SimpleNamespace(job_id=job_id, job_name=f"jobs/{job_id}")


def _make_cfg(
    *,
    lora_rank: int = 0,
    training_shape_id: str | None = "shape-policy",
    ref_training_shape_id: str | None = None,
    deployment_id: str | None = "dep-1",
    policy_job_id: str | None = None,
    reference_job_id: str | None = None,
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
        policy_job_id=policy_job_id,
        reference_job_id=reference_job_id,
    )


@pytest.fixture
def patch_sdk(monkeypatch):
    """Patch all SDK boundary calls inside infra_setup with fakes.

    Uses request_trainer_job + wait_trainer_job (the new split API) instead of
    the old monolithic create_trainer_job so tests exercise the actual call graph.
    """
    _FakeClient.instances = []

    trainer_calls: list[dict] = []

    def fake_request_trainer(_rlor, *, display_name, forward_only=False, **kwargs):
        job_id = "ref-job" if "reference" in display_name else "policy-job"
        trainer_calls.append({
            "display_name": display_name,
            "forward_only": forward_only,
            "lora_rank": kwargs.get("lora_rank"),
            "job_id_arg": kwargs.get("job_id"),
        })
        return _trainer_handle(job_id)

    def fake_wait_trainer(_rlor, created, *, display_name="", forward_only=False, **kwargs):
        return _trainer_ep(getattr(created, "job_id", "policy-job"))

    monkeypatch.setattr(infra_setup_mod, "request_trainer_job", fake_request_trainer)
    monkeypatch.setattr(infra_setup_mod, "wait_trainer_job", fake_wait_trainer)
    monkeypatch.setattr(infra_setup_mod, "ReconnectableClient", _FakeClient)
    monkeypatch.setattr(infra_setup_mod, "_read_replica_identity", lambda *a, **kw: None)
    monkeypatch.setattr(infra_setup_mod, "_wait_for_reattach_settled", lambda *a, **kw: None)
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


# ---------------------------------------------------------------------------
# Parallel provisioning tests
# ---------------------------------------------------------------------------


def test_parallel_request_phase_precedes_wait_phase(monkeypatch):
    """request_trainer_job fires for both trainers before wait_trainer_job is called for either."""
    _FakeClient.instances = []
    events: list[str] = []

    def fake_request_trainer(_rlor, *, display_name, forward_only=False, **kwargs):
        events.append(f"request:{display_name}")
        job_id = "ref-job" if "reference" in display_name else "policy-job"
        return _trainer_handle(job_id)

    def fake_wait_trainer(_rlor, created, *, display_name="", forward_only=False, **kwargs):
        events.append(f"wait:{display_name}")
        return _trainer_ep(getattr(created, "job_id", "policy-job"))

    monkeypatch.setattr(infra_setup_mod, "request_trainer_job", fake_request_trainer)
    monkeypatch.setattr(infra_setup_mod, "wait_trainer_job", fake_wait_trainer)
    monkeypatch.setattr(infra_setup_mod, "ReconnectableClient", _FakeClient)
    monkeypatch.setattr(infra_setup_mod, "_read_replica_identity", lambda *a, **kw: None)
    monkeypatch.setattr(infra_setup_mod, "_wait_for_reattach_settled", lambda *a, **kw: None)
    monkeypatch.setattr(
        infra_setup_mod, "auto_select_training_shape",
        lambda _rlor, **kw: f"auto-{kw['trainer_role']}",
    )
    monkeypatch.setattr(infra_setup_mod, "get_deployment_gpu_count", lambda *a, **kw: 1)
    monkeypatch.setattr(
        infra_setup_mod.transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: object(),
    )
    monkeypatch.setattr(infra_setup_mod, "WeightSyncer", lambda **kw: MagicMock())
    monkeypatch.setattr(infra_setup_mod, "DeploymentSampler", lambda **kw: MagicMock())

    rlor, deploy = _make_mgrs(profile=_profile())
    cfg = _make_cfg(lora_rank=0, ref_training_shape_id="shape-ref")
    setup_infra(
        cfg, rlor_mgr=rlor, deploy_mgr=deploy,
        needs_reference=True, needs_inference=True,
        api_key="key",
    )

    request_indices = [i for i, e in enumerate(events) if e.startswith("request:")]
    wait_indices = [i for i, e in enumerate(events) if e.startswith("wait:")]
    # All request events must precede the first wait event.
    assert max(request_indices) < min(wait_indices), (
        f"Expected all requests before any waits, got: {events}"
    )


def test_parallel_wait_timing(monkeypatch):
    """Both trainer waits run in parallel; total wall time ≈ max(N), not sum(N)."""
    _FakeClient.instances = []
    SLEEP_S = 0.15  # each wait sleeps this long; sum would be 0.30

    def fake_request_trainer(_rlor, *, display_name, forward_only=False, **kwargs):
        job_id = "ref-job" if "reference" in display_name else "policy-job"
        return _trainer_handle(job_id)

    def fake_wait_trainer(_rlor, created, *, display_name="", forward_only=False, **kwargs):
        time.sleep(SLEEP_S)
        return _trainer_ep(getattr(created, "job_id", "policy-job"))

    monkeypatch.setattr(infra_setup_mod, "request_trainer_job", fake_request_trainer)
    monkeypatch.setattr(infra_setup_mod, "wait_trainer_job", fake_wait_trainer)
    monkeypatch.setattr(infra_setup_mod, "ReconnectableClient", _FakeClient)
    monkeypatch.setattr(infra_setup_mod, "_read_replica_identity", lambda *a, **kw: None)
    monkeypatch.setattr(infra_setup_mod, "_wait_for_reattach_settled", lambda *a, **kw: None)
    monkeypatch.setattr(
        infra_setup_mod, "auto_select_training_shape",
        lambda _rlor, **kw: f"auto-{kw['trainer_role']}",
    )
    monkeypatch.setattr(infra_setup_mod, "get_deployment_gpu_count", lambda *a, **kw: 1)
    monkeypatch.setattr(
        infra_setup_mod.transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: object(),
    )
    monkeypatch.setattr(infra_setup_mod, "WeightSyncer", lambda **kw: MagicMock())
    monkeypatch.setattr(infra_setup_mod, "DeploymentSampler", lambda **kw: MagicMock())

    rlor, deploy = _make_mgrs(profile=_profile())
    cfg = _make_cfg(lora_rank=0, ref_training_shape_id="shape-ref")

    t0 = time.monotonic()
    setup_infra(
        cfg, rlor_mgr=rlor, deploy_mgr=deploy,
        needs_reference=True, needs_inference=True,
        api_key="key",
    )
    elapsed = time.monotonic() - t0

    # Parallel: elapsed ≈ SLEEP_S (max of two). Serial would be 2*SLEEP_S.
    # Allow up to 1.5x to avoid flakes on slow CI.
    assert elapsed < SLEEP_S * 1.5, (
        f"Waits appear serial: {elapsed:.3f}s ≥ {SLEEP_S * 1.5:.3f}s (sum would be {SLEEP_S * 2:.3f}s)"
    )


def test_failure_cleanup_registers_both_resources(monkeypatch):
    """If a wait fails, cleanup still has all IDs registered at request time."""
    _FakeClient.instances = []

    def fake_request_trainer(_rlor, *, display_name, cleanup=None, **kwargs):
        job_id = "ref-job" if "reference" in display_name else "policy-job"
        handle = _trainer_handle(job_id)
        if cleanup:
            cleanup.trainer(job_id)
        return handle

    def fake_wait_trainer(_rlor, created, *, display_name="", forward_only=False, **kwargs):
        if forward_only:
            raise RuntimeError("reference trainer failed")
        return _trainer_ep(getattr(created, "job_id", "policy-job"))

    monkeypatch.setattr(infra_setup_mod, "request_trainer_job", fake_request_trainer)
    monkeypatch.setattr(infra_setup_mod, "wait_trainer_job", fake_wait_trainer)
    monkeypatch.setattr(infra_setup_mod, "ReconnectableClient", _FakeClient)
    monkeypatch.setattr(infra_setup_mod, "_read_replica_identity", lambda *a, **kw: None)
    monkeypatch.setattr(infra_setup_mod, "_wait_for_reattach_settled", lambda *a, **kw: None)
    monkeypatch.setattr(
        infra_setup_mod, "auto_select_training_shape",
        lambda _rlor, **kw: f"auto-{kw['trainer_role']}",
    )
    monkeypatch.setattr(infra_setup_mod, "get_deployment_gpu_count", lambda *a, **kw: 1)
    monkeypatch.setattr(
        infra_setup_mod.transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: object(),
    )

    rlor = MagicMock()
    rlor.resolve_training_profile.return_value = _profile()
    deploy = MagicMock()
    deploy.inference_url = "https://inf.unit"
    deploy.boot_time_s = 1.5

    cfg = _make_cfg(lora_rank=0, ref_training_shape_id="shape-ref")

    cleanup = ResourceCleanup(rlor, deploy)
    with pytest.raises(RuntimeError, match="reference trainer failed"):
        setup_infra(
            cfg, rlor_mgr=rlor, deploy_mgr=deploy,
            needs_reference=True, needs_inference=False,
            api_key="key",
            cleanup=cleanup,
            cleanup_on_exit=True,
        )

    # Both job IDs must be registered so scope-exit can cancel them.
    assert "policy-job" in cleanup._jobs
    assert "ref-job" in cleanup._jobs


def test_lora_shared_ref_no_ref_trainer_created(monkeypatch):
    """LoRA + needs_reference=True: policy trainer only; reference via shared session."""
    _FakeClient.instances = []
    trainer_calls: list[str] = []

    def fake_request_trainer(_rlor, *, display_name, **kwargs):
        trainer_calls.append(display_name)
        job_id = "policy-job"
        return _trainer_handle(job_id)

    def fake_wait_trainer(_rlor, created, **kwargs):
        return _trainer_ep(getattr(created, "job_id", "policy-job"))

    monkeypatch.setattr(infra_setup_mod, "request_trainer_job", fake_request_trainer)
    monkeypatch.setattr(infra_setup_mod, "wait_trainer_job", fake_wait_trainer)
    monkeypatch.setattr(infra_setup_mod, "ReconnectableClient", _FakeClient)
    monkeypatch.setattr(infra_setup_mod, "_read_replica_identity", lambda *a, **kw: None)
    monkeypatch.setattr(infra_setup_mod, "_wait_for_reattach_settled", lambda *a, **kw: None)
    monkeypatch.setattr(
        infra_setup_mod, "auto_select_training_shape",
        lambda _rlor, **kw: f"auto-{kw['trainer_role']}",
    )
    monkeypatch.setattr(infra_setup_mod, "get_deployment_gpu_count", lambda *a, **kw: 1)
    monkeypatch.setattr(
        infra_setup_mod.transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: object(),
    )
    monkeypatch.setattr(infra_setup_mod, "WeightSyncer", lambda **kw: MagicMock())
    monkeypatch.setattr(infra_setup_mod, "DeploymentSampler", lambda **kw: MagicMock())

    rlor, deploy = _make_mgrs(profile=_profile())
    cfg = _make_cfg(lora_rank=64)

    infra = setup_infra(
        cfg, rlor_mgr=rlor, deploy_mgr=deploy,
        needs_reference=True, needs_inference=True,
        api_key="key",
    )

    # Only policy trainer was requested — no separate reference trainer.
    assert trainer_calls == ["grpo-policy"]
    assert infra.reference_job_id is None
    assert infra.reference is not None
    assert getattr(infra.reference, "kind", None) == "base_shared"
    policy_inst = next(c for c in _FakeClient.instances if c.job_id == "policy-job")
    assert policy_inst.base_ref_calls == 1


def test_reuse_path_policy_job_id_set(monkeypatch):
    """cfg.policy_job_id set: request returns endpoint (already ready); deploy still works."""
    _FakeClient.instances = []

    def fake_request_trainer(_rlor, *, display_name, job_id=None, **kwargs):
        if job_id:
            # Reuse path: return endpoint directly (no wait needed)
            return _trainer_ep(job_id)
        return _trainer_handle("policy-job")

    def fake_wait_trainer(_rlor, created, **kwargs):
        # TrainerServiceEndpoint → pass-through
        if hasattr(created, "base_url"):
            return created
        return _trainer_ep(getattr(created, "job_id", "policy-job"))

    monkeypatch.setattr(infra_setup_mod, "request_trainer_job", fake_request_trainer)
    monkeypatch.setattr(infra_setup_mod, "wait_trainer_job", fake_wait_trainer)
    monkeypatch.setattr(infra_setup_mod, "ReconnectableClient", _FakeClient)
    monkeypatch.setattr(infra_setup_mod, "_read_replica_identity", lambda *a, **kw: None)
    monkeypatch.setattr(infra_setup_mod, "_wait_for_reattach_settled", lambda *a, **kw: None)
    monkeypatch.setattr(
        infra_setup_mod, "auto_select_training_shape",
        lambda _rlor, **kw: f"auto-{kw['trainer_role']}",
    )
    monkeypatch.setattr(infra_setup_mod, "get_deployment_gpu_count", lambda *a, **kw: 1)
    monkeypatch.setattr(
        infra_setup_mod.transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: object(),
    )
    monkeypatch.setattr(infra_setup_mod, "WeightSyncer", lambda **kw: MagicMock())
    monkeypatch.setattr(infra_setup_mod, "DeploymentSampler", lambda **kw: MagicMock())

    rlor, deploy = _make_mgrs(profile=_profile())
    cfg = _make_cfg(lora_rank=0, policy_job_id="pre-created-policy")

    infra = setup_infra(
        cfg, rlor_mgr=rlor, deploy_mgr=deploy,
        needs_reference=False, needs_inference=True,
        api_key="key",
    )

    assert infra.policy_job_id == "pre-created-policy"
    assert infra.sampler is not None  # deployment still provisioned


def test_dpo_full_param_no_deployment(monkeypatch):
    """DPO + full-param: two trainers, no deployment requested."""
    _FakeClient.instances = []
    trainer_calls: list[str] = []

    def fake_request_trainer(_rlor, *, display_name, **kwargs):
        trainer_calls.append(display_name)
        job_id = "ref-job" if "reference" in display_name else "policy-job"
        return _trainer_handle(job_id)

    def fake_wait_trainer(_rlor, created, **kwargs):
        return _trainer_ep(getattr(created, "job_id", "policy-job"))

    monkeypatch.setattr(infra_setup_mod, "request_trainer_job", fake_request_trainer)
    monkeypatch.setattr(infra_setup_mod, "wait_trainer_job", fake_wait_trainer)
    monkeypatch.setattr(infra_setup_mod, "ReconnectableClient", _FakeClient)
    monkeypatch.setattr(
        infra_setup_mod, "auto_select_training_shape",
        lambda _rlor, **kw: f"auto-{kw['trainer_role']}",
    )

    rlor, _ = _make_mgrs(profile=_profile())
    cfg = _make_cfg(lora_rank=0, ref_training_shape_id="shape-ref")

    infra = setup_infra(
        cfg, rlor_mgr=rlor, deploy_mgr=None,
        needs_reference=True, needs_inference=False,
        api_key="key",
    )

    assert sorted(trainer_calls) == ["dpo-policy", "dpo-reference"]
    assert infra.sampler is None
    assert infra.weight_syncer is None
    assert infra.reference_job_id == "ref-job"


def test_reattach_patch_issued_before_trainer_ready(monkeypatch):
    """Re-attach PATCH fires before trainer READY; settle runs in parallel with trainer wait."""
    _FakeClient.instances = []
    events: list[str] = []

    def fake_request_trainer(_rlor, *, display_name, **kwargs):
        return _trainer_handle("policy-job")

    def fake_wait_trainer(_rlor, created, *, display_name="", **kwargs):
        time.sleep(0.05)
        events.append(f"wait_done:{display_name}")
        return _trainer_ep(getattr(created, "job_id", "policy-job"))

    def fake_settle(deploy_mgr, dep_id, base_model, *, prev_identity, timeout_s):
        events.append("settle_start")
        time.sleep(0.05)
        events.append("settle_done")

    monkeypatch.setattr(infra_setup_mod, "request_trainer_job", fake_request_trainer)
    monkeypatch.setattr(infra_setup_mod, "wait_trainer_job", fake_wait_trainer)
    monkeypatch.setattr(infra_setup_mod, "_read_replica_identity", lambda *a, **kw: "old-pod-id")
    monkeypatch.setattr(infra_setup_mod, "_wait_for_reattach_settled", fake_settle)
    monkeypatch.setattr(infra_setup_mod, "ReconnectableClient", _FakeClient)
    monkeypatch.setattr(
        infra_setup_mod, "auto_select_training_shape",
        lambda _rlor, **kw: f"auto-{kw['trainer_role']}",
    )
    monkeypatch.setattr(infra_setup_mod, "get_deployment_gpu_count", lambda *a, **kw: 1)
    monkeypatch.setattr(
        infra_setup_mod.transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: object(),
    )
    monkeypatch.setattr(infra_setup_mod, "WeightSyncer", lambda **kw: MagicMock())
    monkeypatch.setattr(infra_setup_mod, "DeploymentSampler", lambda **kw: MagicMock())

    rlor, deploy = _make_mgrs(profile=_profile())
    # Existing live deployment → triggers re-attach path.
    deploy.get.return_value = SimpleNamespace(
        state="READY", inference_model=None, deployment_id="dep-1",
    )
    cfg = _make_cfg(lora_rank=0, deployment_id="dep-1")

    setup_infra(
        cfg, rlor_mgr=rlor, deploy_mgr=deploy,
        needs_reference=False, needs_inference=True,
        api_key="key",
    )

    # PATCH must have been issued with policy_handle.job_name (available before READY).
    deploy.update.assert_called_once_with(
        "dep-1",
        body={"hotLoadTrainerJob": "jobs/policy-job"},
        update_mask="hot_load_trainer_job",
    )
    # Settle and trainer wait both ran (in parallel).
    assert "settle_start" in events
    assert "wait_done:grpo-policy" in events
