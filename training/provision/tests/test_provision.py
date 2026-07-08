from __future__ import annotations

import importlib.util
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_PROVISION_PATH = Path(__file__).resolve().parents[1] / "provision.py"
_SPEC = importlib.util.spec_from_file_location("provision_script", _PROVISION_PATH)
assert _SPEC is not None and _SPEC.loader is not None
module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = module
_SPEC.loader.exec_module(module)


def _cfg(*, trainer_job_id: str | None = None, deployment_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        base_model="accounts/fireworks/models/qwen3-8b",
        lora_rank=16,
        max_seq_len=4096,
        learning_rate=1e-5,
        kl_beta=0.0,
        step_timeout=0,
        weight_sync_timeout=600,
        trainer=SimpleNamespace(job_id=trainer_job_id),
        deployment=SimpleNamespace(
            deployment_id=deployment_id,
            tokenizer_model="Qwen/Qwen3-8B",
            replica_count=2,
        ),
        concurrency=SimpleNamespace(
            initial_window=None,
            min_window=1,
            max_window=32,
            prefill_queue_target=0.5,
        ),
    )


class _FakeService:
    trainer_job_id = "trainer-1"
    deployment_id = "deployment-1"
    managed_deployment_id = "deployment-1"
    reference_client_job_id = "reference-client-1"
    reference_trainer_job_id = "reference-trainer-1"
    max_context_length = 4096
    accelerator_type = "NVIDIA_B200"
    accelerator_count = 8
    training_profile = object()

    def __init__(self) -> None:
        self.closed = False

    def create_training_client(self, _base_model: str, *, lora_rank: int) -> object:
        return object()

    def create_reference_client(self, _base_model: str, *, lora_rank: int) -> object:
        return object()

    def create_deployment_sampler(self, *, tokenizer: object, concurrency_controller: object) -> object:
        return object()

    def close(self) -> None:
        self.closed = True


def _stub_runtime(monkeypatch: pytest.MonkeyPatch, calls: list[dict]) -> _FakeService:
    service = _FakeService()

    def fake_build_service_client(**kwargs):
        calls.append(kwargs)
        return service

    monkeypatch.setattr(module, "build_service_client", fake_build_service_client)
    monkeypatch.setattr(module, "load_deployment_tokenizer", lambda _deployment: object())
    monkeypatch.setattr(
        module,
        "AdaptiveConcurrencyController",
        lambda **_kwargs: object(),
    )
    return service


def test_init_enables_cleanup_for_fresh_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    _stub_runtime(monkeypatch, calls)

    module.init_fireworks_infra(
        "rl",
        _cfg(),
        api_key="fw-key",
        base_url="https://api.fireworks.ai",
        cleanup_on_close=True,
    )

    assert calls[0]["cleanup_trainer_on_close"] is True
    assert calls[0]["cleanup_deployment_on_close"] == "scale_to_zero"


def test_init_does_not_cleanup_existing_resources_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    _stub_runtime(monkeypatch, calls)

    module.init_fireworks_infra(
        "rl",
        _cfg(trainer_job_id="existing-trainer", deployment_id="existing-deployment"),
        api_key="fw-key",
        base_url="https://api.fireworks.ai",
        cleanup_on_close=True,
    )

    assert calls[0]["cleanup_trainer_on_close"] is False
    assert calls[0]["cleanup_deployment_on_close"] is None


def test_init_can_cleanup_existing_resources_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    _stub_runtime(monkeypatch, calls)

    module.init_fireworks_infra(
        "rl",
        _cfg(trainer_job_id="existing-trainer", deployment_id="existing-deployment"),
        api_key="fw-key",
        base_url="https://api.fireworks.ai",
        cleanup_on_close=True,
        cleanup_existing=True,
    )

    assert calls[0]["cleanup_trainer_on_close"] is True
    assert calls[0]["cleanup_deployment_on_close"] == "scale_to_zero"


def test_progress_line_includes_resource_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module.time, "monotonic", lambda: 25.0)
    infra = module.FireworksProvisionInfra(
        mode="rl",
        service=object(),
        training_client=object(),
        policy=object(),
        reference=None,
        sampler=object(),
        policy_job_id="trainer-1",
        reference_job_id="reference-1",
        deployment_id="deployment-1",
        max_seq_len=4096,
        accelerator_type="NVIDIA_B200",
        accelerator_count=8,
        training_profile=object(),
    )

    line = module._format_progress(infra, started_at=10.0)

    assert "[   15s] Fireworks rl infra alive" in line
    assert "trainer=trainer-1" in line
    assert "deployment=deployment-1" in line
    assert "reference=reference-1" in line


def test_init_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        module.init_fireworks_infra("ppo", _cfg(), api_key="fw-key", base_url="https://api.fireworks.ai")


def test_sft_mode_uses_tokenizer_model_and_no_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    _stub_runtime(monkeypatch, calls)
    cfg = _cfg()
    cfg.tokenizer_model = "Qwen/Qwen3-8B"
    cfg.serverless = False

    module.init_fireworks_infra(
        "sft",
        cfg,
        api_key="fw-key",
        base_url="https://api.fireworks.ai",
        cleanup_on_close=True,
    )

    assert calls[0]["tokenizer_model"] == "Qwen/Qwen3-8B"
    assert calls[0]["deployment"] is None


def test_load_yaml_provision_resolves_named_rl_config(tmp_path: Path) -> None:
    config_path = tmp_path / "fireworks.yaml"
    config_path.write_text(
        """
common:
  base_model: accounts/test/models/base
  tokenizer_model: Qwen/Test
  lora_rank: 16
trainers:
  policy:
    training_shape_id: accounts/test/trainingShapes/policy
deployments:
  rollout:
    tokenizer_model: Qwen/Test
    replica_count: 2
recipe:
  rl_small:
    trainer: policy
    deployment: rollout
    kl_beta: 0.2
""",
        encoding="utf-8",
    )

    mode, cfg = module._load_yaml_provision(mode=None, recipe="rl_small", path=config_path)

    assert mode == "rl"
    assert cfg.base_model == "accounts/test/models/base"
    assert cfg.lora_rank == 16
    assert cfg.kl_beta == 0.2
    assert cfg.trainer.training_shape_id == "accounts/test/trainingShapes/policy"
    assert cfg.deployment.tokenizer_model == "Qwen/Test"
    assert cfg.deployment.replica_count == 2


def test_trainer_base_model_defaults_to_common_base_model(tmp_path: Path) -> None:
    config_path = tmp_path / "fireworks.yaml"
    config_path.write_text(
        """
common:
  base_model: accounts/test/models/common-base
  tokenizer_model: Qwen/Test
trainers:
  policy:
    training_shape_id: accounts/test/trainingShapes/policy
recipe:
  rl_small:
    trainer: policy
    deployment:
      tokenizer_model: Qwen/Test
""",
        encoding="utf-8",
    )

    _mode, cfg = module._load_yaml_provision(mode=None, recipe="rl_small", path=config_path)

    assert cfg.base_model == "accounts/test/models/common-base"


def test_trainer_base_model_can_override_common_base_model(tmp_path: Path) -> None:
    config_path = tmp_path / "fireworks.yaml"
    config_path.write_text(
        """
common:
  base_model: accounts/test/models/common-base
  tokenizer_model: Qwen/Test
trainers:
  policy:
    training_shape_id: accounts/test/trainingShapes/policy
    base_model: accounts/test/models/policy-base
recipe:
  rl_small:
    trainer: policy
    deployment:
      tokenizer_model: Qwen/Test
""",
        encoding="utf-8",
    )

    _mode, cfg = module._load_yaml_provision(mode=None, recipe="rl_small", path=config_path)

    assert cfg.base_model == "accounts/test/models/policy-base"


def test_load_yaml_provision_supports_weight_sync_deployment_on_trainer(tmp_path: Path) -> None:
    config_path = tmp_path / "fireworks.yaml"
    config_path.write_text(
        """
common:
  base_model: accounts/test/models/base
  tokenizer_model: Qwen/Test
trainers:
  policy:
    training_shape_id: accounts/test/trainingShapes/policy
    weight_sync_deployment: rollout
deployments:
  rollout:
    tokenizer_model: Qwen/Test
    replica_count: 2
recipe:
  rl_small:
    trainer: policy
""",
        encoding="utf-8",
    )

    _mode, cfg = module._load_yaml_provision(mode=None, recipe="rl_small", path=config_path)

    assert cfg.deployment.tokenizer_model == "Qwen/Test"
    assert cfg.deployment.replica_count == 2


def test_trainer_deployment_field_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "fireworks.yaml"
    config_path.write_text(
        """
common:
  base_model: accounts/test/models/base
  tokenizer_model: Qwen/Test
trainers:
  policy:
    training_shape_id: accounts/test/trainingShapes/policy
    deployment: rollout
recipe:
  rl_small:
    trainer: policy
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="weight_sync_deployment"):
        module._load_yaml_provision(mode=None, recipe="rl_small", path=config_path)


def test_load_yaml_provision_supports_reference_trainer(tmp_path: Path) -> None:
    config_path = tmp_path / "fireworks.yaml"
    config_path.write_text(
        """
common:
  base_model: accounts/test/models/base
  tokenizer_model: Qwen/Test
trainers:
  policy:
    training_shape_id: accounts/test/trainingShapes/policy
  reference:
    training_shape_id: accounts/test/trainingShapes/reference
recipe:
  rl_full:
    trainer: policy
    reference_trainer: reference
    deployment:
      tokenizer_model: Qwen/Test
""",
        encoding="utf-8",
    )

    _mode, cfg = module._load_yaml_provision(mode="rl", recipe="rl_full", path=config_path)

    assert cfg.trainer.training_shape_id == "accounts/test/trainingShapes/policy"
    assert cfg.trainer.reference_training_shape_id == "accounts/test/trainingShapes/reference"


def test_load_yaml_provision_passes_through_recipe_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "fireworks.yaml"
    config_path.write_text(
        """
common:
  base_model: accounts/test/models/base
  tokenizer_model: Qwen/Test
trainers:
  policy:
    training_shape_id: accounts/test/trainingShapes/policy
deployments:
  rollout:
    tokenizer_model: Qwen/Test
recipe:
  rl_custom:
    trainer: policy
    deployment: rollout
    completions_per_prompt: 8
    prompt_groups_per_step: 3
    max_rows: 12
    dapo:
      eps_clip_high: 0.35
""",
        encoding="utf-8",
    )

    _mode, cfg = module._load_yaml_provision(mode=None, recipe="rl_custom", path=config_path)

    assert cfg.completions_per_prompt == 8
    assert cfg.prompt_groups_per_step == 3
    assert cfg.max_rows == 12
    assert cfg.dapo.eps_clip_high == 0.35


def test_load_yaml_provision_hydrates_distillation_teacher_model(tmp_path: Path) -> None:
    config_path = tmp_path / "fireworks.yaml"
    config_path.write_text(
        """
common:
  base_model: accounts/test/models/student
  tokenizer_model: Qwen/Test
trainers:
  policy:
    training_shape_id: accounts/test/trainingShapes/policy
deployments:
  rollout:
    tokenizer_model: Qwen/Test
recipe:
  distillation_multi:
    mode: distillation
    trainer: policy
    deployment: rollout
    max_completion_tokens: 256
    teacher_model: accounts/test/models/math-teacher
""",
        encoding="utf-8",
    )

    mode, cfg = module._load_yaml_provision(mode=None, recipe="distillation_multi", path=config_path)

    assert mode == "distillation"
    assert cfg.max_completion_tokens == 256
    assert cfg.teacher_model == "accounts/test/models/math-teacher"


def test_load_yaml_provision_requires_distillation_teacher_model(tmp_path: Path) -> None:
    config_path = tmp_path / "fireworks.yaml"
    config_path.write_text(
        """
common:
  base_model: accounts/test/models/student
  tokenizer_model: Qwen/Test
trainers:
  policy:
    training_shape_id: accounts/test/trainingShapes/policy
deployments:
  rollout:
    tokenizer_model: Qwen/Test
recipe:
  distillation_no_teacher:
    mode: distillation
    trainer: policy
    deployment: rollout
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="teacher_model"):
        module._load_yaml_provision(mode=None, recipe="distillation_no_teacher", path=config_path)


def test_mode_can_be_inferred_from_recipe_name(tmp_path: Path) -> None:
    config_path = tmp_path / "fireworks.yaml"
    config_path.write_text(
        """
common:
  base_model: accounts/test/models/base
  tokenizer_model: Qwen/Test
trainers:
  policy:
    training_shape_id: accounts/test/trainingShapes/policy
recipe:
  sft:
    trainer: policy
""",
        encoding="utf-8",
    )

    mode, cfg = module._load_yaml_provision(mode=None, recipe="sft", path=config_path)

    assert mode == "sft"
    assert cfg.trainer.training_shape_id == "accounts/test/trainingShapes/policy"


def test_resolve_config_path_prefers_explicit_config() -> None:
    assert module._resolve_config_path("/tmp/custom.yaml", None) == Path("/tmp/custom.yaml")


def test_resolve_config_path_maps_config_name_to_sibling_file() -> None:
    resolved = module._resolve_config_path(None, "fireworks_sft")
    assert resolved == module.FIREWORKS_YAML.with_name("fireworks_sft.yaml")


def test_resolve_config_path_defaults_to_fireworks_yaml() -> None:
    assert module._resolve_config_path(None, None) == module.FIREWORKS_YAML


def test_resolve_config_path_rejects_both_flags() -> None:
    with pytest.raises(ValueError, match="either --config or --config-name"):
        module._resolve_config_path("/tmp/custom.yaml", "fireworks_sft")


def test_shipped_config_files_load() -> None:
    base = module.FIREWORKS_YAML.parent
    sft_mode, _ = module._load_yaml_provision(mode=None, recipe=None, path=base / "fireworks_sft.yaml")
    rft_mode, _ = module._load_yaml_provision(mode=None, recipe=None, path=base / "fireworks_rft.yaml")
    distill_mode, _ = module._load_yaml_provision(mode=None, recipe=None, path=base / "fireworks_distillation.yaml")
    dpo_mode, dpo_cfg = module._load_yaml_provision(mode=None, recipe=None, path=base / "fireworks_dpo.yaml")
    assert sft_mode == "sft"
    assert rft_mode == "rl"  # rft is an alias for rl
    assert distill_mode == "distillation"
    assert dpo_mode == "dpo"
    assert dpo_cfg.beta == 0.1


def test_rft_alias_resolves_to_rl() -> None:
    assert module._canonical_mode("rft") == "rl"
    assert module._mode_from_recipe_name("rft") == "rl"
    assert module._mode_from_recipe_name("rft_small") == "rl"


def test_dpo_mode_provisions_policy_and_reference_without_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    service = _stub_runtime(monkeypatch, calls)
    reference_calls: list[int] = []
    service.create_reference_client = lambda _base_model, *, lora_rank: reference_calls.append(lora_rank) or object()  # type: ignore[assignment]

    cfg = _cfg()
    cfg.tokenizer_model = "Qwen/Qwen3-8B"

    infra = module.init_fireworks_infra(
        "dpo",
        cfg,
        api_key="fw-key",
        base_url="https://api.fireworks.ai",
    )

    assert infra.mode == "dpo"
    assert calls[0]["deployment"] is None
    assert calls[0]["reference_required"] is True
    assert reference_calls  # a reference client was created


class _FakeTrainerManager:
    def __init__(self, states: dict[str, str]) -> None:
        self.states = states

    def get(self, job_id: str) -> dict:
        return {"state": self.states[job_id]}


class _FakeDeploymentManager:
    def __init__(self, states: dict[str, str]) -> None:
        self.states = states

    def get(self, deployment_id: str) -> SimpleNamespace | None:
        state = self.states.get(deployment_id)
        if state is None:
            return None
        return SimpleNamespace(state=state)


class _DeletingTrainerManager:
    def __init__(self) -> None:
        self.calls = 0

    def try_get(self, job_id: str) -> dict | None:
        self.calls += 1
        if self.calls == 1:
            return {"state": "JOB_STATE_DELETING"}
        return None


class _DeletingDeploymentManager:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, deployment_id: str) -> SimpleNamespace | None:
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(state="DELETING")
        return None


def test_status_check_reports_all_healthy_resources() -> None:
    infra = module.FireworksProvisionInfra(
        mode="distillation",
        service=object(),
        training_client=object(),
        policy=object(),
        policy_job_id="trainer-1",
        reference_job_id="reference-1",
        deployment_id="student-dep",
        teacher_job_ids=["teacher-1"],
        max_seq_len=4096,
        accelerator_type="NVIDIA_B200",
        accelerator_count=8,
        training_profile=object(),
        trainer_manager=_FakeTrainerManager({
            "trainer-1": "JOB_STATE_RUNNING",
            "reference-1": "JOB_STATE_READY",
            "teacher-1": "JOB_STATE_RUNNING",
        }),
        deployment_manager=_FakeDeploymentManager({
            "student-dep": "READY",
        }),
    )

    assert infra.unhealthy_statuses() == []


def test_status_check_reports_unhealthy_resources() -> None:
    infra = module.FireworksProvisionInfra(
        mode="rl",
        service=object(),
        training_client=object(),
        policy=object(),
        policy_job_id="trainer-1",
        deployment_id="deployment-1",
        max_seq_len=4096,
        accelerator_type="NVIDIA_B200",
        accelerator_count=8,
        training_profile=object(),
        trainer_manager=_FakeTrainerManager({"trainer-1": "JOB_STATE_FAILED"}),
        deployment_manager=_FakeDeploymentManager({"deployment-1": "FAILED"}),
    )

    unhealthy = infra.unhealthy_statuses()

    assert [(status.kind, status.resource_id, status.state) for status in unhealthy] == [
        ("trainer", "trainer-1", "JOB_STATE_FAILED"),
        ("deployment", "deployment-1", "FAILED"),
    ]
    report = module._format_unhealthy_report(unhealthy)
    assert "trainer trainer-1: state=JOB_STATE_FAILED" in report
    assert "deployment deployment-1: state=FAILED" in report
    assert "Continuing to monitor remaining resources." in report


def test_run_until_interrupted_keeps_monitoring_after_unhealthy_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    infra = module.FireworksProvisionInfra(
        mode="distillation",
        service=object(),
        training_client=object(),
        policy=object(),
        policy_job_id="policy-1",
        deployment_id="deployment-1",
        teacher_job_ids=["teacher-1"],
        max_seq_len=4096,
        accelerator_type="NVIDIA_B200",
        accelerator_count=8,
        training_profile=object(),
        trainer_manager=_FakeTrainerManager({
            "policy-1": "JOB_STATE_RUNNING",
            "teacher-1": "JOB_STATE_COMPLETED",
        }),
        deployment_manager=_FakeDeploymentManager({
            "deployment-1": "READY",
        }),
    )
    slept = False
    cleaned_up = False

    def fake_sleep(_seconds: float) -> None:
        nonlocal slept
        slept = True
        raise KeyboardInterrupt

    def fake_cleanup(
        _infra: object,
        *,
        started_at: float,
        progress_interval_s: float,
    ) -> None:
        nonlocal cleaned_up
        cleaned_up = True

    monkeypatch.setattr(module, "init_fireworks_infra", lambda *_args, **_kwargs: infra)
    monkeypatch.setattr(module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(module.time, "sleep", fake_sleep)
    monkeypatch.setattr(module, "_cleanup_until_complete", fake_cleanup)

    result = module.run_until_interrupted("distillation", object(), progress_interval_s=1.0)

    assert result is infra
    assert slept is True
    assert cleaned_up is True
    output = capsys.readouterr().out
    assert "trainer teacher-1: state=JOB_STATE_COMPLETED" in output
    assert "Continuing to monitor remaining resources." in output
    assert "Exiting provision monitor." not in output


def test_cleanup_until_complete_keeps_polling_until_resources_are_gone(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = _FakeService()
    infra = module.FireworksProvisionInfra(
        mode="rl",
        service=service,
        training_client=object(),
        policy=object(),
        policy_job_id="trainer-1",
        deployment_id="deployment-1",
        max_seq_len=4096,
        accelerator_type="NVIDIA_B200",
        accelerator_count=8,
        training_profile=object(),
        trainer_manager=_DeletingTrainerManager(),
        deployment_manager=_DeletingDeploymentManager(),
    )
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(module.time, "monotonic", lambda: 20.0)

    module._cleanup_until_complete(infra, started_at=10.0, progress_interval_s=0.1)

    assert service.closed is True
    output = capsys.readouterr().out
    assert "trainer trainer-1: waiting, state=JOB_STATE_DELETING" in output
    assert "deployment deployment-1: waiting, state=DELETING" in output
    assert "trainer trainer-1: done, state=NOT_FOUND" in output
    assert "deployment deployment-1: done, state=NOT_FOUND" in output


def test_cleanup_treats_archived_trainer_as_done() -> None:
    # A deleted RLOR trainer is surfaced on the public API as JOB_STATE_ARCHIVED
    # (retention tombstone). Cleanup must treat that as fully torn down so the
    # monitor exits instead of polling forever.
    manager = SimpleNamespace(try_get=lambda _job_id: {"state": "JOB_STATE_ARCHIVED"})

    status = module._check_trainer_cleanup_status(manager, "trainer-1")

    assert status.state == "JOB_STATE_ARCHIVED"
    assert status.healthy is True


def test_cleanup_signal_handler_shields_after_first_interrupt(capsys: pytest.CaptureFixture[str]) -> None:
    _begin_cleanup, restore_signals = module._install_cleanup_signal_handlers()
    handler = signal.getsignal(signal.SIGINT)
    assert callable(handler)
    try:
        # First Ctrl+C raises once to unwind into the cleanup path.
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)

        # Every subsequent Ctrl+C is swallowed without re-raising, no matter how
        # many times it is pressed, so teardown is never interrupted.
        for _ in range(5):
            handler(signal.SIGINT, None)
    finally:
        restore_signals()

    assert "cleanup is already in progress and will not be interrupted" in capsys.readouterr().out
