"""Unit tests for managed artifacts in training.recipes.async_rl_loop."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.recipes import async_rl_loop
from training.utils import DeployConfig, RunnerConfig


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class _FakeManager:
    inference_url = "https://inference.unit.test"

    def __init__(self, **_kwargs):
        pass


class _FakePolicy:
    def forward(self, data, _loss_name):
        return SimpleNamespace(
            loss_fn_outputs=[{"logprobs": SimpleNamespace(data=[-0.1])} for _ in data],
        )

    def forward_backward_custom(self, *_args, **_kwargs):
        return SimpleNamespace(metrics={"loss": 0.25})

    def optim_step(self, *_args, **_kwargs):
        return SimpleNamespace(metrics={"optim": 1.0})


class _FakeWeightSyncer:
    def __init__(self, **_kwargs):
        pass

    def save_and_hotload(self, *_args, **_kwargs):
        pass


class _FakeCheckpoints:
    def __init__(self, *_args, **_kwargs):
        pass

    def resume(self, **_kwargs):
        return None

    def save(self, *_args, **_kwargs):
        pass

    def promote_latest(self, *_args, **_kwargs):
        pass


class _FakeCleanup:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _patch_common_async_loop_deps(monkeypatch, events: list[str]) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "unit-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://api.unit.test")
    monkeypatch.setattr(async_rl_loop, "validate_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(async_rl_loop, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(async_rl_loop, "wandb_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(async_rl_loop, "wandb_finish", lambda: events.append("wandb_finish"))
    monkeypatch.setattr(async_rl_loop, "read_api_extra_headers_env", lambda: {})
    monkeypatch.setattr(async_rl_loop, "TrainerJobManager", _FakeManager)
    monkeypatch.setattr(async_rl_loop, "DeploymentManager", _FakeManager)
    monkeypatch.setattr(async_rl_loop, "ResourceCleanup", _FakeCleanup)
    monkeypatch.setattr(async_rl_loop, "WeightSyncer", _FakeWeightSyncer)
    monkeypatch.setattr(async_rl_loop, "TrainingCheckpoints", _FakeCheckpoints)
    monkeypatch.setattr(async_rl_loop, "load_deployment_tokenizer", lambda _cfg: object())
    monkeypatch.setattr(async_rl_loop, "build_loss_fn", lambda _args: lambda *args: object())
    monkeypatch.setattr(async_rl_loop, "combine_prompt_groups", lambda _groups: ([object()], [], [], [], []))
    monkeypatch.setattr(async_rl_loop, "compute_minibatch_metrics", lambda *_args: {"train/loss": 0.25})
    monkeypatch.setattr(
        async_rl_loop,
        "compute_step_metrics",
        lambda **_kwargs: {
            "train/loss": 0.25,
            "rollout/reward": 1.0,
            "rollout/accuracy": 1.0,
        },
    )
    monkeypatch.setattr(async_rl_loop, "flush_timing", lambda: {})
    monkeypatch.setattr(async_rl_loop, "total_target_tokens", lambda _groups: 123)
    monkeypatch.setattr(async_rl_loop.tinker, "AdamParams", lambda **kwargs: kwargs)


def _config(tmp_path: Path) -> async_rl_loop.Config:
    return async_rl_loop.Config(
        log_path=str(tmp_path / "logs"),
        completions_per_prompt=2,
        prompt_groups_per_step=1,
        kl_beta=0.0,
        save_final_checkpoint=False,
        deployment=DeployConfig(tokenizer_model="unit-tokenizer"),
        runner=RunnerConfig(
            status_file=str(tmp_path / "status.json"),
            metadata_file=str(tmp_path / "metadata.json"),
            metrics_file=str(tmp_path / "metrics.jsonl"),
        ),
    )


def test_managed_artifacts_written_during_async_run(monkeypatch, tmp_path: Path) -> None:
    """Status, metadata, metrics, and resources callback fire before completion."""
    events: list[str] = []
    _patch_common_async_loop_deps(monkeypatch, events)

    def fake_setup_infra(**_kwargs):
        events.append("setup_infra_done")
        return SimpleNamespace(
            policy=_FakePolicy(),
            reference=None,
            policy_profile=SimpleNamespace(accelerator_type="NVIDIA_H100_80GB", accelerator_count=8),
            policy_job_id="policy-job-1",
            reference_job_id=None,
            inference_model="accounts/test/models/hot",
            boot_metrics={},
            closeables=[],
            max_seq_len=4096,
            training_shape_id="training-shape",
            deployment_id="deployment-1",
        )

    async def fake_run_async_rl_loop(**kwargs):
        events.append("training_started")
        step, _metrics = kwargs["train_fns"].train_step(
            kwargs["global_step"],
            [object()],
            {"resolved_rows": 1},
        )
        return step, {"resolved_rows": 1}

    callback_payloads: list[dict] = []

    def on_resources_ready(**resources):
        events.append("resource_callback")
        callback_payloads.append(resources)

    monkeypatch.setattr(async_rl_loop, "setup_infra", fake_setup_infra)
    monkeypatch.setattr(async_rl_loop, "run_async_rl_loop", fake_run_async_rl_loop)

    async_rl_loop.main(
        _config(tmp_path),
        rollout_fn_factory=lambda _setup: lambda _row: None,
        rows=[{"id": "row-1"}],
        on_resources_ready=on_resources_ready,
    )

    assert events.index("setup_infra_done") < events.index("resource_callback")
    assert events.index("resource_callback") < events.index("training_started")
    assert callback_payloads == [
        {
            "deployment_id": "deployment-1",
            "policy_job_id": "policy-job-1",
            "reference_job_id": None,
        }
    ]
    assert _read_json(tmp_path / "status.json")["message"] == "done"
    assert _read_json(tmp_path / "status.json")["details"][0]["percent"] == 100
    assert _read_jsonl(tmp_path / "metrics.jsonl") == [
        {
            "step": 1,
            "rollout/reward": 1.0,
            "rollout/accuracy": 1.0,
            "rollout/step": 1,
            "train/step": 1,
            "ctx/completions_per_prompt": 2,
            "ctx/prompt_groups_per_step": 1,
            "ctx/max_head_offpolicy_versions": 0,
            "ctx/ppo_n_minibatches": 1,
            "ctx/weight_sync_interval": 1,
            "ctx/max_completion_tokens": 1024,
            "ctx/temperature": 1.0,
            "ctx/shuffle": 1,
            "ctx/seed": 0,
            "ctx/current_version": 0,
        }
    ]
    assert _read_json(tmp_path / "metadata.json")["metadata"]["tokens"] == 123


def test_runner_writes_failed_status_when_async_run_fails(monkeypatch, tmp_path: Path) -> None:
    """RunnerIO context writes a terminal failure status on async loop errors."""
    events: list[str] = []
    _patch_common_async_loop_deps(monkeypatch, events)

    monkeypatch.setattr(
        async_rl_loop,
        "setup_infra",
        lambda **_kwargs: SimpleNamespace(
            policy=_FakePolicy(),
            reference=None,
            policy_profile=SimpleNamespace(accelerator_type="NVIDIA_H100_80GB", accelerator_count=8),
            policy_job_id="policy-job-1",
            reference_job_id=None,
            inference_model="accounts/test/models/hot",
            boot_metrics={},
            closeables=[],
            max_seq_len=4096,
            training_shape_id="training-shape",
            deployment_id="deployment-1",
        ),
    )

    async def failing_async_loop(**_kwargs):
        raise RuntimeError("async failure")

    monkeypatch.setattr(async_rl_loop, "run_async_rl_loop", failing_async_loop)

    with pytest.raises(RuntimeError, match="async failure"):
        async_rl_loop.main(
            _config(tmp_path),
            rollout_fn_factory=lambda _setup: lambda _row: None,
            rows=[{"id": "row-1"}],
        )

    status = _read_json(tmp_path / "status.json")
    assert status["code"] == 9
    assert status["message"] == "async failure"


def test_resources_ready_callback_errors_are_logged_and_swallowed(caplog) -> None:
    """A bad resource callback must not abort training startup."""

    def callback(**_resources):
        raise RuntimeError("callback boom")

    async_rl_loop._notify_resources_ready(callback, deployment_id="deployment-1")

    assert "on_resources_ready callback failed" in caplog.text


def test_estimated_total_steps_accounts_for_resume_and_minibatches() -> None:
    """Total-step estimate follows the managed async artifact contract."""
    assert (
        async_rl_loop._estimate_total_steps(
            total_items=11,
            prior_rows_consumed=3,
            prompt_groups_per_step=2,
            ppo_n_minibatches=4,
            step_offset=8,
        )
        == 24
    )
