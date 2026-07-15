"""Pure-logic tests for the async_rl_loop runtime helpers.

Covers the deterministic pieces that don't require tinker, the Fireworks
SDK, or a deployment.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from training.recipes import async_rl_loop
from training.utils.rl.metrics import build_train_chunk_metrics


class _StopAfterProvisioning(RuntimeError):
    pass


class TestConfigDefaults:
    def test_config_has_no_runner_state(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert not hasattr(cfg, "runner")

    def test_config_cleanup_defaults_on(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert cfg.cleanup_on_exit is True

    def test_config_pipeline_chunks_default_to_one(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert cfg.pipeline_chunks_per_step == 1

    def test_config_uses_conservative_loss_defaults(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert cfg.loss_path == "client"
        assert cfg.use_rollout_logprobs is False


# ---------------------------------------------------------------------------
# SDK service construction
# ---------------------------------------------------------------------------


def _build_service_kwargs(monkeypatch: pytest.MonkeyPatch, cfg: async_rl_loop.Config) -> dict:
    calls = []

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(async_rl_loop, "setup_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(async_rl_loop, "validate_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(async_rl_loop, "load_deployment_tokenizer", lambda *args, **kwargs: object())

    def fake_build_service_client(**kwargs):
        calls.append(kwargs)
        raise _StopAfterProvisioning

    monkeypatch.setattr(async_rl_loop, "build_service_client", fake_build_service_client)

    with pytest.raises(_StopAfterProvisioning):
        async_rl_loop.main(
            cfg,
            rows=[{"prompt": "1+1"}],
            rollout_fn_factory=lambda _setup: (lambda _sample: None),
        )

    assert len(calls) == 1
    return calls[0]


def test_main_requests_cleanup_for_sdk_created_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = async_rl_loop.Config(
        log_path="/tmp/async_rl_test_logs",
        deployment=async_rl_loop.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )

    kwargs = _build_service_kwargs(monkeypatch, cfg)

    assert kwargs["cleanup_trainer_on_close"] is True
    assert (
        kwargs["cleanup_deployment_on_close"]
        == async_rl_loop.CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO
    )


def test_main_can_disable_cleanup_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = async_rl_loop.Config(
        log_path="/tmp/async_rl_test_logs",
        cleanup_on_exit=False,
        deployment=async_rl_loop.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )

    kwargs = _build_service_kwargs(monkeypatch, cfg)

    assert kwargs["cleanup_trainer_on_close"] is False
    assert kwargs["cleanup_deployment_on_close"] is None


def test_main_requests_trainer_cleanup_for_empty_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = async_rl_loop.Config(
        log_path="/tmp/async_rl_test_logs",
        trainer=async_rl_loop.TrainerConfig(job_id=""),
        deployment=async_rl_loop.DeployConfig(tokenizer_model="Qwen/Qwen3-1.7B"),
    )

    kwargs = _build_service_kwargs(monkeypatch, cfg)

    assert kwargs["cleanup_trainer_on_close"] is True


class TestTrainChunkMetrics:
    def test_logs_lr_even_when_remote_metrics_are_empty(self) -> None:
        metrics = build_train_chunk_metrics(
            SimpleNamespace(metrics={}),
            SimpleNamespace(metrics={}),
            step=3,
            chunk_idx=2,
            num_chunks=4,
            learning_rate=5e-5,
            run_optimizer_step=False,
        )

        assert metrics == {
            "train/step": 3,
            "train/chunk_idx": 2,
            "train/num_chunks": 4,
            "train/lr": 5e-5,
            "train/run_optimizer_step": 0,
        }

    def test_scheduled_lr_overrides_remote_lr_metric(self) -> None:
        metrics = build_train_chunk_metrics(
            SimpleNamespace(metrics={"loss": 1.25, "step": 99}),
            SimpleNamespace(metrics={"lr": 1e-3, "grad_norm": 0.5}),
            step=7,
            chunk_idx=1,
            num_chunks=2,
            learning_rate=2e-5,
            run_optimizer_step=True,
        )

        assert metrics["train/lr"] == 2e-5
        assert metrics["train/loss"] == 1.25
        assert metrics["train/grad_norm"] == 0.5
        assert metrics["train/run_optimizer_step"] == 1
        assert "train/step_id" not in metrics
        assert metrics["train/step"] == 7
