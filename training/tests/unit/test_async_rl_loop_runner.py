"""Pure-logic tests for the async_rl_loop runtime helpers.

Covers the deterministic pieces that don't require tinker, the Fireworks
SDK, or a deployment:

* ``estimate_async_total_steps`` -- the progress-bar total formula
  (batching, PPO inner minibatches, resume offset, clamping).
* ``Config.runner`` -- the RunnerConfig default + user override plumbing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from training.recipes import async_rl_loop
from training.utils.runner_state import estimate_async_total_steps


class _StopAfterProvisioning(RuntimeError):
    pass


class TestEstimateAsyncTotalSteps:
    """``estimate_async_total_steps`` mirrors the async progress formula."""

    def test_fresh_run_single_minibatch(self) -> None:
        """No resume, one minibatch per rollout -> one step per row."""
        total = estimate_async_total_steps(
            step_offset=0,
            total_items=10,
            prior_rows_consumed=0,
            prompt_groups_per_step=1,
            ppo_n_minibatches=1,
        )
        assert total == 10

    def test_fresh_run_batches_rows(self) -> None:
        """With ``prompt_groups_per_step=4`` we expect ceil(10/4) = 3 rollouts."""
        total = estimate_async_total_steps(
            step_offset=0,
            total_items=10,
            prior_rows_consumed=0,
            prompt_groups_per_step=4,
            ppo_n_minibatches=1,
        )
        assert total == 3

    def test_ppo_inner_minibatches_multiply_steps(self) -> None:
        """Each rollout batch fans out into ``ppo_n_minibatches`` optim steps."""
        total = estimate_async_total_steps(
            step_offset=0,
            total_items=10,
            prior_rows_consumed=0,
            prompt_groups_per_step=2,
            ppo_n_minibatches=3,
        )
        # ceil(10/2) = 5 rollout batches, each 3 optim steps -> 15
        assert total == 15

    def test_resume_adds_step_offset(self) -> None:
        """Resume from step N -> total includes N + remaining estimate."""
        total = estimate_async_total_steps(
            step_offset=4,
            total_items=10,
            prior_rows_consumed=6,
            prompt_groups_per_step=2,
            ppo_n_minibatches=1,
        )
        # remaining = 10 - 6 = 4 rows; ceil(4/2) = 2 rollouts; 4 + 2 = 6
        assert total == 6

    def test_clamps_negative_remaining(self) -> None:
        """If the cursor is past total_items we still return at least the offset."""
        total = estimate_async_total_steps(
            step_offset=7,
            total_items=10,
            prior_rows_consumed=12,
            prompt_groups_per_step=1,
            ppo_n_minibatches=1,
        )
        assert total == 7

    def test_invalid_groups_per_step_floors_to_one(self) -> None:
        """A zero/negative ``prompt_groups_per_step`` is treated as 1."""
        total = estimate_async_total_steps(
            step_offset=0,
            total_items=5,
            prior_rows_consumed=0,
            prompt_groups_per_step=0,
            ppo_n_minibatches=1,
        )
        assert total == 5

    def test_zero_total_items_returns_offset_only(self) -> None:
        """An empty dataset reports the step offset as the total."""
        total = estimate_async_total_steps(
            step_offset=3,
            total_items=0,
            prior_rows_consumed=0,
            prompt_groups_per_step=1,
            ppo_n_minibatches=2,
        )
        assert total == 3


class TestConfigRunnerField:
    """``Config.runner`` accepts a ``RunnerConfig`` so the orchestrator can
    plumb status / metadata / metrics / output_model paths through to the
    cookbook RunnerIO."""

    def test_config_exposes_runner_default(self) -> None:
        """Default ``Config.runner`` is an empty RunnerConfig (no outputs)."""
        from training.utils.runner import RunnerConfig

        cfg = async_rl_loop.Config(log_path="gs://logs")
        assert isinstance(cfg.runner, RunnerConfig)
        assert not cfg.runner.enabled

    def test_config_accepts_user_runner_config(self) -> None:
        """The orchestrator can construct Config with a populated runner."""
        from training.utils.runner import RunnerConfig

        runner_cfg = RunnerConfig(
            status_file="gs://r/status.json",
            metadata_file="gs://r/meta.json",
            metrics_file="gs://r/metrics.jsonl",
            output_model_path="gs://r/model.json",
        )
        cfg = async_rl_loop.Config(log_path="gs://logs", runner=runner_cfg)
        assert cfg.runner.status_file == "gs://r/status.json"
        assert cfg.runner.metadata_file == "gs://r/meta.json"
        assert cfg.runner.metrics_file == "gs://r/metrics.jsonl"
        assert cfg.runner.output_model_path == "gs://r/model.json"
        assert cfg.runner.enabled

    def test_config_cleanup_defaults_on(self) -> None:
        cfg = async_rl_loop.Config(log_path="gs://logs")

        assert cfg.cleanup_on_exit is True


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


class TestMinibatchWandbMetrics:
    def test_logs_lr_even_when_remote_metrics_are_empty(self) -> None:
        metrics = async_rl_loop._build_minibatch_wandb_metrics(
            SimpleNamespace(metrics={}),
            SimpleNamespace(metrics={}),
            step=3,
            minibatch_idx=2,
            num_minibatches=4,
            learning_rate=5e-5,
        )

        assert metrics == {
            "train/step": 3,
            "train/minibatch_idx": 2,
            "train/num_minibatches": 4,
            "train/lr": 5e-5,
        }

    def test_scheduled_lr_overrides_remote_lr_metric(self) -> None:
        metrics = async_rl_loop._build_minibatch_wandb_metrics(
            SimpleNamespace(metrics={"loss": 1.25, "step": 99}),
            SimpleNamespace(metrics={"lr": 1e-3, "grad_norm": 0.5}),
            step=7,
            minibatch_idx=1,
            num_minibatches=2,
            learning_rate=2e-5,
        )

        assert metrics["train/lr"] == 2e-5
        assert metrics["train/loss"] == 1.25
        assert metrics["train/grad_norm"] == 0.5
        assert "train/step_id" not in metrics
        assert metrics["train/step"] == 7


# ---------------------------------------------------------------------------
# End-to-end ``async_rl_loop.main`` exercise (with fake deps)
#
# Unit tests verify the final success/failure status and metrics writes.
# The helper tests above only cover the tiny callback / formula surface;
# these tests stub every heavy dependency and drive ``async_rl_loop.main``
# so the real ``RunnerIO`` writes land on disk.
# ---------------------------------------------------------------------------
