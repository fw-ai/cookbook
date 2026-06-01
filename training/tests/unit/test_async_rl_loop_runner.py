"""Pure-logic tests for the async_rl_loop runtime helpers.

Covers the deterministic pieces that don't require tinker, the Fireworks
SDK, or a deployment:

* ``estimate_async_total_steps`` -- the progress-bar total formula
  (batching, PPO inner minibatches, resume offset, clamping).
* ``Config.runner`` -- the RunnerConfig default + user override plumbing.
"""

from __future__ import annotations

from training.recipes import async_rl_loop
from training.utils.runner_state import estimate_async_total_steps


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


# ---------------------------------------------------------------------------
# End-to-end ``async_rl_loop.main`` exercise (with fake deps)
#
# Unit tests verify the final success/failure status and metrics writes.
# The helper tests above only cover the tiny callback / formula surface;
# these tests stub every heavy dependency and drive ``async_rl_loop.main``
# so the real ``RunnerIO`` writes land on disk.
# ---------------------------------------------------------------------------
