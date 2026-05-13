"""FIR2-1599: artifact writer + resource-ready callback tests for async_rl_loop.

These tests cover the small, deterministic helpers that mediate the
managed runtime artifact contract:

* ``_emit_resources_ready`` -- callback invocation contract.
* ``_estimate_total_steps`` -- progress-bar total formula.

The full async loop is intentionally NOT exercised here: training
itself depends on tinker / Fireworks SDK / a real deployment. The
orchestrator-side test (FIR2-1599) feeds these helpers a fake async
loop to verify the end-to-end resources.json + status contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from training.recipes import async_rl_loop


@dataclass
class _FakeInfra:
    """Subset of ``training.utils.rl.Infra`` exercised by the callback."""

    deployment_id: str | None = None
    policy_job_id: str | None = None
    reference_job_id: str | None = None


class TestEmitResourcesReady:
    """``_emit_resources_ready`` is the FIR2-1599 resource-ready hook surface."""

    def test_noop_when_callback_is_none(self) -> None:
        """A missing callback is silently skipped (no kwargs required)."""
        infra = _FakeInfra(deployment_id="dep-1", policy_job_id="pol-1")
        async_rl_loop._emit_resources_ready(infra, None)

    def test_invokes_callback_with_resource_ids(self) -> None:
        """The callback receives deployment/policy/reference IDs as kwargs."""
        captured: list[dict[str, str | None]] = []

        def callback(**kwargs: object) -> None:
            captured.append(dict(kwargs))

        infra = _FakeInfra(
            deployment_id="dep-abc",
            policy_job_id="policy-xyz",
            reference_job_id="ref-123",
        )
        async_rl_loop._emit_resources_ready(infra, callback)

        assert captured == [
            {
                "deployment_id": "dep-abc",
                "policy_job_id": "policy-xyz",
                "reference_job_id": "ref-123",
            }
        ]

    def test_passes_none_for_missing_reference_job(self) -> None:
        """LoRA shared / forward-only setups have ``reference_job_id=None``."""
        captured: list[dict[str, str | None]] = []

        def callback(**kwargs: object) -> None:
            captured.append(dict(kwargs))

        infra = _FakeInfra(
            deployment_id="dep-1",
            policy_job_id="pol-1",
            reference_job_id=None,
        )
        async_rl_loop._emit_resources_ready(infra, callback)

        assert captured == [
            {"deployment_id": "dep-1", "policy_job_id": "pol-1", "reference_job_id": None}
        ]

    def test_callback_exceptions_do_not_propagate(self, caplog: pytest.LogCaptureFixture) -> None:
        """A broken artifact writer must never abort training."""
        infra = _FakeInfra(deployment_id="dep-1", policy_job_id="pol-1")

        def boom(**_kwargs: object) -> None:
            raise RuntimeError("disk full")

        with caplog.at_level("WARNING"):
            async_rl_loop._emit_resources_ready(infra, boom)

        assert any("on_resources_ready" in rec.message for rec in caplog.records)


class TestEstimateTotalSteps:
    """``_estimate_total_steps`` mirrors the FIR2-1599 progress formula."""

    def test_fresh_run_single_minibatch(self) -> None:
        """No resume, one minibatch per rollout -> one step per row."""
        total = async_rl_loop._estimate_total_steps(
            step_offset=0,
            total_items=10,
            prior_rows_consumed=0,
            prompt_groups_per_step=1,
            ppo_n_minibatches=1,
        )
        assert total == 10

    def test_fresh_run_batches_rows(self) -> None:
        """With ``prompt_groups_per_step=4`` we expect ceil(10/4) = 3 rollouts."""
        total = async_rl_loop._estimate_total_steps(
            step_offset=0,
            total_items=10,
            prior_rows_consumed=0,
            prompt_groups_per_step=4,
            ppo_n_minibatches=1,
        )
        assert total == 3

    def test_ppo_inner_minibatches_multiply_steps(self) -> None:
        """Each rollout batch fans out into ``ppo_n_minibatches`` optim steps."""
        total = async_rl_loop._estimate_total_steps(
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
        total = async_rl_loop._estimate_total_steps(
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
        total = async_rl_loop._estimate_total_steps(
            step_offset=7,
            total_items=10,
            prior_rows_consumed=12,
            prompt_groups_per_step=1,
            ppo_n_minibatches=1,
        )
        assert total == 7

    def test_invalid_groups_per_step_floors_to_one(self) -> None:
        """A zero/negative ``prompt_groups_per_step`` is treated as 1."""
        total = async_rl_loop._estimate_total_steps(
            step_offset=0,
            total_items=5,
            prior_rows_consumed=0,
            prompt_groups_per_step=0,
            ppo_n_minibatches=1,
        )
        assert total == 5

    def test_zero_total_items_returns_offset_only(self) -> None:
        """An empty dataset reports the step offset as the total."""
        total = async_rl_loop._estimate_total_steps(
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
    cookbook RunnerIO (FIR2-1599)."""

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


class TestResourceCallbackSignature:
    """``ResourceCallback`` is a kwargs-only contract so we can add future
    identifiers without breaking existing callers (FIR2-1599)."""

    def test_callable_with_only_documented_kwargs(self) -> None:
        """Today's keyword args are deployment_id / policy_job_id / reference_job_id."""
        seen: list[set[str]] = []

        def cb(**kwargs: object) -> None:
            seen.append(set(kwargs))

        async_rl_loop._emit_resources_ready(
            _FakeInfra(deployment_id="d", policy_job_id="p", reference_job_id="r"),
            cb,
        )
        assert seen == [{"deployment_id", "policy_job_id", "reference_job_id"}]
