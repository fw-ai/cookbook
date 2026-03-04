"""Tests for rollout/batch metric naming."""

from __future__ import annotations

from types import SimpleNamespace

from training.utils.rl.losses import PromptGroup
from training.utils.rl.metrics import build_loop_metrics, compute_step_metrics


def _make_prompt_group() -> PromptGroup:
    target = SimpleNamespace(shape=[3], data=[11, 12, 13])
    datum = SimpleNamespace(loss_fn_inputs={"target_tokens": target})
    return PromptGroup(
        data=[datum],
        advantages=[1.0],
        ref_logprobs=[[0.1, 0.1, 0.1]],
        prompt_len=2,
        rewards=[1.0],
        inf_logprobs=[[-0.2, -0.3, -0.4]],
        completion_lens=[2],
        truncated=[False],
    )


class TestBuildLoopMetrics:
    def test_excludes_loop_rollout_count_metrics(self):
        loop_metrics = build_loop_metrics(
            train_step=3,
            sample_fails=1,
        )

        assert "rollout/filter_drop_count" not in loop_metrics
        assert "rollout/prompts_completed" not in loop_metrics
        assert "rollout/samples_completed" not in loop_metrics
        assert "rollout/tokens_completed" not in loop_metrics
        assert "rollout/sample_fail_ratio" not in loop_metrics
        assert "rollout/raw_reward" not in loop_metrics
        assert "perf/sample_wait_time" not in loop_metrics
        assert "perf/wait_time_ratio" not in loop_metrics
        assert "perf/overlap_ratio" not in loop_metrics
        assert "perf/step_wall_time" not in loop_metrics
        assert "perf/rollout_samples_per_s" not in loop_metrics
        assert "perf/rollout_tokens_per_s" not in loop_metrics
        assert "rollout/filter_drop_ratio" not in loop_metrics
        assert "rollout/zero_std_ratio" not in loop_metrics


class TestComputeStepMetrics:
    def test_uses_canonical_loop_stats_keys(self):
        metrics = compute_step_metrics(
            prompt_groups=[_make_prompt_group()],
            fwd_bwd_results=[SimpleNamespace(metrics={"loss": 0.5})],
            optim_result=SimpleNamespace(metrics={"lr": 1e-5}),
            n_accum=4,
            timing_metrics={"perf/fwd_bwd_time": 1.0},
            loop_stats={
                "valid_prompt_groups": 6,
                "total_sampled": 7,
                "filter_drops": 1,
                "sample_fails": 2,
                "sample_wait_time": 3.0,
                "step_wall_time": 4.0,
                "all_raw_rewards": [1.0, 0.0],
            },
            completions_per_prompt=8,
        )

        assert metrics["rollout/valid_prompt_groups"] == 6
        assert metrics["rollout/samples_completed"] == 1
        assert metrics["rollout/filter_reject_ratio"] == 1 / 7
        assert metrics["rollout/sample_fail_count"] == 2
        assert metrics["perf/sample_wait_time"] == 3.0
        assert metrics["perf/wait_time_ratio"] == 0.75
        assert metrics["perf/overlap_ratio"] == 0.25
        assert metrics["perf/step_wall_time"] == 4.0
        assert metrics["perf/rollout_samples_per_s"] == 0.25
        assert metrics["perf/rollout_tokens_per_s"] == 0.5

        assert "rollout/pass@1" not in metrics
        assert "rollout/pass@2" not in metrics
        assert "rollout/pass@4" not in metrics
        assert "rollout/valid_prompts" not in metrics
        assert "rollout/filter_ratio" not in metrics
        assert "rollout/sample_fails" not in metrics
        assert "batch/mean_groups_per_fwd_bwd" not in metrics
