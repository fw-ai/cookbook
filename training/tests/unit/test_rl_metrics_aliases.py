"""Tests for rollout/batch metric naming."""

from __future__ import annotations

from types import SimpleNamespace

import tinker
from tinker.lib.chunked_fwdbwd_helpers import combine_fwd_bwd_output_results

from training.utils.rl.losses import PromptGroup
from training.utils.rl.metrics import (
    build_accumulated_async_loop_stats,
    build_loop_metrics,
    compute_step_metrics,
)


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
        assert "perf/trainer_wait_for_sampler_time" not in loop_metrics
        assert "perf/trainer_idle_ratio" not in loop_metrics
        assert "perf/scheduler_step_wall_time" not in loop_metrics
        assert "perf/rollout_batch_wall_time" not in loop_metrics
        assert "perf/rollout_batch_samples_per_s" not in loop_metrics
        assert "perf/rollout_batch_tokens_per_s" not in loop_metrics
        assert "perf/step_samples_per_s" not in loop_metrics
        assert "perf/step_tokens_per_s" not in loop_metrics
        assert "rollout/filter_drop_ratio" not in loop_metrics
        assert "rollout/zero_std_ratio" not in loop_metrics


class TestAsyncMetricHelpers:
    def test_build_accumulated_async_loop_stats_merges_chunk_stats(self):
        pg1 = _make_prompt_group()
        pg2 = _make_prompt_group()
        latest = {
            "filter_drops": 1,
            "sample_fails": 2,
            "total_sampled": 3,
            "resolved_rows": 4,
        }

        loop_stats = build_accumulated_async_loop_stats(
            prompt_groups=[pg1, pg2],
            latest_loop_stats=latest,
            trainer_wait_for_sampler_time=0.7,
            sampler_wait_for_trainer_time=0.2,
            train_wall_time=1.3,
        )

        assert loop_stats is not None
        assert loop_stats is not latest
        assert loop_stats["valid_prompt_groups"] == 2
        assert loop_stats["all_raw_rewards"] == [1.0, 1.0]
        assert loop_stats["trainer_wait_for_sampler_time"] == 0.7
        assert loop_stats["sampler_wait_for_trainer_time"] == 0.2
        assert loop_stats["train_wall_time"] == 1.3
        assert loop_stats["scheduler_step_wall_time"] == 2.0
        assert loop_stats["resolved_rows"] == 4


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
                "trainer_wait_for_sampler_time": 3.0,
                "rollout_batch_wall_time": 2.0,
                "train_wall_time": 1.0,
                "scheduler_step_wall_time": 6.0,
                "all_raw_rewards": [1.0, 0.0],
                "async/in_flight_at_train_start": 3,
                "async/rollout_tasks_completed_during_train": 2,
                "pipeline/chunk_idx": 1,
                "pipeline/chunks_per_step": 4,
            },
            completions_per_prompt=8,
        )

        assert metrics["rollout/valid_prompt_groups"] == 6
        assert metrics["rollout/samples_completed"] == 1
        assert metrics["rollout/trained_datums"] == 1
        assert metrics["train/target_tokens"] == 3
        assert metrics["rollout/filter_accept_ratio"] == 1.0 - 1 / 7
        assert metrics["rollout/filter_reject_ratio"] == 1 / 7
        assert metrics["rollout/raw_samples_completed"] == 2
        assert metrics["rollout/raw_reward"] == 0.5
        assert metrics["rollout/raw_accuracy"] == 0.5
        assert metrics["rollout/sample_fail_count"] == 2
        assert metrics["perf/trainer_wait_for_sampler_time"] == 3.0
        assert metrics["perf/rollout_batch_wall_time"] == 2.0
        assert metrics["perf/train_step_wall_time"] == 1.0
        assert metrics["perf/scheduler_step_wall_time"] == 6.0
        assert metrics["perf/trainer_idle_ratio"] == 0.5
        assert metrics["perf/rollout_batch_wall_ratio"] == 1 / 3
        assert metrics["perf/train_step_wall_ratio"] == 1 / 6
        assert metrics["perf/step_samples_per_s"] == 1 / 6
        assert metrics["perf/step_tokens_per_s"] == 1 / 3
        assert metrics["perf/rollout_batch_samples_per_s"] == 0.5
        assert metrics["perf/rollout_batch_tokens_per_s"] == 1.0
        assert metrics["async/in_flight_at_train_start"] == 3
        assert "async/rollout_tasks_completed_during_train" not in metrics
        assert metrics["pipeline/chunk_idx"] == 1
        assert metrics["pipeline/chunks_per_step"] == 4
        assert "perf/wait_time_ratio" not in metrics
        assert "perf/overlap_ratio" not in metrics
        assert "perf/step_wall_time" not in metrics
        assert "perf/rollout_samples_per_s" not in metrics
        assert "perf/rollout_tokens_per_s" not in metrics

        assert "rollout/pass@1" not in metrics
        assert "rollout/pass@2" not in metrics
        assert "rollout/pass@4" not in metrics
        assert "rollout/valid_prompts" not in metrics
        assert "rollout/filter_ratio" not in metrics
        assert "rollout/sample_fails" not in metrics
        assert "batch/mean_groups_per_fwd_bwd" not in metrics


class TestFwdBwdMinibatchAveraging:
    """With ppo_n_minibatches>1 the per-step train/* metrics must average
    across minibatches, not report only the last one."""

    @staticmethod
    def _fake_fwd_bwd(**metrics):
        return SimpleNamespace(metrics=dict(metrics))

    def test_averages_across_minibatches(self):
        fwd_bwds = [
            self._fake_fwd_bwd(ppo_clip_frac=0.0, ppo_ratio_mean=1.00),
            self._fake_fwd_bwd(ppo_clip_frac=0.1, ppo_ratio_mean=1.05),
            self._fake_fwd_bwd(ppo_clip_frac=0.3, ppo_ratio_mean=1.20),
        ]
        metrics = compute_step_metrics(
            prompt_groups=[],
            fwd_bwd_results=fwd_bwds,
            optim_result=None,
            n_accum=len(fwd_bwds),
            timing_metrics={},
        )
        assert metrics["train/ppo_clip_frac"] == (0.0 + 0.1 + 0.3) / 3
        assert metrics["train/ppo_ratio_mean"] == (1.0 + 1.05 + 1.2) / 3

    def test_k1_matches_single_fwd_bwd_behavior(self):
        """K=1: report the single minibatch's metrics directly (pre-PR behavior)."""
        only = self._fake_fwd_bwd(ppo_clip_frac=0.42, ppo_ratio_mean=1.07)
        metrics = compute_step_metrics(
            prompt_groups=[],
            fwd_bwd_results=[only],
            optim_result=None,
            n_accum=1,
            timing_metrics={},
        )
        assert metrics["train/ppo_clip_frac"] == 0.42
        assert metrics["train/ppo_ratio_mean"] == 1.07

    def test_k1_preserves_tinker_reduced_runtime_dp_sharding_evidence(self):
        server_chunks = [
            tinker.ForwardBackwardOutput(
                loss_fn_output_type="cross_entropy",
                loss_fn_outputs=[],
                metrics={
                    "dp_sharded_counts:min": True,
                    "dp_sharded_counts:max": True,
                    "local_input_sequences:sum": local_count,
                    # ``:last`` is not a Tinker reducer.  This pins the live
                    # failure mode: these legacy spellings are silently lost.
                    "dp_sharded_counts:last": True,
                    "local_input_sequences:last": local_count,
                },
            )
            for local_count in (1, 2)
        ]
        only = combine_fwd_bwd_output_results(server_chunks)

        assert only.metrics["dp_sharded_counts:min"] is True
        assert only.metrics["dp_sharded_counts:max"] is True
        assert only.metrics["local_input_sequences:sum"] == 3
        assert "dp_sharded_counts:last" not in only.metrics
        assert "local_input_sequences:last" not in only.metrics

        metrics = compute_step_metrics(
            prompt_groups=[],
            fwd_bwd_results=[only],
            optim_result=None,
            n_accum=1,
            timing_metrics={},
        )

        # Cookbook metric aggregation retains Tinker's reducer suffixes; the
        # promotion gate validates min == max and maps these to its canonical
        # numerics fields.
        assert metrics["train/dp_sharded_counts:min"] == 1.0
        assert metrics["train/dp_sharded_counts:max"] == 1.0
        assert metrics["train/local_input_sequences:sum"] == 3.0
        assert "train/dp_sharded_counts" not in metrics
        assert "train/local_input_sequences" not in metrics

    def test_empty_fwd_bwd_results_emits_no_train_keys(self):
        metrics = compute_step_metrics(
            prompt_groups=[],
            fwd_bwd_results=[],
            optim_result=None,
            n_accum=0,
            timing_metrics={},
        )
        assert not any(k.startswith("train/ppo_") for k in metrics)
