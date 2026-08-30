"""Tests for rollout/batch metric naming."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import tinker
from tinker.lib.chunked_fwdbwd_helpers import combine_fwd_bwd_output_results

from training.utils.rl.losses import PromptGroup
from training.utils.rl.metrics import compute_step_metrics


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


class TestComputeStepMetrics:
    def test_uses_canonical_loop_stats_keys(self):
        metrics = compute_step_metrics(
            prompt_groups=[_make_prompt_group()],
            fwd_bwd_results=[SimpleNamespace(metrics={"loss": 0.5})],
            optim_result=SimpleNamespace(metrics={"lr": 1e-5}),
            n_accum=4,
            timing_metrics={"perf/fwd_bwd_time": 1.0},
            loop_stats={
                "perf/step_time": 6.0,
                "perf/train_time": 1.0,
                "perf/train_wait_time": 5.0,
                "perf/wait_time_ratio": 5 / 6,
                "perf/train_chunk_wait_time": 0.25,
                "all_raw_rewards": [1.0, 0.0],
                "async/in_flight_samples_mean": 6.5,
                "async/realized_training_chunks": 4,
                "async/trained_against_version": 2,
            },
        )

        rollout_metrics = {
            key: value for key, value in metrics.items() if key.startswith("rollout/")
        }
        assert rollout_metrics == {
            "rollout/raw_reward": 0.5,
            "rollout/filtered_reward": 1.0,
            "rollout/raw_samples": 2,
            "rollout/filtered_samples": 1,
            "rollout/filter_ratio": 0.5,
        }
        assert metrics["train/target_tokens"] == 3
        assert metrics["perf/step_time"] == 6.0
        assert metrics["perf/train_time"] == 1.0
        assert metrics["perf/train_wait_time"] == 5.0
        assert metrics["perf/wait_time_ratio"] == 5 / 6
        assert metrics["perf/train_chunk_wait_time"] == 0.25
        assert metrics["perf/step_samples_per_s"] == 1 / 6
        assert metrics["perf/step_tokens_per_s"] == 1 / 3
        assert metrics["async/in_flight_samples_mean"] == 6.5
        assert metrics["async/realized_training_chunks"] == 4
        assert metrics["async/trained_against_version"] == 2
        assert "perf/overlap_ratio" not in metrics
        assert "perf/trainer_idle_ratio" not in metrics
        assert "perf/trainer_wait_for_sampler_time" not in metrics
        assert "perf/sampler_wait_for_trainer_time" not in metrics
        assert "perf/scheduler_step_wall_time" not in metrics
        assert "perf/rollout_batch_wall_time" not in metrics
        assert "perf/rollout_samples_per_s" not in metrics
        assert "perf/rollout_tokens_per_s" not in metrics

        assert "rollout/pass@1" not in metrics
        assert "rollout/pass@2" not in metrics
        assert "rollout/pass@4" not in metrics
        assert "rollout/valid_prompts" not in metrics
        assert "rollout/sample_fails" not in metrics
        assert "rollout/raw_samples_completed" not in metrics
        assert "rollout/raw_accuracy" not in metrics
        assert "rollout/trained_datums" not in metrics
        assert "rollout/total_sampled" not in metrics
        assert "rollout/filter_accept_ratio" not in metrics
        assert "rollout/fwd_bwd_count" not in metrics
        assert "perf/train_step_wall_ratio" not in metrics
        assert "batch/mean_groups_per_fwd_bwd" not in metrics

    def test_rollout_sample_counts_are_trajectories_not_segments(self):
        group = _make_prompt_group()
        group.data.append(group.data[0])
        group.advantages.append(group.advantages[0])
        group.inf_logprobs.append(group.inf_logprobs[0])
        group.completion_lens.append(group.completion_lens[0])
        group.truncated.append(group.truncated[0])
        group.run_metadata = [
            {
                "segment_count": 2,
                "trainable_tokens": 3,
                "history_wipes": 1,
                "append_token_mismatches": 2,
            }
        ]

        metrics = compute_step_metrics(
            prompt_groups=[group],
            fwd_bwd_results=[],
            optim_result=None,
            n_accum=1,
            timing_metrics={},
            loop_stats={"all_raw_rewards": [1.0]},
        )

        assert len(group.data) == 2
        assert len(group.rewards) == 1
        assert metrics["rollout/raw_samples"] == 1
        assert metrics["rollout/filtered_samples"] == 1
        assert metrics["rollout/trainable_tokens_mean"] == 3
        assert metrics["rollout/history_wipes"] == 1
        assert metrics["rollout/append_token_mismatches"] == 2

    def test_optimizer_metrics_drop_remote_aliases(self):
        metrics = compute_step_metrics(
            prompt_groups=[],
            fwd_bwd_results=[],
            optim_result=SimpleNamespace(
                metrics={
                    "grad_norm": 3.0,
                    "grad_norm:last": 3.0,
                    "grad_norm_pre_norm": 3.0,
                    "grad_norm_post_clip": 3.0,
                    "grad_norm_lora": 3.0,
                    "grad_norm_rms": 0.2,
                    "grad_norm_rms:last": 0.2,
                    "lr:last": 1e-5,
                    "trainer_busy_walltime_pct:last": 80.0,
                }
            ),
            n_accum=0,
            timing_metrics={},
        )

        assert metrics["train/grad_norm"] == 3.0
        assert metrics["train/grad_norm_rms"] == 0.2
        assert set(key for key in metrics if key.startswith("train/grad_norm")) == {
            "train/grad_norm",
            "train/grad_norm_rms",
        }
        assert "train/lr:last" not in metrics
        assert "train/trainer_busy_walltime_pct:last" not in metrics

    def test_optimizer_metrics_keep_effective_clipped_norm(self):
        metrics = compute_step_metrics(
            prompt_groups=[],
            fwd_bwd_results=[],
            optim_result=SimpleNamespace(
                metrics={
                    "grad_norm": 3.0,
                    "grad_norm_post_clip": 1.0,
                }
            ),
            n_accum=0,
            timing_metrics={},
        )

        assert metrics["train/grad_norm"] == 3.0
        assert metrics["train/grad_norm_post_clip"] == 1.0


class TestFwdBwdResultAveraging:
    """Per-step train/* metrics average all forward/backward results."""

    @staticmethod
    def _fake_fwd_bwd(**metrics):
        return SimpleNamespace(metrics=dict(metrics))

    def test_averages_across_forward_backward_results(self):
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

    def test_sums_counts_and_recomputes_mask_ratio(self):
        fwd_bwds = [
            self._fake_fwd_bwd(
                active_tokens=3,
                total_resp_tokens=4,
                mask_ratio=0.75,
            ),
            self._fake_fwd_bwd(
                active_tokens=2,
                total_resp_tokens=6,
                mask_ratio=1 / 3,
            ),
        ]
        metrics = compute_step_metrics(
            prompt_groups=[],
            fwd_bwd_results=fwd_bwds,
            optim_result=None,
            n_accum=2,
            timing_metrics={},
        )

        assert metrics["train/active_tokens"] == 5
        assert metrics["train/total_resp_tokens"] == 10
        assert metrics["train/mask_ratio"] == 0.5

    def test_optional_metric_averages_only_reported_chunks(self):
        metrics = compute_step_metrics(
            prompt_groups=[],
            fwd_bwd_results=[
                self._fake_fwd_bwd(loss=1.0),
                self._fake_fwd_bwd(loss=3.0, inference_k3=0.25),
            ],
            optim_result=None,
            n_accum=2,
            timing_metrics={},
        )

        assert metrics["train/loss"] == 2.0
        assert metrics["train/inference_k3"] == 0.25
        assert not any(key.startswith("kld/") for key in metrics)

    def test_inference_drift_preserves_historical_chunk_mean(self):
        metrics = compute_step_metrics(
            prompt_groups=[],
            fwd_bwd_results=[
                self._fake_fwd_bwd(
                    inference_k1=1.0,
                    inference_k3=2.0,
                    raw_inference_logprob_coverage=1.0,
                ),
                self._fake_fwd_bwd(
                    inference_k1=3.0,
                    inference_k3=4.0,
                    raw_inference_logprob_coverage=0.5,
                ),
            ],
            optim_result=None,
            n_accum=2,
            timing_metrics={},
        )

        assert metrics["train/inference_k1"] == pytest.approx(2.0)
        assert metrics["train/inference_k3"] == pytest.approx(3.0)
        assert metrics["train/raw_inference_logprob_coverage"] == pytest.approx(0.75)

    def test_single_fwd_bwd_result_is_reported_directly(self):
        """Report a single forward/backward result without changing its metrics."""
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
