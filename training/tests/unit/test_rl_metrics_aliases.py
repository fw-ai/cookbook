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
        assert "rollout/history_wipes" not in metrics
        assert "rollout/append_token_mismatches" not in metrics

    def test_tito_sidecar_summaries_merge_with_weighted_distributions(self):
        left = _make_prompt_group()
        right = _make_prompt_group()
        left.run_metadata = [
            {
                "tito_metrics": {
                    "tito/calls/total": 2.0,
                    "tito/lineage/new_segment": 2.0,
                    "tito/lineage/prefix_check": 1.0,
                    "tito/trajectory/policy_turns_count": 1.0,
                    "tito/trajectory/policy_turns_sum": 2.0,
                    "tito/trajectory/policy_turns_mean": 2.0,
                    "tito/trajectory/policy_turns_min": 2.0,
                    "tito/trajectory/policy_turns_max": 2.0,
                    "tito/turn/completion_tokens_count": 2.0,
                    "tito/turn/completion_tokens_sum": 6.0,
                    "tito/turn/completion_tokens_mean": 3.0,
                    "tito/turn/completion_tokens_min": 2.0,
                    "tito/turn/completion_tokens_max": 4.0,
                }
            }
        ]
        right.run_metadata = [
            {
                "tito_metrics": {
                    "tito/calls/total": 1.0,
                    "tito/lineage/new_segment": 1.0,
                    "tito/lineage/prefix_check": 3.0,
                    "tito/trajectory/policy_turns_count": 1.0,
                    "tito/trajectory/policy_turns_sum": 1.0,
                    "tito/trajectory/policy_turns_mean": 1.0,
                    "tito/trajectory/policy_turns_min": 1.0,
                    "tito/trajectory/policy_turns_max": 1.0,
                    "tito/turn/completion_tokens_count": 1.0,
                    "tito/turn/completion_tokens_sum": 12.0,
                    "tito/turn/completion_tokens_mean": 12.0,
                    "tito/turn/completion_tokens_min": 12.0,
                    "tito/turn/completion_tokens_max": 12.0,
                }
            }
        ]

        metrics = compute_step_metrics(
            prompt_groups=[left, right],
            fwd_bwd_results=[],
            optim_result=None,
            n_accum=1,
            timing_metrics={},
        )

        assert metrics["tito/turn/count"] == 3
        assert metrics["tito/turn/output_tokens_mean"] == 6
        assert metrics["tito/turn/output_tokens_min"] == 2
        assert metrics["tito/turn/output_tokens_max"] == 12
        assert metrics["tito/trajectory/count"] == 2
        assert metrics["tito/lineage/splits"] == 1
        assert metrics["tito/lineage/split_ratio"] == 1
        assert "tito/calls/total" not in metrics
        assert not any(name.startswith("debug/tito/") for name in metrics)

    def test_tito_sidecar_metric_root_is_fail_closed(self):
        group = _make_prompt_group()
        group.run_metadata = [{"tito_metrics": {"rollout/tito/calls/total": 1.0}}]

        with pytest.raises(ValueError, match="canonical root"):
            compute_step_metrics(
                prompt_groups=[group],
                fwd_bwd_results=[],
                optim_result=None,
                n_accum=1,
                timing_metrics={},
            )

    def test_tito_sidecar_distribution_schema_is_fail_closed(self):
        group = _make_prompt_group()
        group.run_metadata = [
            {
                "tito_metrics": {
                    "tito/turn/completion_tokens_count": 1.0,
                }
            }
        ]

        with pytest.raises(ValueError, match="lacks count or sum"):
            compute_step_metrics(
                prompt_groups=[group],
                fwd_bwd_results=[],
                optim_result=None,
                n_accum=1,
                timing_metrics={},
            )

    def test_tito_sidecar_zero_count_distribution_has_no_fabricated_stats(self):
        group = _make_prompt_group()
        group.run_metadata = [
            {
                "tito_metrics": {
                    "tito/turn/completion_tokens_count": 0.0,
                    "tito/turn/completion_tokens_sum": 0.0,
                }
            }
        ]

        metrics = compute_step_metrics(
            prompt_groups=[group],
            fwd_bwd_results=[],
            optim_result=None,
            n_accum=1,
            timing_metrics={},
        )

        base = "tito/turn/output_tokens"
        assert metrics["tito/turn/count"] == 0
        assert f"{base}_mean" not in metrics
        assert f"{base}_min" not in metrics
        assert f"{base}_max" not in metrics

    def test_tito_debug_mode_publishes_full_internal_metrics(self):
        group = _make_prompt_group()
        group.run_metadata = [
            {
                "tito_debug_enabled": True,
                "tito_metrics": {
                    "tito/calls/total": 2.0,
                    "tito/turn/completion_tokens_count": 1.0,
                    "tito/turn/completion_tokens_sum": 4.0,
                    "tito/turn/completion_tokens_mean": 4.0,
                    "tito/turn/completion_tokens_min": 4.0,
                    "tito/turn/completion_tokens_max": 4.0,
                },
            }
        ]

        metrics = compute_step_metrics(
            prompt_groups=[group],
            fwd_bwd_results=[],
            optim_result=None,
            n_accum=1,
            timing_metrics={},
        )

        assert metrics["tito/turn/output_tokens_mean"] == 4
        assert metrics["debug/tito/calls/total"] == 2
        assert metrics["debug/tito/turn/completion_tokens_count"] == 1
        # The debug section mirrors the full merged record as-is.
        assert metrics["debug/tito/turn/completion_tokens_mean"] == 4

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
        assert metrics["train/grad_norm_pre_norm"] == 3.0
        assert metrics["train/grad_norm_rms"] == 0.2
        assert metrics["train/grad_norm_lora"] == 3.0
        assert set(key for key in metrics if key.startswith("train/grad_norm")) == {
            "train/grad_norm",
            "train/grad_norm_pre_norm",
            "train/grad_norm_rms",
            "train/grad_norm_lora",
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


class TestTitoHarnessTimingMetrics:
    """Harness/model timing decomposition published from TITO summaries."""

    @staticmethod
    def _summaries() -> list[dict[str, float]]:
        # Two trajectories: per-turn distributions use count/sum/min/max.
        return [
            {
                "tito/turn/inter_call_gap_seconds_count": 3.0,
                "tito/turn/inter_call_gap_seconds_sum": 30.0,
                "tito/turn/inter_call_gap_seconds_min": 5.0,
                "tito/turn/inter_call_gap_seconds_max": 15.0,
                "tito/turn/request_wall_seconds_count": 3.0,
                "tito/turn/request_wall_seconds_sum": 60.0,
                "tito/turn/request_wall_seconds_min": 10.0,
                "tito/turn/request_wall_seconds_max": 30.0,
                "tito/calls/sampler_wall_seconds_count": 3.0,
                "tito/calls/sampler_wall_seconds_sum": 45.0,
                "tito/calls/sampler_wall_seconds_min": 5.0,
                "tito/calls/sampler_wall_seconds_max": 25.0,
                "tito/trial/environment_setup_seconds_count": 1.0,
                "tito/trial/environment_setup_seconds_sum": 12.0,
                "tito/trial/environment_setup_seconds_min": 12.0,
                "tito/trial/environment_setup_seconds_max": 12.0,
            },
            {
                "tito/turn/inter_call_gap_seconds_count": 1.0,
                "tito/turn/inter_call_gap_seconds_sum": 10.0,
                "tito/turn/inter_call_gap_seconds_min": 10.0,
                "tito/turn/inter_call_gap_seconds_max": 10.0,
                "tito/turn/request_wall_seconds_count": 1.0,
                "tito/turn/request_wall_seconds_sum": 20.0,
                "tito/turn/request_wall_seconds_min": 20.0,
                "tito/turn/request_wall_seconds_max": 20.0,
                "tito/calls/sampler_wall_seconds_count": 1.0,
                "tito/calls/sampler_wall_seconds_sum": 15.0,
                "tito/calls/sampler_wall_seconds_min": 15.0,
                "tito/calls/sampler_wall_seconds_max": 15.0,
                "tito/trial/environment_setup_seconds_count": 1.0,
                "tito/trial/environment_setup_seconds_sum": 8.0,
                "tito/trial/environment_setup_seconds_min": 8.0,
                "tito/trial/environment_setup_seconds_max": 8.0,
            },
        ]

    def test_publishes_harness_and_model_sums(self):
        from training.utils.rl.metrics import publish_tito_sidecar_metrics

        metrics: dict = {}
        publish_tito_sidecar_metrics(metrics, self._summaries())

        assert metrics["tito/harness/wall_seconds_sum"] == pytest.approx(40.0)
        assert metrics["tito/model/request_wall_seconds_sum"] == pytest.approx(80.0)
        assert metrics["tito/model/sampler_wall_seconds_sum"] == pytest.approx(60.0)
        assert metrics["tito/harness/environment_setup_seconds_sum"] == pytest.approx(
            20.0
        )
        # 40 harness seconds in a 120-second harness+model window.
        assert metrics["tito/harness/wait_fraction"] == pytest.approx(40.0 / 120.0)

    def test_promotes_harness_distributions(self):
        from training.utils.rl.metrics import publish_tito_sidecar_metrics

        metrics: dict = {}
        publish_tito_sidecar_metrics(metrics, self._summaries())

        assert metrics["tito/turn/wait_for_harness_seconds_mean"] == pytest.approx(10.0)
        assert metrics["tito/turn/wait_for_harness_seconds_min"] == pytest.approx(5.0)
        assert metrics["tito/turn/wait_for_harness_seconds_max"] == pytest.approx(15.0)
        assert metrics["tito/calls/sampling_seconds_mean"] == pytest.approx(15.0)
        assert metrics["tito/trial/environment_setup_seconds_mean"] == pytest.approx(
            10.0
        )

    def test_wait_fraction_omitted_without_model_window(self):
        from training.utils.rl.metrics import publish_tito_sidecar_metrics

        metrics: dict = {}
        publish_tito_sidecar_metrics(metrics, [{}])

        assert "tito/harness/wait_fraction" not in metrics
        assert "tito/harness/wall_seconds_sum" not in metrics


class TestTrialPhaseTimings:
    def test_extracts_harbor_timing_brackets(self):
        from datetime import datetime, timedelta, timezone

        from training.examples.rl.harbor.tito.trial import _trial_phase_timings

        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bracket = lambda seconds: SimpleNamespace(  # noqa: E731
            started_at=start,
            finished_at=start + timedelta(seconds=seconds),
        )
        result = SimpleNamespace(
            environment_setup=bracket(12.0),
            agent_setup=bracket(3.0),
            agent_execution=bracket(90.0),
            verifier=bracket(1.5),
            started_at=start,
            finished_at=start + timedelta(seconds=110.0),
        )

        assert _trial_phase_timings(result) == {
            "environment_setup_seconds": 12.0,
            "agent_setup_seconds": 3.0,
            "agent_execution_seconds": 90.0,
            "verifier_seconds": 1.5,
            "trial_wall_seconds": 110.0,
        }

    def test_omits_unrecorded_brackets(self):
        from training.examples.rl.harbor.tito.trial import _trial_phase_timings

        result = SimpleNamespace(
            environment_setup=None,
            agent_setup=None,
            agent_execution=None,
            verifier=None,
            started_at=None,
            finished_at=None,
        )
        assert _trial_phase_timings(result) == {}

    def test_attach_trial_phase_metrics_builds_one_sample_distributions(self):
        from training.examples.rl.harbor.tito.rollout import (
            _attach_trial_phase_metrics,
        )

        rollout = SimpleNamespace(metadata={"tito_metrics": {}})
        _attach_trial_phase_metrics(rollout, {"environment_setup_seconds": 7.5})

        summary = rollout.metadata["tito_metrics"]
        assert summary == {
            "tito/trial/environment_setup_seconds_count": 1.0,
            "tito/trial/environment_setup_seconds_sum": 7.5,
            "tito/trial/environment_setup_seconds_min": 7.5,
            "tito/trial/environment_setup_seconds_max": 7.5,
        }


class TestTitoTimingDecomposition:
    """Queue/sampling split, unaccounted trial time, and failure accounting."""

    @staticmethod
    def _summary(**overrides) -> dict[str, float]:
        summary = {
            "tito/turn/inter_call_gap_seconds_count": 2.0,
            "tito/turn/inter_call_gap_seconds_sum": 40.0,
            "tito/turn/inter_call_gap_seconds_min": 15.0,
            "tito/turn/inter_call_gap_seconds_max": 25.0,
            "tito/turn/request_wall_seconds_count": 2.0,
            "tito/turn/request_wall_seconds_sum": 60.0,
            "tito/turn/request_wall_seconds_min": 20.0,
            "tito/turn/request_wall_seconds_max": 40.0,
            "tito/calls/sampler_wall_seconds_count": 2.0,
            "tito/calls/sampler_wall_seconds_sum": 45.0,
            "tito/calls/sampler_wall_seconds_min": 15.0,
            "tito/calls/sampler_wall_seconds_max": 30.0,
        }
        summary.update(overrides)
        return summary

    @staticmethod
    def _trial_phases(**seconds) -> dict[str, float]:
        summary: dict[str, float] = {}
        for phase, value in seconds.items():
            base = f"tito/trial/{phase}"
            summary[f"{base}_count"] = 1.0
            summary[f"{base}_sum"] = value
            summary[f"{base}_min"] = value
            summary[f"{base}_max"] = value
        return summary

    def test_queue_overhead_separates_contention_from_sampling(self):
        from training.utils.rl.metrics import publish_tito_sidecar_metrics

        metrics: dict = {}
        publish_tito_sidecar_metrics(metrics, [self._summary()])

        # 60s client-visible model wall, 45s of it real generation.
        assert metrics["tito/model/queue_overhead_seconds_sum"] == pytest.approx(15.0)
        assert metrics["tito/harness/wait_fraction"] == pytest.approx(40.0 / 100.0)
        assert metrics["tito/model/sampling_fraction"] == pytest.approx(45.0 / 100.0)
        assert metrics["tito/model/queue_fraction"] == pytest.approx(15.0 / 100.0)
        assert metrics["tito/calls/sampling_seconds_mean"] == pytest.approx(22.5)
        assert "tito/turn/sampling_seconds_mean" not in metrics

    def test_sampler_wall_above_request_wall_never_goes_negative(self):
        from training.utils.rl.metrics import publish_tito_sidecar_metrics

        metrics: dict = {}
        publish_tito_sidecar_metrics(
            metrics,
            [
                self._summary(
                    **{
                        "tito/calls/sampler_wall_seconds_sum": 90.0,
                        "tito/calls/sampler_wall_seconds_max": 60.0,
                    }
                )
            ],
        )

        assert metrics["tito/model/queue_overhead_seconds_sum"] == 0.0
        assert metrics["tito/model/queue_fraction"] == 0.0
        assert metrics["tito/model/sampling_fraction"] == pytest.approx(60.0 / 100.0)

    def test_unaccounted_trial_time_exposes_inter_phase_gaps(self):
        from training.utils.rl.metrics import publish_tito_sidecar_metrics

        metrics: dict = {}
        summary = self._summary()
        summary.update(
            self._trial_phases(
                environment_setup_seconds=5.0,
                agent_setup_seconds=10.0,
                agent_execution_seconds=100.0,
                verifier_seconds=5.0,
                trial_wall_seconds=140.0,
                bound_utilization=0.25,
            )
        )
        publish_tito_sidecar_metrics(metrics, [summary])

        assert metrics["tito/trial/unaccounted_seconds_sum"] == pytest.approx(20.0)
        assert metrics["tito/harness/trial_wall_seconds_sum"] == pytest.approx(140.0)
        assert metrics["tito/trial/bound_utilization_max"] == pytest.approx(0.25)

    def test_failure_accounting_publishes_without_any_trajectory(self):
        from training.utils.rl import trial_events
        from training.utils.rl.metrics import publish_tito_sidecar_metrics

        trial_events.reset()
        trial_events.record_trial_attempt()
        trial_events.record_trial_attempt()
        trial_events.record_trial_timeout()
        trial_events.record_trial_retry("whole_trial_bound")
        trial_events.record_trial_discarded("RemoteProtocolError")
        trial_events.record_failed_trial_wall(120.0)
        trial_events.record_failed_trial_phases({"agent_setup_seconds": 30.0})

        metrics: dict = {}
        publish_tito_sidecar_metrics(metrics, [], drain_trial_events=True)

        # A step where every trial failed still reports what it burned.
        assert metrics["tito/trial/attempts"] == 2.0
        assert metrics["tito/trial/retries"] == 1.0
        assert metrics["tito/trial/retry_reason/whole_trial_bound"] == 1.0
        assert metrics["tito/trial/discard_reason/remoteprotocolerror"] == 1.0
        assert metrics["tito/trial/whole_trial_bound_firings"] == 1.0
        assert metrics["tito/trial_failed/trial_wall_seconds_sum"] == 120.0
        assert metrics["tito/trial_failed/agent_setup_seconds_max"] == 30.0
        assert not trial_events.drain()

    def test_drained_failure_accounting_does_not_leak_into_the_next_step(self):
        from training.utils.rl import trial_events
        from training.utils.rl.metrics import publish_tito_sidecar_metrics

        trial_events.reset()
        trial_events.record_trial_attempt()
        first: dict = {}
        publish_tito_sidecar_metrics(first, [self._summary()], drain_trial_events=True)
        second: dict = {}
        publish_tito_sidecar_metrics(second, [self._summary()], drain_trial_events=True)

        assert first["tito/trial/attempts"] == 1.0
        assert "tito/trial/attempts" not in second

    def test_evaluation_publisher_does_not_steal_step_failure_accounting(self):
        from training.utils.rl import trial_events
        from training.utils.rl.metrics import publish_tito_sidecar_metrics

        trial_events.reset()
        trial_events.record_trial_discarded("injected_fault")
        evaluation: dict = {}
        publish_tito_sidecar_metrics(evaluation, [self._summary()])
        step: dict = {}
        publish_tito_sidecar_metrics(step, [self._summary()], drain_trial_events=True)

        assert "tito/trial/discarded" not in evaluation
        assert step["tito/trial/discarded"] == 1.0
