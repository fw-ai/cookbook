"""Contracts for the loop-agnostic step-metric helpers."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from fireworks.training.sdk.client import WeightSyncResponse
from fireworks.training.sdk.sampling import ServerMetrics

from training.utils.logging import log_run_resources
from training.utils.metrics import (
    forward_backward_metrics,
    optimizer_metrics,
    sampler_metrics,
    weight_sync_metrics,
)
from training.utils.rl.losses import PromptGroup
from training.utils.rl.metrics import rollout_metrics


def _result(**metrics):
    return SimpleNamespace(metrics=metrics)


class _Sampler:
    def __init__(self, records):
        self.records = list(records)

    def drain_metrics(self):
        records, self.records = self.records, []
        return records


class TestForwardBackwardMetrics:
    def test_reduces_chunks_by_tinker_suffix(self):
        metrics = forward_backward_metrics(
            [
                _result(**{"loss:sum": 2.0, "kl:max": 0.1, "ppo_clip_frac": 0.2}),
                _result(**{"loss:sum": 3.0, "kl:max": 0.3, "ppo_clip_frac": 0.4}),
                _result(**{"loss:sum": 1.0, "kl:min": 0.05}),
            ]
        )

        assert metrics["train/loss:sum"] == 6.0
        assert metrics["train/kl:max"] == 0.3
        assert metrics["train/kl:min"] == 0.05
        # Averaged over the two chunks that reported it, not all three.
        assert metrics["train/ppo_clip_frac"] == pytest.approx(0.3)

    def test_drops_bookkeeping_last_and_nonfinite_values(self):
        metrics = forward_backward_metrics(
            [
                _result(
                    **{
                        "total_tokens:sum": 9,
                        "step_id": 3,
                        "loss_is_finite:last": 1.0,
                        "loss:mean": 1.0,
                        "entropy": math.nan,
                    }
                )
            ]
        )

        assert metrics == {}

    def test_maps_trainer_execution_time_and_busy_ratio(self):
        metrics = forward_backward_metrics(
            [
                _result(
                    **{
                        "trainer/forward_backward_duration_s:sum": 1.5,
                        "trainer_busy_walltime_pct:last": 40.0,
                    }
                ),
                _result(
                    **{
                        "trainer/forward_backward_duration_s:sum": 0.5,
                        "trainer_busy_walltime_pct:last": 55.0,
                    }
                ),
            ]
        )

        assert metrics == {
            "perf/trainer_fwd_bwd_time": 2.0,
            "perf/trainer_busy_walltime_pct": 55.0,
        }

    def test_mask_ratio_from_summed_token_counts(self):
        metrics = forward_backward_metrics(
            [
                _result(active_tokens=3, total_resp_tokens=4),
                _result(active_tokens=1, total_resp_tokens=4),
            ]
        )

        assert metrics["train/active_tokens"] == 4
        assert metrics["train/total_resp_tokens"] == 8
        assert metrics["train/mask_ratio"] == 0.5


class TestOptimizerMetrics:
    def test_keeps_canonical_values_and_trainer_time(self):
        metrics = optimizer_metrics(
            _result(
                **{
                    "lr:last": 1e-5,
                    "grad_norm:last": 2.0,
                    "grad_norm_post_clip": 2.0,
                    "step_id:last": 7,
                    "trainer/optim_step_duration_s:sum": 0.25,
                }
            )
        )

        assert metrics == {
            "train/lr": 1e-5,
            "train/grad_norm": 2.0,
            "perf/trainer_optim_step_time": 0.25,
        }

    def test_post_clip_norm_only_when_clipping_changed_it(self):
        metrics = optimizer_metrics(
            _result(grad_norm=4.0, grad_norm_post_clip=1.0)
        )

        assert metrics["train/grad_norm_post_clip"] == 1.0

    @pytest.mark.parametrize("result", [None, _result()])
    def test_missing_metrics_report_nothing(self, result):
        assert optimizer_metrics(result) == {}


def _record(**fields):
    """A drained sampler record; newer SDKs add client latency and usage."""
    defaults = dict(
        transport_error=False, http_status_code=None, retry_attempt=None
    )
    return SimpleNamespace(**{**defaults, **fields})


class TestSamplerMetrics:
    def test_summarizes_one_drain_of_server_metrics(self):
        sampler = _Sampler(
            [
                _record(
                    client_latency=2.0,
                    server_processing_time=1.5,
                    server_ttft=0.2,
                    prefill_queue_duration=0.1,
                    generation_duration=1.0,
                    completion_tokens=100,
                    prompt_tokens=50,
                    cached_prompt_tokens=25,
                    num_concurrent_requests=8,
                ),
                _record(
                    client_latency=4.0,
                    server_processing_time=3.0,
                    server_ttft=0.4,
                    prefill_queue_duration=0.3,
                    generation_duration=3.0,
                    completion_tokens=300,
                    prompt_tokens=50,
                    cached_prompt_tokens=0,
                    num_concurrent_requests=16,
                ),
                _record(http_status_code=503, retry_attempt=1),
                _record(transport_error=True, retry_attempt=2),
            ]
        )

        metrics = sampler_metrics(sampler.drain_metrics())

        assert metrics["sampler/requests"] == 4
        assert metrics["sampler/errors"] == 2
        assert metrics["sampler/retries"] == 1
        assert metrics["sampler/client_latency_s_mean"] == 3.0
        assert metrics["sampler/client_latency_s_max"] == 4.0
        assert metrics["sampler/server_time_s_mean"] == 2.25
        assert metrics["sampler/ttft_s_mean"] == pytest.approx(0.3)
        assert metrics["sampler/prefill_queue_s_max"] == 0.3
        assert metrics["sampler/concurrent_requests_mean"] == 12
        assert metrics["sampler/overhead_s_mean"] == pytest.approx(0.75)
        assert metrics["sampler/prompt_cache_hit_ratio"] == 0.25
        assert metrics["sampler/generation_tokens_per_s"] == 100.0
        assert "sampler/generation_queue_s_mean" not in metrics
        # The loop drains once; a second drain has nothing left.
        assert sampler_metrics(sampler.drain_metrics()) == {}

    def test_prefix_separates_evaluation_traffic(self):
        metrics = sampler_metrics([_record(client_latency=1.0)], prefix="eval/sampler")
        assert metrics["eval/sampler/requests"] == 1
        assert metrics["eval/sampler/client_latency_s_mean"] == 1.0
        assert not any(key.startswith("sampler/") for key in metrics)

    @pytest.mark.parametrize(
        "records,expected",
        [
            # 503, then success on the retry.
            ([_record(http_status_code=503, retry_attempt=1, error="HTTP 503"), _record(retry_attempt=2)], (2, 1, 1)),
            # HTTP 200 stream truncated, then success on the retry.
            ([_record(http_status_code=200, retry_attempt=1, error="SSE truncation"), _record(retry_attempt=2)], (2, 1, 1)),
            # Non-retryable 400.
            ([_record(http_status_code=400, retry_attempt=1, error="HTTP 400")], (1, 1, 0)),
        ],
        ids=["retry_recovered", "truncation_recovered", "terminal_400"],
    )
    def test_counts_per_attempt_sdk_records(self, records, expected):
        metrics = sampler_metrics(records)
        assert (metrics["sampler/requests"], metrics["sampler/errors"], metrics["sampler/retries"]) == expected

    @pytest.mark.skipif(
        "error" not in getattr(ServerMetrics, "__dataclass_fields__", {}),
        reason="needs an SDK with per-attempt sampler records",
    )
    def test_real_sampler_retry_and_terminal_failure(self, monkeypatch):
        import asyncio
        import random

        import httpx
        from fireworks.training.sdk.sampling import DeploymentSampler

        monkeypatch.setattr(random, "random", lambda: 0.0)

        async def _no_sleep(_seconds):
            return None

        monkeypatch.setattr(asyncio, "sleep", _no_sleep)

        def http_error(status):
            request = httpx.Request("POST", "https://example.invalid/v1/completions")
            response = httpx.Response(status, request=request)
            return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)

        outcomes = [http_error(503), "ok", http_error(400)]

        async def stream(*args, **kwargs):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            choice = {"text": "ok", "finish_reason": "stop", "raw_output": {"completion_token_ids": [7]}}
            return {"choices": [choice]}, ServerMetrics(server_processing_time=0.1)

        sampler = DeploymentSampler(
            inference_url="https://example.invalid", model="m", api_key="k", tokenizer=None
        )
        sampler.async_completions_stream = stream
        asyncio.run(sampler.sample_with_prompt_tokens([1, 2, 3]))  # 503 -> success
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(sampler.sample_with_prompt_tokens([1, 2, 3]))  # terminal 400
        metrics = sampler_metrics(sampler.drain_metrics())
        sampler.close()
        assert (metrics["sampler/requests"], metrics["sampler/errors"], metrics["sampler/retries"]) == (3, 2, 1)

    def test_released_sdk_records_without_client_fields(self):
        record = ServerMetrics(server_processing_time=1.0, generation_duration=2.0)

        metrics = sampler_metrics([record])

        assert metrics["sampler/server_time_s_mean"] == 1.0
        assert "sampler/overhead_s_mean" not in metrics
        assert "sampler/generation_tokens_per_s" not in metrics

    @pytest.mark.parametrize("records", [None, []])
    def test_no_requests_reports_nothing(self, records):
        assert sampler_metrics(records) == {}


class TestWeightSyncMetrics:
    def test_rdma_phase_times(self):
        result = WeightSyncResponse(
            version="step-1", metrics={"capture_s": 1.0, "hotload_s": 2.5}
        )

        assert weight_sync_metrics(result) == {
            "perf/weight_sync_capture_time": 1.0,
            "perf/weight_sync_hotload_time": 2.5,
        }

    def test_file_publication_reports_nothing(self):
        assert weight_sync_metrics(None) == {}


class TestRolloutMetrics:
    def _group(self, **overrides):
        values = dict(
            data=[],
            advantages=[],
            prompt_len=1,
            rewards=[1.0, 0.0],
            completion_lens=[10, 30],
            truncated=[False, True],
        )
        values.update(overrides)
        return PromptGroup(**values)

    def test_completion_and_filter_statistics(self):
        metrics = rollout_metrics(
            [self._group()], raw_rewards=[1.0, 0.0, 1.0, 1.0]
        )

        assert metrics == {
            "rollout/raw_reward": 0.75,
            "rollout/filtered_reward": 0.5,
            "rollout/raw_samples": 4,
            "rollout/filtered_samples": 2,
            "rollout/filter_ratio": 0.5,
            "rollout/completion_tokens_mean": 20.0,
            "rollout/completion_tokens_max": 30,
            "rollout/truncated_ratio": 0.5,
        }

    def test_without_raw_rewards_or_lengths(self):
        group = self._group(completion_lens=[], truncated=[])

        assert rollout_metrics([group]) == {}


class TestLogRunResources:
    def test_records_resolved_ids_and_skips_unresolved(self):
        class _Service:
            trainer_job_id = "trainer-1"
            reference_trainer_job_id = None

            @property
            def deployment_id(self):
                raise RuntimeError("SDK-managed service did not resolve deployment id")

        assert log_run_resources(_Service()) == {"trainer_job_id": "trainer-1"}

    def test_updates_active_wandb_config(self, monkeypatch):
        updates = []
        fake_wandb = SimpleNamespace(
            run=object(),
            config=SimpleNamespace(
                update=lambda values, allow_val_change: updates.append(values)
            ),
        )
        monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
        service = SimpleNamespace(trainer_job_id="t", deployment_id="d")

        log_run_resources(service)

        assert updates == [{"trainer_job_id": "t", "deployment_id": "d"}]
