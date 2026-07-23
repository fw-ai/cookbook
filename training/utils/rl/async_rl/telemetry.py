"""Rate-limited observability for asynchronous RL scheduling."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, TypeAlias, TypeVar

from training.utils.rl.async_rl.batch import OptimizerBatch
from training.utils.rl.losses import PromptGroup
from training.utils.rl.metrics import (
    build_accumulated_async_loop_stats,
    compute_step_metrics,
)

logger = logging.getLogger(__name__)

ProducerSnapshot: TypeAlias = Mapping[str, int | float | bool | None]
ProducerMetricsFn: TypeAlias = Callable[[dict[str, Any]], None]
StepMetricsFn: TypeAlias = Callable[[dict[str, Any], int], None]
PostStepMetricsFn: TypeAlias = Callable[[dict[str, Any]], None]
T = TypeVar("T")

DEFAULT_PRODUCER_METRICS_INTERVAL_S = 10.0

_STEP_MEAN_GAUGES = {
    "in_flight_samples": "async/in_flight_samples_mean",
    "admission_capacity": "async/admission_capacity_samples_mean",
    "staleness_capacity": "async/staleness_capacity_samples_mean",
    "concurrency_capacity": "async/concurrency_capacity_samples_mean",
}


def producer_metric_values(snapshot: ProducerSnapshot) -> dict[str, int | float]:
    """Translate a producer snapshot into stable public metrics."""

    values: dict[str, int | float] = {
        "producer/published_version": int(snapshot["published_version"] or 0),
        "producer/in_flight_samples": int(snapshot["in_flight_samples"] or 0),
        "producer/admission_capacity_samples": int(snapshot["admission_capacity"] or 0),
        "producer/staleness_capacity_samples": int(snapshot["staleness_capacity"] or 0),
        "producer/rows_submitted_total": int(snapshot["rows_submitted"] or 0),
        "producer/rows_completed_total": int(snapshot["rows_accepted"] or 0)
        + int(snapshot["rows_rejected"] or 0),
        "producer/rows_rejected_total": int(snapshot["rows_rejected"] or 0),
        "producer/completion_refill_attempts_total": int(
            snapshot["completion_refill_attempts"] or 0
        ),
        "producer/completion_refill_rows_submitted_total": int(
            snapshot["completion_refill_rows_submitted"] or 0
        ),
        "producer/recoverable_errors_total": int(snapshot["recoverable_errors"] or 0),
        "producer/source_exhausted": int(bool(snapshot["source_exhausted"])),
    }
    concurrency_capacity = snapshot["concurrency_capacity"]
    if concurrency_capacity is not None:
        values["producer/concurrency_capacity_samples"] = int(concurrency_capacity)
    return values


class ProducerStepMetricsReducer:
    """Time-weight sampled producer gauges between optimizer publications."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last_time: float | None = None
        self._last_values: dict[str, float | None] = {}
        self._weighted_sums = {name: 0.0 for name in _STEP_MEAN_GAUGES}
        self._durations = {name: 0.0 for name in _STEP_MEAN_GAUGES}

    def observe(self, snapshot: ProducerSnapshot) -> None:
        now = self._clock()
        values = {
            name: None if snapshot[name] is None else float(snapshot[name])
            for name in _STEP_MEAN_GAUGES
        }
        if self._last_time is not None:
            elapsed = max(0.0, now - self._last_time)
            for name, value in self._last_values.items():
                if value is None:
                    continue
                self._weighted_sums[name] += value * elapsed
                self._durations[name] += elapsed
        self._last_time = now
        self._last_values = values

    def finish_step(self, snapshot: ProducerSnapshot) -> dict[str, float]:
        self.observe(snapshot)
        metrics = {
            metric_name: self._weighted_sums[name] / self._durations[name]
            for name, metric_name in _STEP_MEAN_GAUGES.items()
            if self._durations[name] > 0
        }
        self._weighted_sums = {name: 0.0 for name in _STEP_MEAN_GAUGES}
        self._durations = {name: 0.0 for name in _STEP_MEAN_GAUGES}
        return metrics


class ProducerMetricsReporter:
    """Poll and emit changed producer state on an independent event axis."""

    def __init__(
        self,
        *,
        snapshot_fn: Callable[[], ProducerSnapshot],
        metrics_fn: ProducerMetricsFn,
        interval_s: float = DEFAULT_PRODUCER_METRICS_INTERVAL_S,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("producer metrics interval must be > 0")
        self._snapshot_fn = snapshot_fn
        self._metrics_fn = metrics_fn
        self._interval_s = interval_s
        self._started_at = time.monotonic()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._last_values: dict[str, int | float] | None = None
        self._next_event = 0
        self._failure: BaseException | None = None
        self._step_reducer = ProducerStepMetricsReducer()

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("producer metrics reporter already started")
        self._step_reducer.observe(self._snapshot_fn())
        self._task = asyncio.create_task(
            self._run(),
            name="async-rl-producer-metrics",
        )

    def finish_step(self) -> dict[str, float]:
        if self._task is None:
            raise RuntimeError("producer metrics reporter is not started")
        if self._failure is not None:
            raise self._failure
        return self._step_reducer.finish_step(self._snapshot_fn())

    async def aclose(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
        if self._failure is not None:
            raise self._failure

    async def _run(self) -> None:
        try:
            await self._emit(force=True)
            while True:
                try:
                    await asyncio.wait_for(
                        self._stop.wait(),
                        timeout=self._interval_s,
                    )
                except TimeoutError:
                    await self._emit(force=False)
                    continue
                await self._emit(force=True)
                return
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._failure = error

    async def _emit(self, *, force: bool) -> None:
        snapshot = self._snapshot_fn()
        self._step_reducer.observe(snapshot)
        values = producer_metric_values(snapshot)
        if not force and values == self._last_values:
            return
        metrics: dict[str, Any] = {
            "producer/event": self._next_event,
            "producer/elapsed_time_s": time.monotonic() - self._started_at,
            **values,
        }
        await asyncio.to_thread(self._metrics_fn, metrics)
        self._last_values = values
        self._next_event += 1


class AsyncRLTelemetry:
    """Build public metrics from producer snapshots and completed batches."""

    def __init__(
        self,
        *,
        producer_metrics_fn: ProducerMetricsFn,
        step_metrics_fn: StepMetricsFn,
        post_step_metrics_fn: PostStepMetricsFn | None = None,
    ) -> None:
        self._producer_metrics_fn = producer_metrics_fn
        self._step_metrics_fn = step_metrics_fn
        self._post_step_metrics_fn = post_step_metrics_fn
        self._producer_snapshot_fn: Callable[[], ProducerSnapshot] | None = None
        self._reporter: ProducerMetricsReporter | None = None

    def start(self, snapshot_fn: Callable[[], ProducerSnapshot]) -> None:
        if self._reporter is not None:
            raise RuntimeError("async RL telemetry already started")
        self._producer_snapshot_fn = snapshot_fn
        self._reporter = ProducerMetricsReporter(
            snapshot_fn=snapshot_fn,
            metrics_fn=self._producer_metrics_fn,
        )
        self._reporter.start()

    async def aclose(self) -> None:
        if self._reporter is not None:
            await self._reporter.aclose()

    @property
    def failure(self) -> BaseException | None:
        return None if self._reporter is None else self._reporter.failure

    def snapshot(self) -> ProducerSnapshot:
        if self._producer_snapshot_fn is None:
            raise RuntimeError("async RL telemetry is not started")
        return self._producer_snapshot_fn()

    def finish_step(
        self,
        *,
        batch: OptimizerBatch,
        trained_against_version: int,
        prompt_groups: Sequence[PromptGroup],
        fwd_bwd_results: Sequence[Any],
        optim_result: Any,
        timing_metrics: dict[str, Any],
        weight_sync_time: float | None,
        learning_rate: float,
    ) -> dict[str, Any]:
        reporter = self._reporter
        if reporter is None:
            raise RuntimeError("async RL telemetry is not started")
        snapshot = self.snapshot()
        offsets = [
            trained_against_version - submit_version
            for submit_version in batch.submit_versions
        ]
        train_wall_time = 0.0
        if batch._train_started_at is not None:
            train_wall_time = (
                batch._train_finished_at or time.monotonic()
            ) - batch._train_started_at
        rollout_wait = (
            float(snapshot["rollout_wait_for_trainer_time_total"] or 0.0)
            - batch._rollout_wait_at_train_start
        )
        loop_stats: dict[str, Any] = {
            "async/version_offset_mean": sum(offsets) / len(offsets),
            "async/version_offset_max": max(offsets),
            "async/version_offset_min": min(offsets),
            "async/trained_against_version": trained_against_version,
            "async/realized_training_chunks": batch.realized_chunks,
            "all_raw_rewards": batch.completed_rewards,
            "trainer_wait_for_sampler_time": batch._trainer_wait_for_rollout_time,
            "sampler_wait_for_trainer_time": max(0.0, rollout_wait),
            "perf/trainer_wait_for_chunk_time": batch._trainer_wait_for_chunk_time,
            "perf/train_wall_time": train_wall_time,
            **reporter.finish_step(),
        }
        accumulated_stats = build_accumulated_async_loop_stats(
            latest_loop_stats=loop_stats,
            trainer_wait_for_sampler_time=batch._trainer_wait_for_rollout_time,
            sampler_wait_for_trainer_time=max(0.0, rollout_wait),
            train_wall_time=train_wall_time,
        )
        metrics = compute_step_metrics(
            prompt_groups=prompt_groups,
            fwd_bwd_results=fwd_bwd_results,
            optim_result=optim_result,
            n_accum=len(fwd_bwd_results),
            timing_metrics=timing_metrics,
            loop_stats=accumulated_stats,
        )
        if weight_sync_time is not None:
            metrics["perf/weight_sync_time"] = weight_sync_time
        metrics["train/lr"] = learning_rate
        metrics["rollout/step"] = batch.batch_id
        metrics["train/step"] = batch.batch_id
        logger.info(
            "Rollout %d | Reward raw %.3f filtered %.3f | "
            "Samples raw %d filtered %d | Filter %.1f%% | RefKL %.4f",
            batch.batch_id,
            metrics.get("rollout/raw_reward", 0.0),
            metrics.get("rollout/filtered_reward", 0.0),
            metrics.get("rollout/raw_samples", 0),
            metrics.get("rollout/filtered_samples", 0),
            metrics.get("rollout/filter_ratio", 0.0) * 100,
            metrics.get("train/ref_kl", 0.0),
        )
        self._step_metrics_fn(metrics, batch.batch_id)
        if self._post_step_metrics_fn is not None:
            self._post_step_metrics_fn(metrics)
        return metrics

    def final_stats(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        return {
            **snapshot,
            "total_accepted": int(snapshot["rows_accepted"] or 0),
            "sample_fails": int(snapshot["sample_fails"] or 0),
            "filter_drops": int(snapshot["filter_drops"] or 0),
            "resolved_rows": int(snapshot["resolved_rows"] or 0),
        }


async def run_async_rl_lifecycle(
    run_training: Callable[[PostStepMetricsFn | None], Awaitable[T]],
    *,
    post_step_metrics_fn: PostStepMetricsFn | None = None,
) -> T:
    """Run a recipe-owned lifecycle with an optional step observer."""

    return await run_training(post_step_metrics_fn)


__all__ = [
    "AsyncRLTelemetry",
    "PostStepMetricsFn",
    "ProducerMetricsFn",
    "ProducerMetricsReporter",
    "ProducerStepMetricsReducer",
    "producer_metric_values",
    "run_async_rl_lifecycle",
]
