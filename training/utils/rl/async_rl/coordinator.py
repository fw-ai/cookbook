"""Public coordinator joining rollout production and serialized training."""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeAlias, TypeVar

from training.train_loop import DynamicFilterFn
from training.utils.data import compute_advantages
from training.utils.rl.async_rl.batch import OptimizerBatch
from training.utils.rl.async_rl.errors import (
    CircuitBreakerConfig,
    RolloutErrorClassifier,
    classify_rollout_error,
)
from training.utils.rl.async_rl.producer import (
    RolloutProducer,
    RolloutRow,
    _ProducerFailed,
    _ProducerFinished,
    _PRODUCER_FINISHED,
)
from training.utils.rl.rollout.group_assembler import AdvantageFn

PostStepMetricsFn: TypeAlias = Callable[[dict[str, Any]], None]
T = TypeVar("T")
logger = logging.getLogger(__name__)


class AsyncRLCoordinator:
    """Small coordinator joining the producer, batch queue, and trainer thread."""

    def __init__(
        self,
        *,
        rows: Iterable[RolloutRow],
        completions_per_prompt: int,
        prompt_groups_per_step: int,
        training_chunks_per_step: int,
        max_head_off_policy_versions: int,
        max_concurrent_rollouts: int | None = None,
        advantage_fn: AdvantageFn = compute_advantages,
        with_reference: bool = False,
        router_replay_completion_only: bool = False,
        min_group_size: int = 1,
        dynamic_filter_fn: DynamicFilterFn | None = None,
        global_step: int = 0,
        resolved_rows_offset: int = 0,
        resolved_rows_fn: Callable[[], int] | None = None,
        error_classifier: RolloutErrorClassifier = classify_rollout_error,
        circuit_breaker: CircuitBreakerConfig | None = None,
    ) -> None:
        self._output: asyncio.Queue[
            OptimizerBatch | _ProducerFailed | _ProducerFinished
        ] = asyncio.Queue()
        self._trainer_active = False
        self._optimizer_batch_id: int | None = None
        self._active_operation: str | None = None
        self._active_batch: OptimizerBatch | None = None
        self._completions_per_prompt = completions_per_prompt
        self._started = False
        self._closed = False
        self._executor: ThreadPoolExecutor | None = None
        self._producer = RolloutProducer(
            rows=rows,
            output=self._output,
            completions_per_prompt=completions_per_prompt,
            prompt_groups_per_step=prompt_groups_per_step,
            training_chunks_per_step=training_chunks_per_step,
            max_head_off_policy_versions=max_head_off_policy_versions,
            max_concurrent_rollouts=max_concurrent_rollouts,
            advantage_fn=advantage_fn,
            with_reference=with_reference,
            router_replay_completion_only=router_replay_completion_only,
            min_group_size=min_group_size,
            dynamic_filter_fn=dynamic_filter_fn,
            initial_version=global_step,
            resolved_rows_offset=resolved_rows_offset,
            resolved_rows_fn=resolved_rows_fn,
            trainer_state=self._trainer_state,
            error_classifier=error_classifier,
            circuit_breaker=circuit_breaker,
        )

    @property
    def global_step(self) -> int:
        return self._producer.published_version

    @property
    def published_version(self) -> int:
        return self._producer.published_version

    def start(self) -> None:
        if self._started:
            raise RuntimeError("async RL coordinator already started")
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="async-rl-trainer",
        )
        self._producer.start()
        self._started = True

    async def next_batch(self) -> OptimizerBatch | None:
        self._require_started()
        if self._active_batch is not None:
            raise RuntimeError(
                f"batch {self._active_batch.batch_id} must publish before next_batch"
            )
        wait_started = time.monotonic()
        item = await self._output.get()
        wait_time = time.monotonic() - wait_started
        if item is _PRODUCER_FINISHED:
            return None
        if isinstance(item, _ProducerFailed):
            raise item.error
        item._trainer_wait_for_rollout_time = wait_time
        self._active_batch = item
        return item

    async def run_blocking(
        self,
        operation: str,
        function: Callable[..., T],
        *args: Any,
        optimizer_batch: OptimizerBatch | None = None,
        **kwargs: Any,
    ) -> T:
        """Run one native blocking call without blocking rollout production."""

        self._require_started()
        if self._active_operation is not None:
            raise RuntimeError(
                f"trainer operation {self._active_operation!r} is already active"
            )
        if optimizer_batch is not None:
            if self._active_batch is not optimizer_batch:
                raise RuntimeError(
                    f"batch {optimizer_batch.batch_id} is not the active trainer batch"
                )
            if optimizer_batch._train_started_at is None:
                optimizer_batch._train_started_at = time.monotonic()
                optimizer_batch._counter_start = self._producer.snapshot()
                optimizer_batch._max_ready_chunks_during_train = max(
                    optimizer_batch._max_ready_chunks_during_train,
                    optimizer_batch.ready_chunks,
                )
                optimizer_batch._max_in_flight_during_train = int(
                    self._producer.snapshot()["in_flight_samples"] or 0
                )
        executor = self._executor
        if executor is None:
            raise RuntimeError("trainer executor is unavailable")
        self._active_operation = operation
        self._trainer_active = True
        self._optimizer_batch_id = (
            None if optimizer_batch is None else optimizer_batch.batch_id
        )
        call = functools.partial(function, *args, **kwargs)
        worker_future = executor.submit(call)
        try:
            result = await self._await_joined(worker_future)
            if optimizer_batch is not None and operation == "optimizer":
                optimizer_batch._train_finished_at = time.monotonic()
            return result
        finally:
            if optimizer_batch is not None:
                optimizer_batch._max_ready_chunks_during_train = max(
                    optimizer_batch._max_ready_chunks_during_train,
                    optimizer_batch.ready_chunks,
                )
            self._trainer_active = False
            self._optimizer_batch_id = None
            self._active_operation = None

    def raise_if_failed(self, batch: OptimizerBatch | None = None) -> None:
        failure = self._producer.failure
        if failure is None:
            return
        if batch is None or failure.affects_batch_id <= batch.batch_id:
            raise failure.error

    def publish(self, batch: OptimizerBatch) -> dict[str, Any]:
        """Publish after hotload, commit accepted rows, and return batch metrics."""

        if self._active_batch is not batch:
            raise RuntimeError(f"batch {batch.batch_id} is not active")
        trained_against_version = self.published_version
        resolved_rows = self._producer.publish(batch)
        self._active_batch = None
        metrics = self._batch_metrics(
            batch,
            trained_against_version=trained_against_version,
        )
        metrics["resolved_rows"] = resolved_rows
        return metrics

    def snapshot(self) -> dict[str, int | float | bool | None]:
        """Return an observation-only coordinator snapshot."""

        return self._producer.snapshot()

    def final_stats(self) -> dict[str, Any]:
        snapshot = self._producer.snapshot()
        return {
            **snapshot,
            "total_accepted": int(snapshot["accepted_samples"] or 0)
            // self._completions_per_prompt,
            "sample_fails": int(snapshot["sample_fails"] or 0),
            "filter_drops": int(snapshot["filter_drops"] or 0),
            "resolved_rows": self._producer.resolved_rows,
        }

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._producer.aclose()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    async def __aenter__(self) -> AsyncRLCoordinator:
        self.start()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def _await_joined(self, worker_future: Future[T]) -> T:
        async_future = asyncio.wrap_future(worker_future)
        try:
            return await asyncio.shield(async_future)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(async_future)
            except BaseException:
                pass
            raise

    def _trainer_state(self) -> tuple[bool, int | None]:
        return self._trainer_active, self._optimizer_batch_id

    def _batch_metrics(
        self,
        batch: OptimizerBatch,
        *,
        trained_against_version: int,
    ) -> dict[str, Any]:
        snapshot = self._producer.snapshot()
        start = batch._counter_start or snapshot

        versions = batch.submit_versions
        offsets = [trained_against_version - version for version in versions]
        train_wall_time = 0.0
        if batch._train_started_at is not None:
            train_finished_at = (
                batch._train_finished_at
                if batch._train_finished_at is not None
                else time.monotonic()
            )
            train_wall_time = train_finished_at - batch._train_started_at
        rollout_wait = float(
            snapshot["rollout_wait_for_trainer_time_total"] or 0.0
        ) - float(start["rollout_wait_for_trainer_time_total"] or 0.0)
        groups = batch.prompt_groups
        return {
            "async/version_offset_mean": sum(offsets) / len(offsets),
            "async/version_offset_max": max(offsets),
            "async/version_offset_min": min(offsets),
            "async/in_flight": int(snapshot["in_flight_samples"] or 0),
            "async/max_in_flight_during_train": batch._max_in_flight_during_train,
            "async/completion_refills_during_train": (
                int(snapshot["completion_refills_during_train"] or 0)
                - int(start["completion_refills_during_train"] or 0)
            ),
            "async/running_samples": int(snapshot["reserved_samples"] or 0),
            "async/accepted_samples": int(snapshot["accepted_samples"] or 0),
            "async/staleness_capacity_at_step": int(
                snapshot["staleness_capacity"] or 0
            ),
            "async/concurrency_capacity_at_step": (
                -1
                if snapshot["concurrency_capacity"] is None
                else int(snapshot["concurrency_capacity"] or 0)
            ),
            "ctx/current_version": trained_against_version,
            "pipeline/chunks_per_step": batch.realized_chunks,
            "pipeline/prompt_groups_per_step": batch.target_groups,
            "async/max_ready_training_chunks_during_train": (
                batch._max_ready_chunks_during_train
            ),
            "batch/optimizer_prompt_groups": batch.realized_groups,
            "valid_prompt_groups": batch.realized_groups,
            "all_raw_rewards": [reward for group in groups for reward in group.rewards],
            "total_sampled": batch.sampled_rows * self._completions_per_prompt,
            "filter_drops": batch.filter_drops,
            "sample_fails": batch.sample_fails,
            # Preserve the existing metric names for dashboard compatibility.
            "trainer_wait_for_sampler_time": batch._trainer_wait_for_rollout_time,
            "sampler_wait_for_trainer_time": max(0.0, rollout_wait),
            "perf/trainer_wait_for_chunk_time": batch._trainer_wait_for_chunk_time,
            "perf/train_wall_time": train_wall_time,
        }

    def _require_started(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("async RL coordinator is not running")


async def run_async_rl_lifecycle(
    run_training: Callable[[PostStepMetricsFn | None], Awaitable[T]],
    *,
    post_step_metrics_fn: PostStepMetricsFn | None = None,
) -> T:
    """Run the recipe-owned async training lifecycle."""

    return await run_training(post_step_metrics_fn)


__all__ = [
    "AsyncRLCoordinator",
    "PostStepMetricsFn",
    "run_async_rl_lifecycle",
]
