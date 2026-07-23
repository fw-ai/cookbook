"""Public coordinator joining rollout production and serialized training."""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

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

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PublishResult:
    """Domain result of publishing one trained optimizer batch."""

    resolved_rows: int
    trained_against_version: int


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
        self._active_operation: str | None = None
        self._active_batch: OptimizerBatch | None = None
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
            error_classifier=error_classifier,
            circuit_breaker=circuit_breaker,
        )

    @property
    def global_step(self) -> int:
        return self._producer.published_version

    @property
    def published_version(self) -> int:
        return self._producer.published_version

    @property
    def resolved_rows(self) -> int:
        return self._producer.resolved_rows

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
        tracks_training = optimizer_batch is not None and operation in {
            "train_chunk",
            "optimizer",
        }
        if tracks_training and optimizer_batch is not None:
            if optimizer_batch._train_started_at is None:
                optimizer_batch._train_started_at = time.monotonic()
                optimizer_batch._rollout_wait_at_train_start = float(
                    self._producer.snapshot()[
                        "rollout_wait_for_trainer_time_total"
                    ]
                    or 0.0
                )
        executor = self._executor
        if executor is None:
            raise RuntimeError("trainer executor is unavailable")
        self._active_operation = operation
        call = functools.partial(function, *args, **kwargs)
        worker_future = executor.submit(call)
        try:
            result = await self._await_joined(worker_future)
            if optimizer_batch is not None and operation == "optimizer":
                optimizer_batch._train_finished_at = time.monotonic()
            return result
        finally:
            self._active_operation = None

    def raise_if_failed(self, batch: OptimizerBatch | None = None) -> None:
        failure = self._producer.failure
        if failure is None:
            return
        if batch is None or failure.affects_batch_id <= batch.batch_id:
            raise failure.error

    def publish(self, batch: OptimizerBatch) -> PublishResult:
        """Publish after hotload and commit accepted rows."""

        if self._active_batch is not batch:
            raise RuntimeError(f"batch {batch.batch_id} is not active")
        trained_against_version = self.published_version
        resolved_rows = self._producer.publish(batch)
        self._active_batch = None
        return PublishResult(
            resolved_rows=resolved_rows,
            trained_against_version=trained_against_version,
        )

    def snapshot(self) -> dict[str, int | float | bool | None]:
        """Return an observation-only coordinator snapshot."""

        return self._producer.snapshot()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._producer.aclose()
        finally:
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

    def _require_started(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("async RL coordinator is not running")


__all__ = [
    "AsyncRLCoordinator",
    "PublishResult",
]
