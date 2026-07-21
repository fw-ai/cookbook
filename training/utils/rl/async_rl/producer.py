"""Completion-driven rollout production and off-policy admission."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias

from training.train_loop import DynamicFilterFn
from training.utils.rl.async_rl.batch import (
    OptimizerBatch,
    TrainingChunk,
    balanced_chunk_targets,
)
from training.utils.rl.async_rl.errors import (
    CircuitBreakerConfig,
    CircuitBreakerTripped,
    ErrorDisposition,
    RecoverableCircuitBreaker,
    RolloutErrorClassifier,
)
from training.utils.rl.rollout.group_assembler import (
    AdvantageFn,
    GroupAssembler,
    RowResolution,
)
from training.utils.rl.rollout.types import RolloutRun

logger = logging.getLogger(__name__)

RunFactory: TypeAlias = Callable[[int], Awaitable[RolloutRun | None]]


@dataclass(slots=True)
class RolloutRow:
    """Scheduler envelope for one row admitted and resolved atomically.

    This is not the user rollout-function contract. ``run_factory`` is the
    recipe-owned closure that binds a dataset row to one rollout call.
    """

    row_id: Hashable
    run_factory: RunFactory
    row_meta: dict[str, Any] | None = None
    on_resolved: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        try:
            hash(self.row_id)
        except TypeError as error:
            raise TypeError("row_id must be hashable") from error
        if not callable(self.run_factory):
            raise TypeError("run_factory must be callable")
        if self.row_meta is not None and not isinstance(self.row_meta, dict):
            raise TypeError("row_meta must be a dict or None")
        if self.on_resolved is not None and not callable(self.on_resolved):
            raise TypeError("on_resolved must be callable or None")


@dataclass(frozen=True, slots=True)
class _ProducerFailed:
    error: BaseException
    affects_batch_id: int


class _ProducerFinished:
    pass


_PRODUCER_FINISHED = _ProducerFinished()


@dataclass(slots=True)
class _CursorEntry:
    request: RolloutRow
    durable_reason: str | None = None
    batch_id: int | None = None


@dataclass(frozen=True, slots=True)
class _Draw:
    sequence: int
    sub_index: int
    submission_index: int


class RolloutProducer:
    """Completion-driven prompt-group producer with exact version admission."""

    def __init__(
        self,
        *,
        rows: Iterable[RolloutRow],
        output: asyncio.Queue[OptimizerBatch | _ProducerFailed | _ProducerFinished],
        completions_per_prompt: int,
        prompt_groups_per_step: int,
        training_chunks_per_step: int,
        max_head_off_policy_versions: int,
        max_concurrent_rollouts: int | None,
        advantage_fn: AdvantageFn,
        with_reference: bool,
        router_replay_completion_only: bool,
        min_group_size: int,
        dynamic_filter_fn: DynamicFilterFn | None,
        initial_version: int,
        resolved_rows_offset: int,
        resolved_rows_fn: Callable[[], int] | None,
        trainer_state: Callable[[], tuple[bool, int | None]],
        error_classifier: RolloutErrorClassifier,
        circuit_breaker: CircuitBreakerConfig | None,
    ) -> None:
        _validate_producer_args(
            completions_per_prompt=completions_per_prompt,
            prompt_groups_per_step=prompt_groups_per_step,
            training_chunks_per_step=training_chunks_per_step,
            max_head_off_policy_versions=max_head_off_policy_versions,
            max_concurrent_rollouts=max_concurrent_rollouts,
            min_group_size=min_group_size,
            initial_version=initial_version,
        )
        self._rows = iter(rows)
        self._output = output
        self._cpp = completions_per_prompt
        self._groups_per_batch = prompt_groups_per_step
        self._samples_per_batch = completions_per_prompt * prompt_groups_per_step
        self._chunk_targets = balanced_chunk_targets(
            prompt_groups_per_step,
            training_chunks_per_step,
        )
        self._max_staleness = max_head_off_policy_versions
        self._max_concurrent = max_concurrent_rollouts
        self._dynamic_filter_fn = dynamic_filter_fn
        self._published_version = initial_version
        self._accepted_samples_offset = initial_version * self._samples_per_batch
        self._resolved_rows_offset = resolved_rows_offset
        self._resolved_rows_fn = resolved_rows_fn
        self._trainer_state = trainer_state
        self._error_classifier = error_classifier
        self._breaker = RecoverableCircuitBreaker(circuit_breaker)
        self._assembler = GroupAssembler(
            completions_per_prompt=completions_per_prompt,
            advantage_fn=advantage_fn,
            with_reference=with_reference,
            router_replay_completion_only=router_replay_completion_only,
            min_group_size=min_group_size,
        )

        self._in_flight: dict[asyncio.Task[RolloutRun | None], _Draw] = {}
        self._cursor: dict[int, _CursorEntry] = {}
        self._batches: dict[int, OptimizerBatch] = {}
        self._fill_batch: OptimizerBatch | None = None
        self._prefetched: RolloutRow | None = None
        self._next_sequence = 0
        self._cursor_flushed = 0
        self._next_submission_index = 0
        self._next_batch_id = initial_version + 1
        self._next_publish_batch_id = initial_version + 1
        self._accepted_samples = 0
        self._reserved_samples = 0
        self._pending_sample_fails = 0
        self._pending_filter_drops = 0
        self._max_in_flight = 0
        self._source_exhausted = False
        self._finished = False
        self._closing = False
        self._failure: _ProducerFailed | None = None
        self._driver_wake = asyncio.Event()
        self._driver_task: asyncio.Task[None] | None = None
        self._rollout_wait_for_trainer_started: float | None = None
        self._rollout_wait_for_trainer_total = 0.0
        self._counters: dict[str, int] = {
            "rows_submitted": 0,
            "rows_accepted": 0,
            "rows_rejected": 0,
            "completion_refills_during_train": 0,
            "sample_fails": 0,
            "filter_drops": 0,
            "recoverable_errors": 0,
        }

    @property
    def published_version(self) -> int:
        return self._published_version

    @property
    def failure(self) -> _ProducerFailed | None:
        return self._failure

    @property
    def resolved_rows(self) -> int:
        if self._resolved_rows_fn is not None:
            return int(self._resolved_rows_fn())
        return self._resolved_rows_offset + self._cursor_flushed

    def start(self) -> None:
        if self._driver_task is not None:
            raise RuntimeError("rollout producer already started")
        self._driver_task = asyncio.create_task(
            self._drive(),
            name="async-rl-rollout-producer",
        )

    async def aclose(self) -> None:
        if self._closing:
            if self._driver_task is not None:
                await asyncio.gather(self._driver_task, return_exceptions=True)
            return
        self._closing = True
        self._driver_wake.set()
        if self._driver_task is not None and not self._driver_task.done():
            self._driver_task.cancel()
        if self._driver_task is not None:
            await asyncio.gather(self._driver_task, return_exceptions=True)

    def publish(self, batch: OptimizerBatch) -> int:
        """Publish one hotloaded batch and make its accepted rows durable."""

        if batch.batch_id != self._next_publish_batch_id:
            raise RuntimeError(
                "batch publication must be ordered: "
                f"expected {self._next_publish_batch_id}, got {batch.batch_id}"
            )
        if self._batches.get(batch.batch_id) is not batch:
            raise RuntimeError(f"batch {batch.batch_id} is not owned by this producer")
        if not batch.sealed or not batch._consumed:
            raise RuntimeError(
                f"batch {batch.batch_id} must be fully consumed before publication"
            )
        for sequence in batch._accepted_sequences:
            entry = self._cursor.get(sequence)
            if entry is None or entry.batch_id != batch.batch_id:
                raise RuntimeError(
                    f"accepted row {sequence} is not assigned to batch {batch.batch_id}"
                )
            entry.durable_reason = "accepted"
        self._flush_cursor()
        self._published_version = batch.batch_id
        self._next_publish_batch_id += 1
        del self._batches[batch.batch_id]
        self._mark_rollout_wait_for_trainer_end()
        self._driver_wake.set()
        return self.resolved_rows

    def snapshot(self) -> dict[str, int | float | bool | None]:
        concurrency = self._concurrency_capacity()
        snapshot: dict[str, int | float | bool | None] = {
            "published_version": self._published_version,
            "accepted_samples": self._accepted_samples,
            "reserved_samples": self._reserved_samples,
            "in_flight_samples": len(self._in_flight),
            "max_in_flight_samples": self._max_in_flight,
            "staleness_capacity": max(0, self._staleness_capacity()),
            "concurrency_capacity": (
                None if concurrency is None else max(0, concurrency)
            ),
            "source_exhausted": self._source_exhausted,
            "resolved_rows": self.resolved_rows,
            "rollout_wait_for_trainer_time_total": self._rollout_wait_for_trainer_total
            + self._active_rollout_wait_for_trainer(),
            "fatal": self._failure is not None,
        }
        snapshot.update(self._counters)
        breaker = self._breaker.snapshot
        snapshot.update(
            {
                "breaker/consecutive_failures": breaker.consecutive_failures,
                "breaker/failures_in_window": breaker.failures_in_window,
                "breaker/observations_in_window": breaker.observations_in_window,
                "breaker/failure_rate": breaker.failure_rate,
                "breaker/tripped": breaker.tripped,
            }
        )
        return snapshot

    async def _drive(self) -> None:
        try:
            self._refill(triggered_by_completion=False)
            while not self._closing and self._failure is None:
                self._finish_source_if_drained()
                if self._finished:
                    return
                if self._in_flight:
                    self._driver_wake.clear()
                    self._refill(triggered_by_completion=False)
                    wake_task = asyncio.create_task(self._driver_wake.wait())
                    try:
                        done, _ = await asyncio.wait(
                            (*self._in_flight, wake_task),
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        if not wake_task.done():
                            wake_task.cancel()
                        await asyncio.gather(wake_task, return_exceptions=True)
                    draws = [task for task in done if task is not wake_task]
                    for task in sorted(
                        draws,
                        key=lambda current: self._in_flight[current].submission_index,
                    ):
                        self._retire(task)
                        if self._failure is not None:
                            break
                        self._refill(triggered_by_completion=True)
                    continue

                self._refill(triggered_by_completion=False)
                self._finish_source_if_drained()
                if self._finished or self._failure is not None:
                    return
                if self._in_flight:
                    continue
                if self._waiting_for_publish():
                    self._mark_rollout_wait_for_trainer_start()
                    self._driver_wake.clear()
                    if self._admission_capacity() >= self._cpp:
                        self._mark_rollout_wait_for_trainer_end()
                        continue
                    await self._driver_wake.wait()
                    self._mark_rollout_wait_for_trainer_end()
                    continue
                raise RuntimeError(self._stall_message())
        except asyncio.CancelledError:
            if not self._closing:
                self._fail(RuntimeError("rollout producer was unexpectedly cancelled"))
            raise
        except Exception as error:
            self._fail(error)
        finally:
            await self._cancel_and_join_draws()
            self._mark_rollout_wait_for_trainer_end()

    async def _invoke(self, request: RolloutRow, sub_index: int) -> RolloutRun | None:
        return await request.run_factory(sub_index)

    def _refill(self, *, triggered_by_completion: bool) -> None:
        if self._closing or self._failure is not None or self._source_exhausted:
            self._mark_rollout_wait_for_trainer_end()
            return
        submitted = 0
        while self._ensure_prefetched():
            if self._admission_capacity() < self._cpp:
                break
            request = self._prefetched
            if request is None:
                raise RuntimeError("prefetched row disappeared")
            self._prefetched = None
            self._submit_row(request)
            submitted += 1
        if triggered_by_completion and submitted:
            active, _ = self._trainer_state()
            if active:
                self._counters["completion_refills_during_train"] += 1
        self._record_refill_stop()

    def _record_refill_stop(self) -> None:
        if self._prefetched is None:
            self._mark_rollout_wait_for_trainer_end()
            return
        if self._staleness_capacity() < self._cpp:
            concurrency = self._concurrency_capacity()
            if concurrency is None or concurrency >= self._cpp:
                self._mark_rollout_wait_for_trainer_start()
            return
        self._mark_rollout_wait_for_trainer_end()
        concurrency = self._concurrency_capacity()
        if concurrency is not None and concurrency < self._cpp:
            return
        raise RuntimeError("rollout refill stopped without a limiting budget")

    def _ensure_prefetched(self) -> bool:
        if self._prefetched is not None:
            return True
        if self._source_exhausted:
            return False
        try:
            request = next(self._rows)
        except StopIteration:
            self._source_exhausted = True
            return False
        if not isinstance(request, RolloutRow):
            raise TypeError(
                "rollout row source must yield RolloutRow instances, got "
                f"{type(request).__name__}"
            )
        self._prefetched = request
        return True

    def _submit_row(self, request: RolloutRow) -> None:
        if self._admission_capacity() < self._cpp:
            raise RuntimeError("rollout row submitted without whole-row capacity")
        sequence = self._next_sequence
        self._next_sequence += 1
        self._cursor[sequence] = _CursorEntry(request=request)
        self._reserved_samples += self._cpp
        self._counters["rows_submitted"] += 1
        for sub_index in range(self._cpp):
            self._assembler.note_started(
                sequence,
                submit_version=self._published_version,
                row_meta=request.row_meta if sub_index == 0 else None,
            )
            task = asyncio.create_task(
                self._invoke(request, sub_index),
                name=f"async-rl-rollout-{sequence}-{sub_index}",
            )
            self._in_flight[task] = _Draw(
                sequence=sequence,
                sub_index=sub_index,
                submission_index=self._next_submission_index,
            )
            self._next_submission_index += 1
        self._max_in_flight = max(self._max_in_flight, len(self._in_flight))
        self._note_active_batch_in_flight()

    def _retire(self, task: asyncio.Task[RolloutRun | None]) -> None:
        draw = self._in_flight.pop(task)
        if task.cancelled():
            self._fail(RuntimeError("rollout task was unexpectedly cancelled"))
            return
        error = task.exception()
        if error is not None:
            classification = self._error_classifier(error)
            if classification.disposition is ErrorDisposition.FATAL:
                self._fail(error)
                return
            self._counters["recoverable_errors"] += 1
            try:
                self._breaker.record_failure(error, classification)
            except CircuitBreakerTripped as breaker_error:
                self._fail(breaker_error)
                return
            if self._counters["recoverable_errors"] == 1 or (
                self._counters["recoverable_errors"] % 10 == 0
            ):
                logger.warning(
                    "Async RL recoverable rollout errors: total=%d reason=%s",
                    self._counters["recoverable_errors"],
                    classification.reason,
                )
            resolution = self._assembler.note_dropped(draw.sequence)
        else:
            run = task.result()
            self._breaker.record_success()
            if run is None:
                resolution = self._assembler.note_dropped(draw.sequence)
            elif isinstance(run, RolloutRun):
                resolution = self._assembler.add_run(draw.sequence, run)
            else:
                error = TypeError(
                    "rollout function must return RolloutRun or None, got "
                    f"{type(run).__name__}"
                )
                self._fail(error)
                return
        if resolution is not None:
            self._resolve_row(draw.sequence, resolution)

    def _resolve_row(self, sequence: int, resolution: RowResolution) -> None:
        entry = self._cursor.get(sequence)
        if entry is None:
            raise RuntimeError(f"row sequence {sequence} is missing")
        self._reserved_samples -= self._cpp
        if resolution.pg is None:
            self._reject_row(entry, reason="none")
            return
        if self._dynamic_filter_fn is not None and not self._dynamic_filter_fn(
            resolution.pg
        ):
            self._reject_row(entry, reason="filter")
            return

        self._accepted_samples += self._cpp
        self._counters["rows_accepted"] += 1
        batch = self._batch_for_next_group()
        entry.batch_id = batch.batch_id
        source_token = entry.request.row_id
        chunk = batch._append_group(
            sequence=sequence,
            group=resolution.pg,
            submit_version=resolution.min_submit_version,
            source_token=source_token,
        )
        if chunk is not None:
            self._emit_chunk(batch, chunk)
        if batch.realized_groups == batch.target_groups:
            self._seal_fill_batch()

    def _reject_row(self, entry: _CursorEntry, *, reason: str) -> None:
        if reason == "none":
            self._counters["sample_fails"] += 1
            sample_fails, filter_drops = 1, 0
        elif reason == "filter":
            self._counters["filter_drops"] += 1
            sample_fails, filter_drops = 0, 1
        else:
            raise ValueError(f"unknown rollout rejection reason: {reason}")
        if self._fill_batch is None:
            self._pending_sample_fails += sample_fails
            self._pending_filter_drops += filter_drops
        else:
            self._fill_batch._record_rejections(
                sample_fails=sample_fails,
                filter_drops=filter_drops,
            )
        entry.durable_reason = reason
        self._counters["rows_rejected"] += 1
        self._flush_cursor()

    def _batch_for_next_group(self) -> OptimizerBatch:
        if self._fill_batch is None:
            batch = OptimizerBatch(
                batch_id=self._next_batch_id,
                target_groups=self._groups_per_batch,
                chunk_targets=self._chunk_targets,
            )
            self._next_batch_id += 1
            self._fill_batch = batch
            self._batches[batch.batch_id] = batch
            batch._record_rejections(
                sample_fails=self._pending_sample_fails,
                filter_drops=self._pending_filter_drops,
            )
            self._pending_sample_fails = 0
            self._pending_filter_drops = 0
        return self._fill_batch

    def _emit_chunk(self, batch: OptimizerBatch, chunk: TrainingChunk) -> None:
        batch._put_chunk(chunk)
        if not batch._exposed:
            batch._exposed = True
            self._output.put_nowait(batch)
        active, active_batch_id = self._trainer_state()
        if active:
            if active_batch_id == batch.batch_id:
                batch._max_ready_chunks_during_train = max(
                    batch._max_ready_chunks_during_train,
                    batch.ready_chunks,
                )

    def _seal_fill_batch(self) -> None:
        batch = self._fill_batch
        if batch is None:
            return
        chunk = batch._take_partial_chunk()
        if chunk is not None:
            self._emit_chunk(batch, chunk)
        batch._seal()
        self._fill_batch = None

    def _finish_source_if_drained(self) -> None:
        if not self._source_exhausted or self._in_flight or self._finished:
            return
        self._seal_fill_batch()
        self._finished = True
        self._output.put_nowait(_PRODUCER_FINISHED)

    def _flush_cursor(self) -> None:
        while True:
            entry = self._cursor.get(self._cursor_flushed)
            if entry is None or entry.durable_reason is None:
                return
            if entry.request.on_resolved is not None:
                entry.request.on_resolved(entry.durable_reason)
            del self._cursor[self._cursor_flushed]
            self._cursor_flushed += 1

    def _fail(self, error: BaseException) -> None:
        if self._failure is not None or self._closing:
            return
        affects_batch_id = (
            self._fill_batch.batch_id
            if self._fill_batch is not None
            else self._next_batch_id
        )
        self._failure = _ProducerFailed(error, affects_batch_id)
        if self._fill_batch is not None and self._fill_batch._exposed:
            self._fill_batch._fail(error)
        self._output.put_nowait(self._failure)
        logger.error(
            "Async RL rollout producer failed: batch=%d error_type=%s",
            affects_batch_id,
            type(error).__name__,
        )
        self._driver_wake.set()

    async def _cancel_and_join_draws(self) -> None:
        pending = tuple(self._in_flight)
        self._in_flight.clear()
        for task in pending:
            if not task.done():
                task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def _staleness_capacity(self) -> int:
        accounted = (
            self._accepted_samples_offset
            + self._accepted_samples
            + self._reserved_samples
        )
        return (
            self._published_version + self._max_staleness + 1
        ) * self._samples_per_batch - accounted

    def _concurrency_capacity(self) -> int | None:
        if self._max_concurrent is None:
            return None
        return self._max_concurrent - len(self._in_flight)

    def _admission_capacity(self) -> int:
        staleness = self._staleness_capacity()
        concurrency = self._concurrency_capacity()
        return max(0, staleness if concurrency is None else min(staleness, concurrency))

    def _waiting_for_publish(self) -> bool:
        return any(batch.sealed for batch in self._batches.values())

    def _stall_message(self) -> str:
        return (
            "Async rollout producer stalled with source remaining and no in-flight "
            f"work: accepted={self._accepted_samples}, reserved={self._reserved_samples}, "
            f"staleness_capacity={self._staleness_capacity()}, "
            f"concurrency_capacity={self._concurrency_capacity()}, "
            f"published_version={self._published_version}, "
            f"source_prefetched={self._prefetched is not None}"
        )

    def _mark_rollout_wait_for_trainer_start(self) -> None:
        if self._rollout_wait_for_trainer_started is None:
            self._rollout_wait_for_trainer_started = time.monotonic()

    def _mark_rollout_wait_for_trainer_end(self) -> None:
        if self._rollout_wait_for_trainer_started is not None:
            self._rollout_wait_for_trainer_total += (
                time.monotonic() - self._rollout_wait_for_trainer_started
            )
            self._rollout_wait_for_trainer_started = None

    def _active_rollout_wait_for_trainer(self) -> float:
        if self._rollout_wait_for_trainer_started is None:
            return 0.0
        return time.monotonic() - self._rollout_wait_for_trainer_started

    def _note_active_batch_in_flight(self) -> None:
        active, batch_id = self._trainer_state()
        if not active or batch_id is None:
            return
        batch = self._batches.get(batch_id)
        if batch is not None:
            batch._max_in_flight_during_train = max(
                batch._max_in_flight_during_train,
                len(self._in_flight),
            )


def _validate_producer_args(
    *,
    completions_per_prompt: int,
    prompt_groups_per_step: int,
    training_chunks_per_step: int,
    max_head_off_policy_versions: int,
    max_concurrent_rollouts: int | None,
    min_group_size: int,
    initial_version: int,
) -> None:
    _require_positive_int("completions_per_prompt", completions_per_prompt)
    _require_positive_int("prompt_groups_per_step", prompt_groups_per_step)
    _require_positive_int("training_chunks_per_step", training_chunks_per_step)
    _require_nonnegative_int(
        "max_head_off_policy_versions",
        max_head_off_policy_versions,
    )
    _require_nonnegative_int("global_step", initial_version)
    if max_concurrent_rollouts is not None:
        _require_positive_int(
            "max_concurrent_rollouts",
            max_concurrent_rollouts,
        )
    if max_concurrent_rollouts is not None and (
        max_concurrent_rollouts < completions_per_prompt
    ):
        raise ValueError(
            "max_concurrent_rollouts must fit one whole row: "
            f"{max_concurrent_rollouts} < {completions_per_prompt}"
        )
    _require_positive_int("min_group_size", min_group_size)
    if min_group_size > completions_per_prompt:
        raise ValueError("min_group_size must be <= completions_per_prompt")


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _require_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


__all__ = [
    "RolloutProducer",
    "RolloutRow",
]
