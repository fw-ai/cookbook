"""Optimizer-batch and chunk ownership for asynchronous RL."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Hashable
from dataclasses import dataclass, field

from training.utils.rl.losses import PromptGroup


def balanced_chunk_targets(total_groups: int, requested_chunks: int) -> tuple[int, ...]:
    """Split one optimizer batch into fixed non-empty, balanced chunks."""

    _require_positive_int("total_groups", total_groups)
    _require_positive_int("requested_chunks", requested_chunks)
    chunks = min(total_groups, requested_chunks)
    base, larger = divmod(total_groups, chunks)
    return tuple(base + (index < larger) for index in range(chunks))


@dataclass(frozen=True, slots=True)
class TrainingChunk:
    """One immutable forward/backward unit from an optimizer batch."""

    batch_id: int
    index: int
    planned_chunks: int
    target_groups: int
    groups: tuple[PromptGroup, ...]
    submit_versions: tuple[int, ...]
    source_tokens: tuple[Hashable, ...]

    @property
    def actual_groups(self) -> int:
        return len(self.groups)

    @property
    def partial(self) -> bool:
        return self.actual_groups < self.target_groups


@dataclass(frozen=True, slots=True)
class _BatchFailed:
    error: BaseException


class _BatchFinished:
    pass


_BATCH_FINISHED = _BatchFinished()


@dataclass(slots=True)
class OptimizerBatch:
    """The sole owner of one optimizer batch and its streamed chunks."""

    batch_id: int
    target_groups: int
    chunk_targets: tuple[int, ...]
    _queue: asyncio.Queue[TrainingChunk | _BatchFailed | _BatchFinished] = field(
        default_factory=asyncio.Queue,
        repr=False,
    )
    _pending_groups: list[PromptGroup] = field(default_factory=list, repr=False)
    _pending_versions: list[int] = field(default_factory=list, repr=False)
    _pending_sources: list[Hashable] = field(default_factory=list, repr=False)
    _accepted_sequences: list[int] = field(default_factory=list, repr=False)
    _all_groups: list[PromptGroup] = field(default_factory=list, repr=False)
    _all_versions: list[int] = field(default_factory=list, repr=False)
    _realized_chunks: int = 0
    _exposed: bool = False
    _sealed: bool = False
    _partial: bool = False
    _consumer_started: bool = False
    _consumed: bool = False
    _ready_chunks: int = 0
    _trainer_wait_for_chunk_time: float = 0.0
    _trainer_wait_for_rollout_time: float = 0.0
    _train_started_at: float | None = None
    _train_finished_at: float | None = None
    _counter_start: dict[str, int | float | bool | None] | None = field(
        default=None,
        repr=False,
    )
    _sample_fails: int = 0
    _filter_drops: int = 0
    _max_ready_chunks_during_train: int = 0
    _max_in_flight_during_train: int = 0

    @property
    def planned_chunks(self) -> int:
        return len(self.chunk_targets)

    @property
    def realized_chunks(self) -> int:
        return self._realized_chunks

    @property
    def realized_groups(self) -> int:
        return len(self._all_groups)

    @property
    def partial(self) -> bool:
        return self._partial

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def prompt_groups(self) -> tuple[PromptGroup, ...]:
        return tuple(self._all_groups)

    @property
    def submit_versions(self) -> tuple[int, ...]:
        return tuple(self._all_versions)

    @property
    def ready_chunks(self) -> int:
        return self._ready_chunks

    @property
    def sample_fails(self) -> int:
        return self._sample_fails

    @property
    def filter_drops(self) -> int:
        return self._filter_drops

    @property
    def sampled_rows(self) -> int:
        return self.realized_groups + self._sample_fails + self._filter_drops

    async def chunks(self) -> AsyncIterator[TrainingChunk]:
        """Yield chunks once, waiting only when the next target is not ready."""

        if self._consumer_started:
            raise RuntimeError(
                f"batch {self.batch_id} chunks can only be consumed once"
            )
        self._consumer_started = True
        while True:
            wait_started = time.monotonic()
            item = await self._queue.get()
            self._trainer_wait_for_chunk_time += time.monotonic() - wait_started
            if item is _BATCH_FINISHED:
                self._consumed = True
                return
            if isinstance(item, _BatchFailed):
                raise item.error
            self._ready_chunks -= 1
            yield item

    def _append_group(
        self,
        *,
        sequence: int,
        group: PromptGroup,
        submit_version: int,
        source_token: Hashable,
    ) -> TrainingChunk | None:
        if self._sealed:
            raise RuntimeError(f"batch {self.batch_id} is already sealed")
        self._accepted_sequences.append(sequence)
        self._all_groups.append(group)
        self._all_versions.append(submit_version)
        self._pending_groups.append(group)
        self._pending_versions.append(submit_version)
        self._pending_sources.append(source_token)
        target = self.chunk_targets[self._realized_chunks]
        if len(self._pending_groups) < target:
            return None
        return self._take_chunk(target)

    def _take_partial_chunk(self) -> TrainingChunk | None:
        if not self._pending_groups:
            return None
        return self._take_chunk(self.chunk_targets[self._realized_chunks])

    def _take_chunk(self, target_groups: int) -> TrainingChunk:
        chunk = TrainingChunk(
            batch_id=self.batch_id,
            index=self._realized_chunks,
            planned_chunks=self.planned_chunks,
            target_groups=target_groups,
            groups=tuple(self._pending_groups),
            submit_versions=tuple(self._pending_versions),
            source_tokens=tuple(self._pending_sources),
        )
        self._pending_groups.clear()
        self._pending_versions.clear()
        self._pending_sources.clear()
        self._realized_chunks += 1
        return chunk

    def _put_chunk(self, chunk: TrainingChunk) -> None:
        self._queue.put_nowait(chunk)
        self._ready_chunks += 1

    def _record_rejections(self, *, sample_fails: int, filter_drops: int) -> None:
        if sample_fails < 0 or filter_drops < 0:
            raise ValueError("rejection counts must be nonnegative")
        self._sample_fails += sample_fails
        self._filter_drops += filter_drops

    def _seal(self) -> None:
        if self._sealed:
            raise RuntimeError(f"batch {self.batch_id} sealed twice")
        self._sealed = True
        self._partial = self.realized_groups < self.target_groups
        self._queue.put_nowait(_BATCH_FINISHED)

    def _fail(self, error: BaseException) -> None:
        if not self._sealed:
            self._queue.put_nowait(_BatchFailed(error))


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


__all__ = [
    "OptimizerBatch",
    "TrainingChunk",
    "balanced_chunk_targets",
]
