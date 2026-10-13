"""Optimizer-batch and group ownership for asynchronous RL."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Hashable
from dataclasses import dataclass, field

from training.utils.rl.losses import PromptGroup


@dataclass(frozen=True, slots=True)
class TrainingGroup:
    """One immutable forward/backward unit from an optimizer batch."""

    batch_id: int
    index: int
    group: PromptGroup
    submit_version: int
    source_token: Hashable
    ready_at: float = field(default_factory=time.monotonic)
    ready_utc: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class _BatchFailed:
    error: BaseException


class _BatchFinished:
    pass


_BATCH_FINISHED = _BatchFinished()


@dataclass(slots=True)
class OptimizerBatch:
    """The sole owner of one optimizer batch and its streamed groups."""

    batch_id: int
    target_groups: int
    _queue: asyncio.Queue[TrainingGroup | _BatchFailed | _BatchFinished] = field(
        default_factory=asyncio.Queue,
        repr=False,
    )
    _accepted_sequences: list[int] = field(default_factory=list, repr=False)
    _all_groups: list[PromptGroup] = field(default_factory=list, repr=False)
    _all_versions: list[int] = field(default_factory=list, repr=False)
    _completed_rewards: list[float] = field(default_factory=list, repr=False)
    _exposed: bool = False
    _sealed: bool = False
    _partial: bool = False
    _consumer_started: bool = False
    _consumed: bool = False
    _ready_groups: int = 0
    _train_group_wait_time: float = 0.0
    _train_started_at: float | None = None
    _train_finished_at: float | None = None
    _group_wait_at_train_start: float = 0.0

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
    def completed_rewards(self) -> tuple[float, ...]:
        """Rewards from every assembled group before dynamic filtering."""

        return tuple(self._completed_rewards)

    @property
    def ready_groups(self) -> int:
        return self._ready_groups

    def start_training(self, *, at: float | None = None) -> None:
        """Record the first submitted training request and its group-wait boundary."""
        if self._train_started_at is None:
            self._train_started_at = time.monotonic() if at is None else at
            self._group_wait_at_train_start = self._train_group_wait_time

    async def groups(self) -> AsyncIterator[TrainingGroup]:
        """Yield groups once, waiting only when the next target is not ready."""

        if self._consumer_started:
            raise RuntimeError(f"batch {self.batch_id} groups can only be consumed once")
        self._consumer_started = True
        while True:
            wait_started = time.monotonic()
            item = await self._queue.get()
            self._train_group_wait_time += time.monotonic() - wait_started
            if item is _BATCH_FINISHED:
                self._consumed = True
                return
            if isinstance(item, _BatchFailed):
                raise item.error
            self._ready_groups -= 1
            yield item

    async def ready_group_batches(self) -> AsyncIterator[list[TrainingGroup]]:
        """Drain complete groups already ready, with no collection delay.

        While the consumer builds/submits one request, the producer keeps
        filling this queue. The next request takes that entire ready prefix.
        """

        if self._consumer_started:
            raise RuntimeError(f"batch {self.batch_id} groups can only be consumed once")
        self._consumer_started = True
        while True:
            wait_started = time.monotonic()
            item = await self._queue.get()
            self._train_group_wait_time += time.monotonic() - wait_started
            ready: list[TrainingGroup] = []
            finished = False
            while True:
                if item is _BATCH_FINISHED:
                    finished = True
                    break
                if isinstance(item, _BatchFailed):
                    raise item.error
                self._ready_groups -= 1
                ready.append(item)
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            if ready:
                yield ready
            if finished:
                self._consumed = True
                return

    def _append_group(
        self,
        *,
        sequence: int,
        group: PromptGroup,
        submit_version: int,
        source_token: Hashable,
    ) -> TrainingGroup:
        if self._sealed:
            raise RuntimeError(f"batch {self.batch_id} is already sealed")
        self._accepted_sequences.append(sequence)
        self._all_groups.append(group)
        self._all_versions.append(submit_version)
        self._completed_rewards.extend(group.rewards)
        return TrainingGroup(
            batch_id=self.batch_id,
            index=self.realized_groups - 1,
            group=group,
            submit_version=submit_version,
            source_token=source_token,
        )

    def _put_group(self, group: TrainingGroup) -> None:
        self._queue.put_nowait(group)
        self._ready_groups += 1

    def _record_filtered_rewards(self, rewards: tuple[float, ...]) -> None:
        self._completed_rewards.extend(rewards)

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
    "TrainingGroup",
]
