"""Per-row group assembler for the per-sample rollout API.

The per-sample ``rollout_fn`` returns one :class:`RolloutSample` per call.
The async loop fans each row out to ``completions_per_prompt`` parallel
samples, then this assembler joins them back into a :class:`PromptGroup`
once all expected samples for a row land (or a partial-emission policy
fires).  Group advantages are computed only at the join point, so the
trainer never sees a half-formed group.

This mirrors AReaL/slime, where the user-facing function produces one
trajectory and the framework owns group assembly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional

from training.utils.data import compute_advantages
from training.utils.rl.losses import PromptGroup
from training.utils.rl.rollout.types import (
    Rollout,
    RolloutSample,
    rollout_to_prompt_group,
)


logger = logging.getLogger(__name__)


__all__ = [
    "GroupAssembler",
    "PendingGroup",
    "RowResolution",
]


AdvantageFn = Callable[[List[float]], List[float]]


@dataclass
class RowResolution:
    """Outcome of a row once all its samples have settled.

    ``pg`` is the assembled PromptGroup, or ``None`` when no group
    survived (every sample dropped, advantage_fn produced non-finite
    values, or the surviving sample count fell below ``min_group_size``).
    ``min_submit_version`` is the oldest submit version among the row's
    samples.
    """

    pg: Optional[PromptGroup]
    min_submit_version: int


@dataclass
class PendingGroup:
    """In-flight group of samples for one row."""

    row_id: Hashable
    expected_n: int
    samples: List[RolloutSample] = field(default_factory=list)
    submit_versions: List[int] = field(default_factory=list)
    row_meta: Optional[dict] = None
    # Number of samples started for this row.  ``samples + dropped == started``;
    # the group is "settled" when ``started == expected_n``.
    started: int = 0
    dropped: int = 0

    @property
    def settled(self) -> bool:
        return (len(self.samples) + self.dropped) >= self.expected_n

    @property
    def min_submit_version(self) -> int:
        return min(self.submit_versions) if self.submit_versions else 0


class GroupAssembler:
    """Join per-sample rollouts into PromptGroups by row id.

    Each row is assigned an ``expected_n`` (typically
    ``completions_per_prompt``).  The async loop calls :meth:`note_started`
    when a sample is submitted and one of :meth:`add_sample` /
    :meth:`note_dropped` when the task resolves.  Once a row's group is
    fully settled, the assembler packs the surviving samples through
    :func:`rollout_to_prompt_group` and returns the resulting PromptGroup
    (or ``None`` if every sample for the row was dropped or the
    ``advantage_fn`` produced non-finite values).

    The assembler is single-threaded -- the async loop drives it from
    the event-loop thread.
    """

    def __init__(
        self,
        *,
        completions_per_prompt: int,
        advantage_fn: AdvantageFn = compute_advantages,
        with_reference: bool = False,
        router_replay_completion_only: bool = False,
        min_group_size: int = 1,
    ) -> None:
        if completions_per_prompt < 1:
            raise ValueError("completions_per_prompt must be >= 1")
        if min_group_size < 1:
            raise ValueError("min_group_size must be >= 1")
        self._n = completions_per_prompt
        self._advantage_fn = advantage_fn
        self._with_reference = with_reference
        self._r3_completion_only = router_replay_completion_only
        self._min_group_size = min_group_size
        self._pending: Dict[Hashable, PendingGroup] = {}

    def note_started(
        self,
        row_id: Hashable,
        *,
        submit_version: int,
        row_meta: Optional[dict] = None,
    ) -> None:
        """Record that one sample for ``row_id`` was submitted.

        The first call for a given ``row_id`` materializes the
        :class:`PendingGroup` slot.  Subsequent calls just bump ``started``
        and append the submit version.  ``row_meta`` is stored on the
        first call; later calls are ignored (all samples for a row share
        meta).
        """
        group = self._pending.get(row_id)
        if group is None:
            group = PendingGroup(
                row_id=row_id,
                expected_n=self._n,
                row_meta=dict(row_meta) if row_meta else None,
            )
            self._pending[row_id] = group
        group.started += 1
        group.submit_versions.append(submit_version)

    def add_sample(
        self,
        row_id: Hashable,
        sample: RolloutSample,
    ) -> Optional[RowResolution]:
        """Record one resolved sample.

        Returns ``None`` if the row still has samples in flight, or a
        :class:`RowResolution` once the row has fully settled.  The
        resolution carries the assembled :class:`PromptGroup` (or
        ``None`` if no group survived).
        """
        group = self._require(row_id)
        group.samples.append(sample)
        return self._maybe_emit(row_id, group)

    def note_dropped(
        self,
        row_id: Hashable,
    ) -> Optional[RowResolution]:
        """Record that one sample for ``row_id`` failed.

        Same return contract as :meth:`add_sample`.
        """
        group = self._require(row_id)
        group.dropped += 1
        return self._maybe_emit(row_id, group)

    def _require(self, row_id: Hashable) -> PendingGroup:
        group = self._pending.get(row_id)
        if group is None:
            raise KeyError(
                f"GroupAssembler: row {row_id!r} has no pending group "
                "(note_started was not called).",
            )
        return group

    def _maybe_emit(
        self,
        row_id: Hashable,
        group: PendingGroup,
    ) -> Optional[RowResolution]:
        if not group.settled:
            return None
        del self._pending[row_id]
        min_version = group.min_submit_version
        if len(group.samples) < self._min_group_size:
            logger.info(
                "GroupAssembler: dropping row %r (got %d/%d samples; min=%d)",
                row_id, len(group.samples), group.expected_n, self._min_group_size,
            )
            return RowResolution(pg=None, min_submit_version=min_version)
        rollout = Rollout(samples=group.samples, row_meta=group.row_meta)
        pg = rollout_to_prompt_group(
            rollout,
            advantage_fn=self._advantage_fn,
            with_reference=self._with_reference,
            router_replay_completion_only=self._r3_completion_only,
        )
        return RowResolution(pg=pg, min_submit_version=min_version)

    def pending_rows(self) -> int:
        """Number of rows with at least one in-flight sample."""
        return len(self._pending)

    def drain(self) -> List[RowResolution]:
        """Force-emit any pending rows whose surviving samples meet
        ``min_group_size``.

        Use only at shutdown / after the sample iterator is exhausted and
        all in-flight tasks have resolved.  Rows still missing samples
        below ``min_group_size`` are dropped silently.
        """
        out: List[RowResolution] = []
        for row_id in list(self._pending):
            group = self._pending[row_id]
            del self._pending[row_id]
            min_version = group.min_submit_version
            if len(group.samples) < self._min_group_size:
                continue
            rollout = Rollout(samples=group.samples, row_meta=group.row_meta)
            pg = rollout_to_prompt_group(
                rollout,
                advantage_fn=self._advantage_fn,
                with_reference=self._with_reference,
                router_replay_completion_only=self._r3_completion_only,
            )
            if pg is not None:
                out.append(RowResolution(pg=pg, min_submit_version=min_version))
        return out
