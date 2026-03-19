"""Tests for rollout scheduling infrastructure (rollout.py).

Covers:
- collect_sync_batch: correct collection, filtering, stats
- AsyncRolloutScheduler: capacity formula, version hard stop, refill,
  get_state/restore roundtrip, data_exhausted transitions, overlap
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List

import pytest

from training.utils.rl.losses import PromptGroup
from training.utils.rl.rollout import (
    AsyncRolloutScheduler,
    DynamicFilterFn,
    RolloutStats,
    collect_sync_batch,
)


def _make_pg(rewards: list[float] | None = None) -> PromptGroup:
    """Helper: minimal PromptGroup."""
    return PromptGroup(
        data=[],
        advantages=[],
        ref_logprobs=[],
        prompt_len=0,
        rewards=rewards or [1.0],
    )


# ---------------------------------------------------------------------------
# collect_sync_batch
# ---------------------------------------------------------------------------


class TestCollectSyncBatch:
    @pytest.mark.asyncio
    async def test_collects_all_results(self):
        async def ok():
            return _make_pg([1.0])

        groups, stats = await collect_sync_batch([ok(), ok(), ok()], target=3)
        assert len(groups) == 3
        assert stats.valid_groups == 3
        assert stats.total_sampled == 3
        assert stats.sample_fails == 0
        assert stats.filter_drops == 0

    @pytest.mark.asyncio
    async def test_counts_none_as_sample_fail(self):
        async def fail():
            return None

        groups, stats = await collect_sync_batch([fail(), fail()], target=2)
        assert len(groups) == 0
        assert stats.sample_fails == 2
        assert stats.total_sampled == 2

    @pytest.mark.asyncio
    async def test_filter_fn_drops_and_counts(self):
        async def ok():
            return _make_pg([0.0, 0.0])

        async def varied():
            return _make_pg([0.0, 1.0])

        def reject_uniform(pg: PromptGroup) -> bool:
            return len(set(pg.rewards)) > 1

        groups, stats = await collect_sync_batch(
            [ok(), varied(), ok()],
            filter_fn=reject_uniform,
            target=3,
        )
        assert len(groups) == 1
        assert stats.filter_drops == 2
        assert stats.valid_groups == 1

    @pytest.mark.asyncio
    async def test_exception_propagates(self):
        async def explode():
            raise ValueError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await collect_sync_batch([explode()], target=1)

    @pytest.mark.asyncio
    async def test_rewards_aggregated_in_stats(self):
        async def pg1():
            return _make_pg([1.0, 0.0])

        async def pg2():
            return _make_pg([0.5])

        _, stats = await collect_sync_batch([pg1(), pg2()], target=2)
        assert stats.raw_rewards == [1.0, 0.0, 0.5]

    @pytest.mark.asyncio
    async def test_wall_time_is_positive(self):
        async def ok():
            return _make_pg()

        _, stats = await collect_sync_batch([ok()], target=1)
        assert stats.wall_time >= 0


# ---------------------------------------------------------------------------
# AsyncRolloutScheduler
# ---------------------------------------------------------------------------


def _sample_factory(pg: PromptGroup | None = None):
    """Returns a sample_fn_factory that always succeeds with *pg*."""
    if pg is None:
        pg = _make_pg([0.0, 1.0])

    async def _sample(row: dict) -> PromptGroup | None:
        return pg

    return _sample


def _failing_factory():
    """Returns a sample_fn_factory that always returns None."""

    async def _sample(row: dict) -> PromptGroup | None:
        return None

    return _sample


class TestAsyncRolloutSchedulerCapacity:
    def test_initial_capacity(self):
        sched = AsyncRolloutScheduler(step_target=4, max_head_offpolicy_versions=2)
        assert sched._staleness_cap() == (2 + 0 + 1) * 4
        assert sched._concurrency_cap() == (2 + 1) * 4
        assert sched._capacity() == 12

    def test_capacity_grows_with_version(self):
        sched = AsyncRolloutScheduler(step_target=4, max_head_offpolicy_versions=2)
        sched.bump_version()
        assert sched.current_version == 1
        assert sched._staleness_cap() == (2 + 1 + 1) * 4

    def test_capacity_shrinks_with_accepted(self):
        sched = AsyncRolloutScheduler(
            step_target=4,
            max_head_offpolicy_versions=2,
            total_accepted=8,
        )
        assert sched._staleness_cap() == (2 + 0 + 1) * 4 - 8

    def test_capacity_zero_floor(self):
        sched = AsyncRolloutScheduler(
            step_target=1,
            max_head_offpolicy_versions=0,
            total_accepted=1,
        )
        assert sched._staleness_cap() == 0
        assert sched._capacity() == 0


class TestAsyncRolloutSchedulerCollectBatch:
    @pytest.mark.asyncio
    async def test_basic_collect(self):
        sched = AsyncRolloutScheduler(step_target=2, max_head_offpolicy_versions=2)
        rows = iter([{"id": 1}, {"id": 2}, {"id": 3}])
        groups, stats = await sched.collect_batch(_sample_factory(), rows)
        assert len(groups) == 2
        assert stats.valid_groups == 2
        assert stats.sample_fails == 0

    @pytest.mark.asyncio
    async def test_version_offsets_tracked(self):
        """Accepted groups should record staleness (current_version - sample_version)."""
        sched = AsyncRolloutScheduler(step_target=2, max_head_offpolicy_versions=3)
        rows = iter([{}, {}, {}, {}])

        # Step 0: collect 2 groups — all submitted at version 0, collected at version 0
        groups, stats = await sched.collect_batch(_sample_factory(), rows)
        assert len(groups) == 2
        assert stats.version_offsets == [0, 0]

        # Bump version (simulates train step)
        sched.bump_version()
        assert sched.current_version == 1

        # Step 1: any in-flight from version 0 + new from version 1
        groups2, stats2 = await sched.collect_batch(_sample_factory(), rows)
        assert len(groups2) == 2
        # All offsets should be 0 or 1 (submitted at version 0 or 1, collected at version 1)
        for offset in stats2.version_offsets:
            assert 0 <= offset <= 1

    @pytest.mark.asyncio
    async def test_filtering_triggers_refill(self):
        call_count = 0

        async def _sample(row: dict) -> PromptGroup | None:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _make_pg([0.0, 0.0])
            return _make_pg([0.0, 1.0])

        def reject_uniform(pg: PromptGroup) -> bool:
            return len(set(pg.rewards)) > 1

        sched = AsyncRolloutScheduler(
            step_target=1,
            max_head_offpolicy_versions=5,
            filter_fn=reject_uniform,
        )
        rows = iter([{} for _ in range(10)])
        groups, stats = await sched.collect_batch(lambda row: _sample(row), rows)
        assert len(groups) >= 1
        assert stats.filter_drops >= 2

    @pytest.mark.asyncio
    async def test_data_exhausted_transitions(self):
        sched = AsyncRolloutScheduler(step_target=1, max_head_offpolicy_versions=2)
        assert sched.data_exhausted is False

        rows = iter([{"id": 1}])
        groups, _ = await sched.collect_batch(_sample_factory(), rows)
        assert len(groups) == 1

        groups2, _ = await sched.collect_batch(_sample_factory(), rows)
        assert sched.data_exhausted is True

    @pytest.mark.asyncio
    async def test_partial_batch_on_exhaustion(self):
        sched = AsyncRolloutScheduler(step_target=3, max_head_offpolicy_versions=5)
        rows = iter([{"id": 1}, {"id": 2}])
        groups, stats = await sched.collect_batch(_sample_factory(), rows)
        assert len(groups) == 2
        assert stats.valid_groups == 2

    @pytest.mark.asyncio
    async def test_sample_fails_counted(self):
        sched = AsyncRolloutScheduler(step_target=1, max_head_offpolicy_versions=5)
        rows = iter([{}, {}, {}])
        groups, stats = await sched.collect_batch(_failing_factory(), rows)
        assert len(groups) == 0
        assert stats.sample_fails >= 1


class TestAsyncRolloutSchedulerVersionBump:
    def test_bump_increments(self):
        sched = AsyncRolloutScheduler(step_target=1, max_head_offpolicy_versions=2)
        assert sched.current_version == 0
        sched.bump_version()
        assert sched.current_version == 1
        sched.bump_version()
        assert sched.current_version == 2

    def test_bump_opens_capacity(self):
        sched = AsyncRolloutScheduler(
            step_target=1,
            max_head_offpolicy_versions=0,
            total_accepted=1,
        )
        assert sched._capacity() == 0
        sched.bump_version()
        assert sched._staleness_cap() == (0 + 1 + 1) * 1 - 1
        assert sched._capacity() >= 1


class TestAsyncRolloutSchedulerGetState:
    def test_roundtrip(self):
        sched = AsyncRolloutScheduler(
            step_target=4,
            max_head_offpolicy_versions=2,
            global_step=3,
            total_accepted=10,
            total_rejected=2,
            rows_submitted=15,
        )
        state = sched.get_state()
        assert state == {
            "rows_submitted": 15,
            "total_accepted": 10,
            "total_rejected": 2,
        }

        restored = AsyncRolloutScheduler(
            step_target=4,
            max_head_offpolicy_versions=2,
            global_step=3,
            **state,
        )
        assert restored.current_version == 3
        assert restored._total_accepted == 10
        assert restored._total_rejected == 2
        assert restored._rows_submitted == 15

    @pytest.mark.asyncio
    async def test_counters_update_during_collection(self):
        sched = AsyncRolloutScheduler(step_target=2, max_head_offpolicy_versions=3)
        rows = iter([{}, {}, {}])
        await sched.collect_batch(_sample_factory(), rows)

        state = sched.get_state()
        assert state["total_accepted"] == 2
        assert state["rows_submitted"] >= 2


class TestAsyncRolloutSchedulerOverlap:
    @pytest.mark.asyncio
    async def test_inflight_tasks_persist_between_calls(self):
        """In-flight tasks from capacity submissions should persist and be
        harvested in the next collect_batch call."""
        sched = AsyncRolloutScheduler(step_target=1, max_head_offpolicy_versions=3)
        rows = iter([{"id": i} for i in range(10)])

        groups1, _ = await sched.collect_batch(_sample_factory(), rows)
        assert len(groups1) == 1
        sched.bump_version()

        groups2, _ = await sched.collect_batch(_sample_factory(), rows)
        assert len(groups2) >= 1
