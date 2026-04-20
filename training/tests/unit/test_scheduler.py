"""Unit tests for training.utils.rl.scheduler.AsyncRolloutScheduler."""

from __future__ import annotations

import asyncio

import pytest

from training.utils.rl.losses import PromptGroup
from training.utils.rl.scheduler import AsyncRolloutScheduler, RolloutStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pg(rewards: list[float]) -> PromptGroup:
    return PromptGroup(
        data=[],
        advantages=[],
        ref_logprobs=[],
        prompt_len=0,
        rewards=rewards,
    )


def _accept_all(_pg: PromptGroup) -> bool:
    return True


def _reject_uniform(pg: PromptGroup) -> bool:
    return len(set(pg.rewards)) > 1


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_init_rejects_invalid_step_target():
    with pytest.raises(ValueError, match="step_target"):
        AsyncRolloutScheduler(step_target=0, max_head_offpolicy_versions=0)


def test_init_rejects_negative_offpolicy():
    with pytest.raises(ValueError, match="max_head_offpolicy_versions"):
        AsyncRolloutScheduler(step_target=1, max_head_offpolicy_versions=-1)


def test_max_concurrent_capped_by_policy_window():
    s = AsyncRolloutScheduler(
        step_target=2, max_head_offpolicy_versions=1, max_concurrent=100,
    )
    # policy_window = (1 + 1) * 2 = 4, so max_concurrent clamped to 4
    assert s._max_concurrent == 4


# ---------------------------------------------------------------------------
# stream_groups happy path
# ---------------------------------------------------------------------------


def test_stream_groups_yields_step_target_groups():
    rows = [{"i": i} for i in range(5)]

    async def sample_fn(row: dict) -> PromptGroup:
        return _make_pg([float(row["i"]), 1.0 - float(row["i"])])

    async def run():
        s = AsyncRolloutScheduler(step_target=3, max_head_offpolicy_versions=0)
        results = []
        async for pg, version in s.stream_groups(sample_fn, iter(rows)):
            results.append((pg, version))
        return results, s

    results, s = asyncio.run(run())
    assert len(results) == 3
    assert all(v == 0 for _, v in results)
    assert s._total_accepted == 3


def test_stream_groups_filters_with_filter_fn():
    rows = [{"i": i} for i in range(10)]

    async def sample_fn(row: dict) -> PromptGroup:
        # Even-indexed rows produce uniform rewards (rejected by filter)
        if row["i"] % 2 == 0:
            return _make_pg([1.0, 1.0])
        return _make_pg([1.0, 0.0])

    async def run():
        s = AsyncRolloutScheduler(
            step_target=3,
            max_head_offpolicy_versions=0,
            filter_fn=_reject_uniform,
        )
        results = [pg async for pg, _ in s.stream_groups(sample_fn, iter(rows))]
        return results, s

    results, s = asyncio.run(run())
    assert len(results) == 3
    assert all(len(set(pg.rewards)) > 1 for pg in results)
    assert s._total_rejected >= 3  # at least 3 even-indexed rows rejected


def test_stream_groups_handles_sample_failures():
    rows = [{"i": i} for i in range(10)]

    async def sample_fn(row: dict) -> PromptGroup | None:
        if row["i"] in (0, 1, 2):
            raise RuntimeError("simulated sampling failure")
        return _make_pg([1.0, 0.0])

    async def run():
        s = AsyncRolloutScheduler(step_target=3, max_head_offpolicy_versions=0)
        results = [pg async for pg, _ in s.stream_groups(sample_fn, iter(rows))]
        return results

    results = asyncio.run(run())
    assert len(results) == 3


def test_stream_groups_stops_on_data_exhaustion():
    rows = [{"i": i} for i in range(2)]

    async def sample_fn(row: dict) -> PromptGroup:
        return _make_pg([1.0, 0.0])

    async def run():
        s = AsyncRolloutScheduler(step_target=10, max_head_offpolicy_versions=0)
        results = [pg async for pg, _ in s.stream_groups(sample_fn, iter(rows))]
        return results, s

    results, s = asyncio.run(run())
    assert len(results) == 2
    assert s.data_exhausted


# ---------------------------------------------------------------------------
# collect_batch
# ---------------------------------------------------------------------------


def test_collect_batch_returns_step_target_and_stats():
    rows = [{"i": i} for i in range(5)]

    async def sample_fn(row: dict) -> PromptGroup:
        return _make_pg([1.0, 0.0])

    async def run():
        s = AsyncRolloutScheduler(step_target=3, max_head_offpolicy_versions=0)
        return await s.collect_batch(sample_fn, iter(rows))

    accepted, stats = asyncio.run(run())
    assert len(accepted) == 3
    assert isinstance(stats, RolloutStats)
    assert stats.valid_groups == 3
    assert stats.total_sampled >= 3


def test_collect_batch_tracks_filter_drops_in_stats():
    rows = [{"i": i} for i in range(10)]

    async def sample_fn(row: dict) -> PromptGroup:
        # First 3 rows uniform (filtered), rest varied
        if row["i"] < 3:
            return _make_pg([1.0, 1.0])
        return _make_pg([1.0, 0.0])

    async def run():
        s = AsyncRolloutScheduler(
            step_target=3,
            max_head_offpolicy_versions=0,
            filter_fn=_reject_uniform,
        )
        return await s.collect_batch(sample_fn, iter(rows))

    accepted, stats = asyncio.run(run())
    assert len(accepted) == 3
    assert stats.filter_drops >= 3


# ---------------------------------------------------------------------------
# Versioning + get_state
# ---------------------------------------------------------------------------


def test_bump_version_increments_current_version():
    s = AsyncRolloutScheduler(step_target=1, max_head_offpolicy_versions=0)
    assert s.current_version == 0
    s.bump_version()
    s.bump_version()
    assert s.current_version == 2


def test_get_state_returns_resumable_counters():
    s = AsyncRolloutScheduler(
        step_target=1,
        max_head_offpolicy_versions=0,
        total_accepted=7,
        total_rejected=3,
        rows_submitted=12,
    )
    state = s.get_state()
    assert state == {
        "rows_submitted": 12,
        "total_accepted": 7,
        "total_rejected": 3,
    }


def test_resume_from_state():
    s = AsyncRolloutScheduler(
        step_target=2,
        max_head_offpolicy_versions=1,
        global_step=4,
        total_accepted=8,
        total_rejected=2,
        rows_submitted=10,
    )
    assert s.current_version == 4
    assert s._total_accepted == 8


# ---------------------------------------------------------------------------
# Staleness gating
# ---------------------------------------------------------------------------


def test_staleness_cap_respects_offpolicy_budget():
    # offpolicy=2, step_target=4, version=0, accepted=0, in_flight=0:
    # cap = (2 + 0 + 1) * 4 - (0 + 0) = 12
    s = AsyncRolloutScheduler(step_target=4, max_head_offpolicy_versions=2)
    assert s._staleness_cap() == 12

    # After accepting 4, cap shrinks by 4
    s._total_accepted = 4
    assert s._staleness_cap() == 8

    # After bumping version, cap grows by step_target
    s.bump_version()
    assert s._staleness_cap() == 12


def test_strict_on_policy_caps_at_step_target():
    # offpolicy=0, step_target=3: cap = (0 + 0 + 1) * 3 = 3
    s = AsyncRolloutScheduler(step_target=3, max_head_offpolicy_versions=0)
    assert s._staleness_cap() == 3
