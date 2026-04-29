"""Unit + integration tests for ``RawRowCursor`` and ``raw_rows_from_stats``.

Covers cursor mechanics (clamping, resume ambiguity) and the end-to-end
behavior in ``run_rl_loop`` that the cursor is responsible for: cursor
advances by raw rows including dynamic-filter drops and sample failures.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from training.tests._dataloader_test_helpers import load_cursor, load_rl_train


@dataclass
class _PromptGroup:
    rewards: list[float]


RawRowCursor = load_cursor()
TrainStepFns, run_rl_loop = load_rl_train("dataloader_cursor", prompt_group_cls=_PromptGroup)


# ---------------------------------------------------------------------------
# RawRowCursor: resume() — clamping, ambiguity, fallback
# ---------------------------------------------------------------------------


def test_resume_with_persisted_value_sets_cursor():
    cursor = RawRowCursor(max_rows=10)
    cursor.resume(5)
    assert cursor.value == 5


def test_resume_clamps_persisted_past_max():
    cursor = RawRowCursor(max_rows=3)
    cursor.resume(99)
    assert cursor.value == 3


def test_resume_clamps_negative_persisted_to_zero():
    cursor = RawRowCursor(max_rows=10)
    cursor.resume(-5)
    assert cursor.value == 0


def test_resume_with_none_uses_fallback():
    """``persisted=None`` (no dataloader.json) → cursor takes ``fallback``."""
    cursor = RawRowCursor(max_rows=10)
    cursor.resume(None, fallback=4)
    assert cursor.value == 4


def test_resume_with_zero_persisted_and_zero_fallback_trusts_zero():
    """Fresh start (both 0) is trusted, not routed through fallback."""
    cursor = RawRowCursor(max_rows=10)
    cursor.resume(0, fallback=0)
    assert cursor.value == 0


def test_resume_with_zero_persisted_and_nonzero_fallback_takes_fallback():
    """Legacy ckpt signal: ``persisted=0`` with non-zero fallback → step-derived path."""
    cursor = RawRowCursor(max_rows=10)
    cursor.resume(0, fallback=4)
    assert cursor.value == 4


def test_resume_prefers_persisted_over_fallback_when_both_nonzero():
    cursor = RawRowCursor(max_rows=10)
    cursor.resume(7, fallback=4)
    assert cursor.value == 7


def test_resume_clamps_fallback_past_max():
    cursor = RawRowCursor(max_rows=3)
    cursor.resume(None, fallback=10)
    assert cursor.value == 3


def test_resume_without_max_rows_does_not_clamp():
    """Multi-epoch SFT/DPO cursors run unbounded."""
    cursor = RawRowCursor()  # max_rows=None
    cursor.resume(10_000)
    assert cursor.value == 10_000


# ---------------------------------------------------------------------------
# RawRowCursor: record() — accumulation, clamping, defensive paths
# ---------------------------------------------------------------------------


def test_record_advances_cursor():
    cursor = RawRowCursor(max_rows=10)
    cursor.record(3)
    cursor.record(5)
    assert cursor.value == 8


def test_record_clamps_at_max_rows():
    cursor = RawRowCursor(max_rows=3)
    cursor.record(99)
    assert cursor.value == 3


def test_record_zero_is_noop():
    cursor = RawRowCursor(max_rows=10)
    cursor.record(0)
    assert cursor.value == 0


def test_record_negative_does_not_rewind():
    cursor = RawRowCursor(max_rows=10)
    cursor.record(5)
    cursor.record(-3)
    assert cursor.value == 5


def test_record_after_resume_accumulates():
    cursor = RawRowCursor(max_rows=10)
    cursor.resume(2)
    cursor.record(3)
    assert cursor.value == 5


def test_record_without_max_rows_grows_unbounded():
    cursor = RawRowCursor()
    cursor.record(1_000_000)
    assert cursor.value == 1_000_000


# ---------------------------------------------------------------------------
# raw_rows_from_stats helper (RL-side adapter)
# ---------------------------------------------------------------------------


def test_raw_rows_from_stats_with_total_sampled():
    from training.tests._dataloader_test_helpers import _load_module, _stub_fireworks_sdk, _stub_rl_losses
    _stub_fireworks_sdk()
    _stub_rl_losses(_PromptGroup)
    m = _load_module("rl_train_for_stats", "utils/rl/train.py")
    assert m.raw_rows_from_stats({"total_sampled": 7}, accepted_rows=2) == 7


def test_raw_rows_from_stats_falls_back_when_key_missing():
    from training.tests._dataloader_test_helpers import _load_module
    m = _load_module("rl_train_for_stats", "utils/rl/train.py")
    assert m.raw_rows_from_stats({"filter_drops": 1}, accepted_rows=2) == 2


def test_raw_rows_from_stats_falls_back_on_corrupted_value():
    from training.tests._dataloader_test_helpers import _load_module
    m = _load_module("rl_train_for_stats", "utils/rl/train.py")
    assert m.raw_rows_from_stats({"total_sampled": "bad"}, accepted_rows=2) == 2
    assert m.raw_rows_from_stats({"total_sampled": None}, accepted_rows=2) == 2


def test_raw_rows_from_stats_falls_back_on_none_stats():
    from training.tests._dataloader_test_helpers import _load_module
    m = _load_module("rl_train_for_stats", "utils/rl/train.py")
    assert m.raw_rows_from_stats(None, accepted_rows=4) == 4


# ---------------------------------------------------------------------------
# Integration: end-to-end ``run_rl_loop`` + cursor (the original PR's contract)
# ---------------------------------------------------------------------------


def test_pipeline_clean_run_advances_cursor_to_dataset_length():
    """No drops, no fails: cursor at end equals dataset length."""
    rows = [{"id": i, "reward": 1.0} for i in range(8)]
    cursor = RawRowCursor(max_rows=len(rows))
    cursor.resume(None, fallback=0)
    from training.tests._dataloader_test_helpers import _load_module
    m = _load_module("rl_train_for_int", "utils/rl/train.py")

    async def sample_one(row):
        await asyncio.sleep(0)
        return _PromptGroup(rewards=[row["reward"]])

    def train_step(step, groups, stats):
        cursor.record(m.raw_rows_from_stats(stats, accepted_rows=len(groups)))
        return step + 1, {}

    asyncio.run(
        run_rl_loop(
            sample_fns=(sample_one(row) for row in rows[cursor.value:]),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            global_step=0,
            weight_sync_interval=2,
        )
    )

    assert cursor.value == 8


def test_pipeline_with_drops_advances_cursor_by_raw_rows_not_accepted():
    """Cursor advances by raw rows (incl. drops/fails), not accepted groups —
    the cursor-drift failure mode this PR fixes."""
    # Place drops/fails before the second accept in each window so their
    # stats land in the dispatched batch (see test_tail_drops_in_window_are_unaccounted).
    rows = [
        {"id": 0, "reward": 1.0},
        {"id": 1, "fail": True},    # sample fail
        {"id": 2, "reward": 0.0},   # filter drop
        {"id": 3, "reward": 1.0},
        {"id": 4, "reward": 1.0},
        {"id": 5, "reward": 1.0},
    ]
    cursor = RawRowCursor(max_rows=len(rows))
    cursor.resume(None, fallback=0)
    from training.tests._dataloader_test_helpers import _load_module
    m = _load_module("rl_train_for_drops", "utils/rl/train.py")

    async def sample_one(row):
        await asyncio.sleep(0)
        if row.get("fail"):
            return None
        return _PromptGroup(rewards=[row["reward"]])

    def train_step(step, groups, stats):
        cursor.record(m.raw_rows_from_stats(stats, accepted_rows=len(groups)))
        return step + 1, {}

    asyncio.run(
        run_rl_loop(
            sample_fns=(sample_one(row) for row in rows[cursor.value:]),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            dynamic_filter_fn=lambda g: g.rewards[0] > 0,
            global_step=0,
            weight_sync_interval=2,
        )
    )

    # 4 accepted + 1 drop + 1 fail = 6 raw rows.
    assert cursor.value == 6


def test_pipeline_resume_does_not_reprocess_consumed_rows():
    """Persisted cursor jumps past consumed rows; the loop never re-samples them."""
    rows = [{"id": i, "reward": 1.0} for i in range(10)]
    cursor = RawRowCursor(max_rows=len(rows))
    cursor.resume(4)  # pretend 4 raw rows already consumed
    sampled_ids: list[int] = []
    from training.tests._dataloader_test_helpers import _load_module
    m = _load_module("rl_train_for_resume", "utils/rl/train.py")

    async def sample_one(row):
        await asyncio.sleep(0)
        sampled_ids.append(row["id"])
        return _PromptGroup(rewards=[row["reward"]])

    def train_step(step, groups, stats):
        cursor.record(m.raw_rows_from_stats(stats, accepted_rows=len(groups)))
        return step + 1, {}

    asyncio.run(
        run_rl_loop(
            sample_fns=(sample_one(row) for row in rows[cursor.value:]),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            global_step=0,
            weight_sync_interval=1,
        )
    )

    assert sampled_ids == [4, 5, 6, 7, 8, 9]
    assert cursor.value == 10


def test_tail_drops_in_window_are_unaccounted():
    """Documents a pre-existing ``run_rl_loop`` gap: drops/fails after a
    window's final dispatch never reach ``train_step``. Flip the assert if fixed."""
    rows = [
        {"id": 0, "reward": 1.0},
        {"id": 1, "reward": 1.0},   # batch fills, dispatches
        {"id": 2, "reward": 0.0},   # tail drop after dispatch
        {"id": 3, "reward": 0.0},   # tail drop after dispatch
    ]
    cursor = RawRowCursor(max_rows=len(rows))
    cursor.resume(None, fallback=0)
    from training.tests._dataloader_test_helpers import _load_module
    m = _load_module("rl_train_for_tail", "utils/rl/train.py")

    async def sample_one(row):
        await asyncio.sleep(0)
        return _PromptGroup(rewards=[row["reward"]])

    def train_step(step, groups, stats):
        cursor.record(m.raw_rows_from_stats(stats, accepted_rows=len(groups)))
        return step + 1, {}

    asyncio.run(
        run_rl_loop(
            sample_fns=(sample_one(row) for row in rows[cursor.value:]),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            dynamic_filter_fn=lambda g: g.rewards[0] > 0,
            global_step=0,
            weight_sync_interval=2,
        )
    )

    # Only the first dispatched batch was recorded; rows 2,3 are uncounted.
    assert cursor.value == 2
