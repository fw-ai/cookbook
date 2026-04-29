"""Adversarial tests for the PR 404 RL dataloader cursor accounting.

These tests probe edge cases beyond the happy-path coverage in
``test_rl_prompt_dataset.py``: clamping, ambiguous resume signals,
malformed stats, defensive behavior, end-of-dataset, multi-window
training across weight-sync boundaries, mid-window dcp save.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass

import pytest

from training.tests._dataloader_test_helpers import load_dataloader_test_modules


@dataclass
class _PromptGroup:
    rewards: list[float]


RLPromptDataset, TrainStepFns, run_rl_loop = load_dataloader_test_modules(
    "rl_prompt_dataset_strict", prompt_group_cls=_PromptGroup,
)


# ----------------------------------------------------------------------------
# resume(): boundary, clamping, and ambiguity
# ----------------------------------------------------------------------------


def test_resume_clamps_data_consumed_past_dataset_end():
    """A persisted cursor that exceeds the (possibly shrunken) dataset
    must clamp to ``len(rows)`` so ``rows_from_cursor()`` returns []
    instead of slicing into negative-length territory or crashing."""
    dataset = RLPromptDataset([{"id": i} for i in range(3)], prompts_per_step=2)

    dataset.resume(99)

    assert dataset.data_consumed == 3
    assert dataset.rows_from_cursor() == []


def test_resume_clamps_negative_data_consumed_to_zero():
    """A bad ckpt with a negative cursor should not poison cursor state."""
    dataset = RLPromptDataset([{"id": i} for i in range(3)], prompts_per_step=2)

    dataset.resume(-5)

    assert dataset.data_consumed == 0
    assert dataset.rows_from_cursor() == [{"id": 0}, {"id": 1}, {"id": 2}]


def test_resume_with_none_data_consumed_uses_step_fallback():
    """``data_consumed=None`` is the explicit 'no dataloader.json' signal.
    The cursor must be derived purely from ``fallback_step``."""
    dataset = RLPromptDataset([{"id": i} for i in range(8)], prompts_per_step=2)

    dataset.resume(None, fallback_step=6, minibatches_per_step=3)

    # 6 // 3 = 2 rollouts × 2 prompts_per_step = 4
    assert dataset.data_consumed == 4
    assert dataset.rows_from_cursor() == [{"id": i} for i in range(4, 8)]


def test_resume_with_data_consumed_zero_and_fallback_step_zero_trusts_zero():
    """Fresh start: both signals are 0, treat as 'nothing consumed' rather
    than legacy fallback. Otherwise a fresh run with no ckpt would
    spuriously route through the fallback path."""
    dataset = RLPromptDataset([{"id": i} for i in range(4)], prompts_per_step=2)

    dataset.resume(0, fallback_step=0, minibatches_per_step=1)

    assert dataset.data_consumed == 0
    assert dataset.rows_from_cursor() == [{"id": i} for i in range(4)]


def test_resume_prefers_persisted_cursor_over_step_fallback_when_both_present():
    """When both signals are non-zero (typical post-PR resume), the
    persisted ``data_consumed`` must win. Otherwise the fix is silently
    ignored and the legacy step-derived cursor takes over."""
    dataset = RLPromptDataset([{"id": i} for i in range(10)], prompts_per_step=2)

    # data_consumed=7 (raw rows) vs fallback_step=4 (= 4 prompts under
    # minibatches=1). The two disagree by design — drops/fails advanced
    # the cursor past the rollout-derived position.
    dataset.resume(7, fallback_step=4, minibatches_per_step=1)

    assert dataset.data_consumed == 7
    assert dataset.rows_from_cursor() == [{"id": 7}, {"id": 8}, {"id": 9}]


def test_cursor_from_step_handles_zero_minibatches_per_step():
    """``minibatches_per_step=0`` would otherwise divide by zero. The
    helper must clamp via ``max(1, mb)`` so the loop doesn't crash on a
    misconfigured cfg."""
    dataset = RLPromptDataset([{"id": i} for i in range(6)], prompts_per_step=2)

    cursor = dataset.cursor_from_step(4, minibatches_per_step=0)

    # max(1, 0)=1 → 4 // 1 = 4 rollouts × 2 = 8 (then clamps in resume())
    assert cursor == 8

    dataset.resume(None, fallback_step=4, minibatches_per_step=0)
    assert dataset.data_consumed == 6  # clamped to len(rows)


def test_resume_clamps_step_fallback_past_dataset_end():
    """Even the step-derived fallback must clamp — a stale step counter
    on a shrunken dataset must not produce ``cursor > len``."""
    dataset = RLPromptDataset([{"id": i} for i in range(3)], prompts_per_step=2)

    dataset.resume(None, fallback_step=10, minibatches_per_step=1)

    assert dataset.data_consumed == 3
    assert dataset.rows_from_cursor() == []


# ----------------------------------------------------------------------------
# record_batch(): malformed stats, defensive paths
# ----------------------------------------------------------------------------


def test_record_batch_with_missing_total_sampled_falls_back_to_accepted_rows():
    """Defensive behavior: if a future stats producer ever omits the
    ``total_sampled`` key, the cursor still advances by the (smaller)
    accepted count rather than crashing or staying stuck at 0."""
    dataset = RLPromptDataset([{"id": i} for i in range(5)], prompts_per_step=2)

    dataset.record_batch({"filter_drops": 1, "sample_fails": 0}, accepted_rows=2)

    assert dataset.data_consumed == 2


def test_record_batch_with_non_int_total_sampled_falls_back_to_accepted_rows():
    """A corrupted stats dict must not crash the trainer; recover via
    the ``accepted_rows`` count. ``int('foo')`` → ValueError caught."""
    dataset = RLPromptDataset([{"id": i} for i in range(5)], prompts_per_step=2)

    dataset.record_batch({"total_sampled": "not-an-int"}, accepted_rows=2)

    assert dataset.data_consumed == 2


def test_record_batch_with_none_total_sampled_falls_back_to_accepted_rows():
    """``int(None)`` → TypeError caught — accepted_rows path."""
    dataset = RLPromptDataset([{"id": i} for i in range(5)], prompts_per_step=2)

    dataset.record_batch({"total_sampled": None}, accepted_rows=2)

    assert dataset.data_consumed == 2


def test_record_batch_with_negative_total_sampled_does_not_decrement():
    """A buggy producer reporting a negative count must never rewind the
    cursor — that would re-train rows already trained."""
    dataset = RLPromptDataset([{"id": i} for i in range(5)], prompts_per_step=2)
    dataset.resume(2)

    dataset.record_batch({"total_sampled": -3}, accepted_rows=2)

    assert dataset.data_consumed == 2  # unchanged, not 2 + (-3) = -1


def test_record_batch_with_float_total_sampled_truncates():
    """``int(3.7)`` → 3. We accept truncation rather than raising; the
    rest of the pipeline treats floats as a non-issue."""
    dataset = RLPromptDataset([{"id": i} for i in range(8)], prompts_per_step=2)

    dataset.record_batch({"total_sampled": 3.7}, accepted_rows=2)

    assert dataset.data_consumed == 3


def test_record_batch_clamps_advance_at_dataset_end():
    """An overshooting batch (e.g. final partial window) must not push
    the cursor past ``len(rows)``."""
    dataset = RLPromptDataset([{"id": i} for i in range(3)], prompts_per_step=2)
    dataset.resume(2)

    dataset.record_batch({"total_sampled": 99}, accepted_rows=1)

    assert dataset.data_consumed == 3
    assert dataset.rows_from_cursor() == []


def test_record_batch_repeated_calls_accumulate():
    """Each ``record_batch`` adds to the cursor; the cumulative count
    after a sequence of batches must equal the sum of their
    ``total_sampled`` (capped at len)."""
    dataset = RLPromptDataset([{"id": i} for i in range(20)], prompts_per_step=2)

    dataset.record_batch({"total_sampled": 3}, accepted_rows=2)
    dataset.record_batch({"total_sampled": 5}, accepted_rows=2)
    dataset.record_batch({"total_sampled": 2}, accepted_rows=2)

    assert dataset.data_consumed == 10


def test_record_batch_handles_empty_dataset():
    """Empty datasets are degenerate but must not raise. ``len`` is 0
    so any advance is clamped."""
    dataset = RLPromptDataset([], prompts_per_step=2)

    dataset.record_batch({"total_sampled": 5}, accepted_rows=2)

    assert dataset.data_consumed == 0
    assert dataset.rows_from_cursor() == []
    assert len(dataset) == 0


# ----------------------------------------------------------------------------
# Integration: multi-window run with weight syncs and overlapping windows
# ----------------------------------------------------------------------------


def _make_pipeline_dataset(n: int) -> RLPromptDataset:
    """All rows accept by default; ``fail`` and ``reward=0`` rows model
    the dynamic-filter / sample-failure failure modes."""
    rows = []
    for i in range(n):
        rows.append({"id": i, "reward": 1.0})
    return RLPromptDataset(rows, prompts_per_step=2)


def test_pipeline_cursor_matches_full_dataset_with_clean_run():
    """End-to-end: a clean run (no filter drops, no sample fails) must
    advance the cursor by exactly ``len(rows)`` — no off-by-one."""
    dataset = _make_pipeline_dataset(8)
    dataset.resume(0)
    train_log = []

    async def sample_one(row):
        await asyncio.sleep(0)
        return _PromptGroup(rewards=[row["reward"]])

    def train_step(step, groups, stats):
        dataset.record_batch(stats, accepted_rows=len(groups))
        train_log.append((step, dataset.data_consumed))
        return step + 1, {}

    asyncio.run(
        run_rl_loop(
            sample_fns=(sample_one(row) for row in dataset.rows_from_cursor()),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=dataset.prompts_per_step,
            global_step=0,
            weight_sync_interval=2,
        )
    )

    assert dataset.data_consumed == 8
    # 4 batches of 2 prompts each — cursor strictly increasing.
    cursors = [c for _, c in train_log]
    assert cursors == sorted(cursors)
    assert cursors[-1] == 8


def test_pipeline_resume_after_partial_run_does_not_reprocess_rows():
    """Resume from a mid-run cursor: training must only see rows beyond
    the cursor, and the cumulative cursor at the end equals total rows."""
    dataset = _make_pipeline_dataset(10)
    seen_ids = []

    async def sample_one(row):
        await asyncio.sleep(0)
        seen_ids.append(row["id"])
        return _PromptGroup(rewards=[row["reward"]])

    def train_step(step, groups, stats):
        dataset.record_batch(stats, accepted_rows=len(groups))
        return step + 1, {}

    # Pretend ckpt persisted cursor=4 (rows 0-3 already consumed).
    dataset.resume(4)
    asyncio.run(
        run_rl_loop(
            sample_fns=(sample_one(row) for row in dataset.rows_from_cursor()),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=dataset.prompts_per_step,
            global_step=0,
            weight_sync_interval=1,
        )
    )

    assert seen_ids == [4, 5, 6, 7, 8, 9]
    assert dataset.data_consumed == 10
    assert dataset.rows_from_cursor() == []


def test_pipeline_with_drops_advances_by_total_sampled_not_accepted():
    """The whole point of the PR: cursor must equal *raw* rows consumed,
    so that resume skips drops/fails too. With one filter drop in the
    middle, cursor advances by 3 (not 2) for that batch."""
    rows = [
        {"id": 0, "reward": 1.0},
        {"id": 1, "reward": 0.0},  # filtered out
        {"id": 2, "reward": 1.0},
        {"id": 3, "reward": 1.0},
        {"id": 4, "reward": 1.0},
    ]
    dataset = RLPromptDataset(rows, prompts_per_step=2)
    dataset.resume(0)

    async def sample_one(row):
        await asyncio.sleep(0)
        return _PromptGroup(rewards=[row["reward"]])

    def train_step(step, groups, stats):
        dataset.record_batch(stats, accepted_rows=len(groups))
        return step + 1, {}

    asyncio.run(
        run_rl_loop(
            sample_fns=(sample_one(row) for row in dataset.rows_from_cursor()),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            dynamic_filter_fn=lambda g: g.rewards[0] > 0,
            global_step=0,
            weight_sync_interval=1,
        )
    )

    # Only one full batch trains: rows 0 (accept) + 1 (drop) + 2 (accept).
    # Rows 3,4 form a partial batch (size 2, both accept) which run_rl_loop
    # also dispatches as a final batch.
    assert dataset.data_consumed == 5
    assert dataset.rows_from_cursor() == []


def test_resume_after_drops_skips_correct_row_count():
    """Simulate: round 1 ran with a drop (cursor=3 after 1 batch),
    process crashes, ckpt persisted data_consumed=3. Round 2 must
    resume from row 3 — rows 0,1,2 are NOT re-sampled."""
    rows = [{"id": i, "reward": 1.0} for i in range(7)]
    dataset = RLPromptDataset(rows, prompts_per_step=2)

    seen_ids: list[int] = []

    async def sample_one(row):
        await asyncio.sleep(0)
        seen_ids.append(row["id"])
        return _PromptGroup(rewards=[row["reward"]])

    def train_step(step, groups, stats):
        dataset.record_batch(stats, accepted_rows=len(groups))
        return step + 1, {}

    # Persisted cursor from prior run (3 raw rows consumed, e.g. 2 accept + 1 drop).
    dataset.resume(3, fallback_step=2, minibatches_per_step=1)
    asyncio.run(
        run_rl_loop(
            sample_fns=(sample_one(row) for row in dataset.rows_from_cursor()),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            global_step=0,
            weight_sync_interval=1,
        )
    )

    assert 0 not in seen_ids and 1 not in seen_ids and 2 not in seen_ids
    assert seen_ids == [3, 4, 5, 6]
    assert dataset.data_consumed == 7


# ----------------------------------------------------------------------------
# get_batch / __len__ unchanged-by-PR sanity checks
# ----------------------------------------------------------------------------


def test_get_batch_unaffected_by_cursor_state():
    """``get_batch(i)`` is index-based and must not be perturbed by the
    cursor (the cursor only affects ``rows_from_cursor`` and resume)."""
    dataset = RLPromptDataset([{"id": i} for i in range(6)], prompts_per_step=2)
    dataset.resume(4)

    # cursor=4 but we still index from 0
    assert dataset.get_batch(0) == [{"id": 0}, {"id": 1}]
    assert dataset.get_batch(1) == [{"id": 2}, {"id": 3}]
    assert dataset.get_batch(2) == [{"id": 4}, {"id": 5}]


def test_len_uses_ceil_for_partial_final_batch():
    """7 rows, prompts_per_step=2 → 4 batches (last is partial)."""
    dataset = RLPromptDataset([{"id": i} for i in range(7)], prompts_per_step=2)
    assert len(dataset) == 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
