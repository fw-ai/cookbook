"""IGPO dataloader-cursor contract tests.

The IGPO recipe shares ``RLPromptDataset`` with the RL recipe (PR 404).
These tests pin the *call sequence* IGPO uses against the dataset:

  resume(persisted, fallback_step) -> rows_from_cursor()
    -> [run async sample/train] -> record_batch(stats, accepted_rows)
    -> ckpt.save(data_consumed=rl_dataset.data_consumed)

so a future regression in either ``igpo_loop.py`` or ``RLPromptDataset``
that drops one of those calls will be caught.

The full ``igpo_loop`` import requires ``tinker`` (not installed in the
sandbox), so these tests load ``data.py`` and ``rl/train.py`` in
isolation via ``importlib.util`` -- mirroring ``test_rl_prompt_dataset.py``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from training.tests._dataloader_test_helpers import load_dataloader_test_modules


@dataclass
class _PromptGroup:
    rewards: list[float]


RLPromptDataset, TrainStepFns, run_rl_loop = load_dataloader_test_modules(
    "igpo_cursor", prompt_group_cls=_PromptGroup,
)


# ---------------------------------------------------------------------------
# Resume + record_batch contract (same dataset class as RL, IGPO-specific
# call sequence: no PPO minibatches, single optim step per batch)
# ---------------------------------------------------------------------------


def test_igpo_fresh_run_advances_cursor_by_total_sampled():
    """IGPO trains 1 optim step per batch (no ppo_n_minibatches), so
    ``record_batch`` is called once per dispatched batch. The cursor
    must advance by raw rows (incl. drops/fails), not by accepted groups.

    Use ``weight_sync_interval=2`` so each window is large enough for
    one full batch to assemble (window_size = 2*2 = 4 coros) and the
    drop/fail accounting is exercised inside a single window rather
    than spilling into partial-batch dispatches."""
    # Place the drop/fail *between* the two accepts in each window so the
    # stats counters carry them into the dispatched batch. (Drops/fails
    # that arrive after a window's final batch dispatch are uncounted by
    # ``run_rl_loop`` -- see test_tail_drops_in_window_are_unaccounted.)
    rows = [
        {"id": 0, "reward": 1.0},
        {"id": 1, "fail": True},          # sample failure
        {"id": 2, "reward": 0.0},          # filter drop
        {"id": 3, "reward": 1.0},
        {"id": 4, "reward": 1.0},
        {"id": 5, "reward": 1.0},
    ]
    dataset = RLPromptDataset(rows, prompts_per_step=2)
    dataset.resume(None, fallback_step=0)

    ckpt_saves: list[int] = []

    async def sample_one(row):
        await asyncio.sleep(0)
        if row.get("fail"):
            return None
        return _PromptGroup(rewards=[row["reward"]])

    def dynamic_filter(group):
        return group.rewards[0] > 0

    def train_step(step, groups, stats):
        # Mirror IGPO's exact call sequence at igpo_loop.py:729-742:
        # increment step, record_batch, then ckpt.save(data_consumed=cursor).
        new_step = step + 1
        dataset.record_batch(stats, accepted_rows=len(groups))
        ckpt_saves.append(dataset.data_consumed)
        return new_step, {}

    asyncio.run(
        run_rl_loop(
            sample_fns=(sample_one(row) for row in dataset.rows_from_cursor()),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            dynamic_filter_fn=dynamic_filter,
            global_step=0,
            weight_sync_interval=2,
        )
    )

    # All 6 rows pulled from the loader; cursor at end equals dataset length.
    assert dataset.data_consumed == 6
    assert dataset.rows_from_cursor() == []
    # Window 1 (rows 0..3): row0 accept, row1 fail, row2 drop, row3 accept →
    #   buffer hits 2 groups at row3, total_sampled=4. cursor 0→4.
    # Window 2 (rows 4,5): both accept → full batch. total_sampled=2. cursor 4→6.
    assert ckpt_saves == [4, 6]


def test_tail_drops_in_window_are_unaccounted():
    """Documents a *pre-existing* gap in ``run_rl_loop``: drops/fails
    that arrive AFTER a window's final full-batch dispatch are not
    surfaced through any ``train_step`` call, so ``record_batch`` is
    never told about them and the cursor underestimates raw rows.

    Not introduced by PR 404 -- the cursor pattern just makes it
    visible. Fixing requires ``run_rl_loop`` to emit a final stats-only
    callback even when the buffer is empty. This test is a regression
    guard: if the gap is fixed in the future, the assertion will flip
    and the test should be updated."""
    rows = [
        {"id": 0, "reward": 1.0},
        {"id": 1, "reward": 1.0},   # batch fills here, dispatch happens
        {"id": 2, "reward": 0.0},   # drop AFTER dispatch -> tail drop
        {"id": 3, "reward": 0.0},   # drop AFTER dispatch -> tail drop
    ]
    dataset = RLPromptDataset(rows, prompts_per_step=2)
    dataset.resume(None, fallback_step=0)

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
            weight_sync_interval=2,  # window_size = 4 = the whole dataset
        )
    )

    # Today's behavior: cursor is 2 (only the first dispatched batch's
    # total_sampled was recorded). Rows 2 and 3 were pulled from the
    # loader and filter-dropped, but their stats were never delivered.
    # On resume we'd re-sample them -- benign in this case but a real
    # gap. Update this assertion to ``== 4`` once run_rl_loop emits a
    # tail-stats callback for the orphaned counters.
    assert dataset.data_consumed == 2


def test_igpo_resume_skips_already_consumed_rows():
    """IGPO's resume contract: persisted ``data_consumed`` jumps the
    cursor past consumed rows; the loop never re-samples them."""
    rows = [{"id": i, "reward": 1.0} for i in range(8)]
    dataset = RLPromptDataset(rows, prompts_per_step=2)

    sampled_ids: list[int] = []

    async def sample_one(row):
        await asyncio.sleep(0)
        sampled_ids.append(row["id"])
        return _PromptGroup(rewards=[row["reward"]])

    def train_step(step, groups, stats):
        dataset.record_batch(stats, accepted_rows=len(groups))
        return step + 1, {}

    # Mirror igpo_loop.py:783-788: resume with persisted cursor=5
    # (e.g. last run trained 2 batches with 1 drop in batch 2 → consumed 5 rows).
    dataset.resume(5, fallback_step=2)
    asyncio.run(
        run_rl_loop(
            sample_fns=(sample_one(row) for row in dataset.rows_from_cursor()),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            global_step=0,
            weight_sync_interval=1,
        )
    )

    # Rows 0..4 must NOT be re-sampled; only 5,6,7 reach the sampler.
    assert sampled_ids == [5, 6, 7]
    assert dataset.data_consumed == 8


def test_igpo_legacy_ckpt_resume_falls_back_to_step_cursor():
    """Pre-PR ckpts have ``data_consumed=0`` in dataloader.json (legacy
    default). The loop must fall back to the step-derived cursor so
    rollouts already done aren't redone."""
    rows = [{"id": i, "reward": 1.0} for i in range(6)]
    dataset = RLPromptDataset(rows, prompts_per_step=2)

    sampled_ids: list[int] = []

    async def sample_one(row):
        await asyncio.sleep(0)
        sampled_ids.append(row["id"])
        return _PromptGroup(rewards=[row["reward"]])

    def train_step(step, groups, stats):
        dataset.record_batch(stats, accepted_rows=len(groups))
        return step + 1, {}

    # Legacy: persisted is 0 (no dataloader.json for this ckpt key) and
    # step_offset=2 (2 rollouts already done).
    dataset.resume(0, fallback_step=2)
    assert dataset.data_consumed == 4

    asyncio.run(
        run_rl_loop(
            sample_fns=(sample_one(row) for row in dataset.rows_from_cursor()),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            global_step=2,
            weight_sync_interval=1,
        )
    )

    assert sampled_ids == [4, 5]
    assert dataset.data_consumed == 6


def test_igpo_dcp_save_intermediate_uses_live_cursor():
    """DCP saves at fixed step intervals must read the *current* cursor,
    not a precomputed offset. After each batch the cursor reflects all
    raw rows consumed so far across the run -- including drops."""
    rows = [
        {"id": 0, "reward": 1.0},
        {"id": 1, "reward": 0.0},   # drop
        {"id": 2, "reward": 1.0},
        {"id": 3, "reward": 1.0},
        {"id": 4, "reward": 1.0},
        {"id": 5, "reward": 1.0},
    ]
    dataset = RLPromptDataset(rows, prompts_per_step=2)
    dataset.resume(None, fallback_step=0)

    save_calls: list[tuple[int, int]] = []

    def train_step(step, groups, stats):
        new_step = step + 1
        dataset.record_batch(stats, accepted_rows=len(groups))
        # IGPO's DCP cadence is per step; emulate it firing every step.
        save_calls.append((new_step, dataset.data_consumed))
        return new_step, {}

    async def sample_one(row):
        await asyncio.sleep(0)
        return _PromptGroup(rewards=[row["reward"]])

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

    # Cursor must be monotonically non-decreasing across saves.
    cursors = [c for _, c in save_calls]
    assert cursors == sorted(cursors)
    # Final save reaches end-of-dataset.
    assert cursors[-1] == 6


# ---------------------------------------------------------------------------
# Regression guards on igpo_loop.py source — catch dropped calls
# ---------------------------------------------------------------------------


def test_igpo_loop_source_calls_dataset_cursor_methods():
    """Static check: ensure ``igpo_loop.py`` actually calls the four
    cursor methods. A drive-by edit that drops one of these silently
    re-introduces the drift bug, which type-check / smoke tests miss."""
    src = (Path(__file__).parent / "../recipes/igpo_loop.py").read_text()
    assert "rl_dataset.resume(" in src, "missing rl_dataset.resume(...) call"
    assert "rl_dataset.rows_from_cursor()" in src, "missing rows_from_cursor()"
    assert "rl_dataset.record_batch(" in src, "missing record_batch(...)"
    assert "data_consumed=rl_dataset.data_consumed" in src, (
        "ckpt.save must persist rl_dataset.data_consumed; old step-derived "
        "formula re-introduces drift on filter/sample drops"
    )


def test_igpo_loop_source_no_legacy_data_consumed_formula():
    """The pre-fix formula ``(step - step_offset) * prompt_groups_per_step``
    must not return. If a future refactor brings it back the cursor will
    silently regress; fail loudly here instead."""
    src = (Path(__file__).parent / "../recipes/igpo_loop.py").read_text()
    assert "(step - step_offset) * prompt_groups_per_step" not in src
    assert "(global_step - step_offset) * prompt_groups_per_step" not in src
