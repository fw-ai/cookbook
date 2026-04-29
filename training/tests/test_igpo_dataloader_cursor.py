"""IGPO cursor-contract tests: pins the resume → rows_from_cursor →
record_batch → ckpt.save call sequence against ``RLPromptDataset``."""

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


# Resume + record_batch contract (IGPO uses 1 optim step per batch).


def test_igpo_fresh_run_advances_cursor_by_total_sampled():
    """Cursor advances by raw rows (incl. drops/fails), not accepted groups."""
    # weight_sync_interval=2 ⇒ window_size=4; place drops between accepts so
    # their stats land in a dispatched batch (see tail-drops test for the gap).
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
        # Mirrors igpo_loop.py: step++, record_batch, ckpt.save(data_consumed=cursor).
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
    """Documents a pre-existing ``run_rl_loop`` gap: drops/fails after a
    window's final dispatch never reach ``record_batch``. Flip to ``== 4`` if fixed."""
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

    # Cursor=2: only the first dispatched batch's stats were recorded.
    assert dataset.data_consumed == 2


def test_igpo_resume_skips_already_consumed_rows():
    """Persisted ``data_consumed`` jumps cursor past consumed rows."""
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

    # Persisted cursor=5 (2 batches, 1 drop in batch 2 → 5 raw rows).
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
    """Pre-PR ckpts (``data_consumed=0``) fall back to step-derived cursor."""
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

    # Legacy: persisted=0 (no dataloader.json), step_offset=2.
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
    """DCP saves read live cursor; cursor is monotonically non-decreasing across saves."""
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
    """Source-string guard: ``igpo_loop.py`` must call all four cursor methods."""
    src = (Path(__file__).parent / "../recipes/igpo_loop.py").read_text()
    assert "rl_dataset.resume(" in src, "missing rl_dataset.resume(...) call"
    assert "rl_dataset.rows_from_cursor()" in src, "missing rows_from_cursor()"
    assert "rl_dataset.record_batch(" in src, "missing record_batch(...)"
    assert "data_consumed=rl_dataset.data_consumed" in src, (
        "ckpt.save must persist rl_dataset.data_consumed; old step-derived "
        "formula re-introduces drift on filter/sample drops"
    )


def test_igpo_loop_source_no_legacy_data_consumed_formula():
    """The legacy ``(step - step_offset) * prompt_groups_per_step`` formula must not return."""
    src = (Path(__file__).parent / "../recipes/igpo_loop.py").read_text()
    assert "(step - step_offset) * prompt_groups_per_step" not in src
    assert "(global_step - step_offset) * prompt_groups_per_step" not in src
