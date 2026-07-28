from __future__ import annotations

import asyncio

import pytest

from training.utils.dataloader import CursorDataLoader
from training.utils.rl import PromptGroup
from training.utils.rl.sync_batch import collect_prompt_groups


def _group(*rewards: float) -> PromptGroup:
    return PromptGroup(
        data=[],
        advantages=[],
        ref_logprobs=None,
        prompt_len=0,
        rewards=list(rewards),
    )


def test_collect_prompt_groups_refills_failures_and_filter_drops() -> None:
    results = {
        0: _group(0.0, 1.0),
        1: _group(1.0, 1.0),
        2: None,
        3: _group(0.0, 1.0),
    }
    calls: list[int] = []

    async def sample_prompt(row: dict, *, cursor_index: int):
        calls.append(cursor_index)
        return results[row["id"]]

    rows = iter(CursorDataLoader([{"id": i} for i in range(4)]))
    groups, indices, stats = asyncio.run(
        collect_prompt_groups(
            rows,
            target_size=2,
            sample_prompt=sample_prompt,
            should_accept=lambda group: len(set(group.rewards)) > 1,
        )
    )

    assert len(groups) == 2
    assert calls == [0, 1, 2, 3]
    assert indices == [0, 1, 2, 3]
    assert stats["total_sampled"] == 4
    assert stats["filter_drops"] == 1
    assert stats["sample_fails"] == 1
    assert stats["all_raw_rewards"] == [0.0, 1.0, 1.0, 1.0, 0.0, 1.0]


def test_collect_prompt_groups_returns_final_partial_batch() -> None:
    async def sample_prompt(_row: dict, *, cursor_index: int):
        return _group(float(cursor_index), float(cursor_index + 1))

    groups, indices, stats = asyncio.run(
        collect_prompt_groups(
            iter(CursorDataLoader([{}, {}])),
            target_size=3,
            sample_prompt=sample_prompt,
            should_accept=None,
        )
    )

    assert len(groups) == 2
    assert indices == [0, 1]
    assert stats["valid_prompt_groups"] == 2


def test_collect_prompt_groups_does_not_hide_rollout_exceptions() -> None:
    async def sample_prompt(_row: dict, *, cursor_index: int):
        raise RuntimeError(f"fatal row {cursor_index}")

    with pytest.raises(RuntimeError, match="fatal row 0"):
        asyncio.run(
            collect_prompt_groups(
                iter(CursorDataLoader([{}])),
                target_size=1,
                sample_prompt=sample_prompt,
                should_accept=None,
            )
        )


def test_collect_prompt_groups_validates_target_size() -> None:
    async def run() -> None:
        await collect_prompt_groups(
            iter(CursorDataLoader([])),
            target_size=0,
            sample_prompt=lambda *_args, **_kwargs: None,
            should_accept=None,
        )

    with pytest.raises(ValueError, match="target_size"):
        asyncio.run(run())
