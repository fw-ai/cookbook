"""Unit tests for training.utils.rl.train_async."""

from __future__ import annotations

import asyncio

import pytest

from training.utils.rl.losses import PromptGroup
from training.utils.rl.train import TrainStepFns
from training.utils.rl.train_async import AsyncConfig, run_rl_loop_async


def _run(coro):
    return asyncio.run(coro)


def _empty_prompt_group(reward: float = 1.0) -> PromptGroup:
    return PromptGroup(
        data=[],
        advantages=[0.0],
        ref_logprobs=None,
        prompt_len=0,
        rewards=[reward],
    )


class TestAsyncConfig:
    def test_defaults(self):
        cfg = AsyncConfig()
        assert cfg.max_steps_off_policy == 2
        assert cfg.groups_per_batch == 8

    def test_rejects_staleness_above_two(self):
        with pytest.raises(ValueError, match="exceeds"):
            AsyncConfig(max_steps_off_policy=3)

    def test_rejects_negative_staleness(self):
        with pytest.raises(ValueError, match=">= 0"):
            AsyncConfig(max_steps_off_policy=-1)

    def test_rejects_zero_groups_per_batch(self):
        with pytest.raises(ValueError, match=">= 1"):
            AsyncConfig(groups_per_batch=0)


class TestRunRlLoopAsync:
    def test_returns_early_when_too_few_rows(self):
        cfg = AsyncConfig(groups_per_batch=8)

        def train_step(step, groups, stats):
            raise AssertionError("train_step should not fire")

        result = _run(
            run_rl_loop_async(
                sample_fn=lambda row: _async_return(_empty_prompt_group()),
                rows=[{"a": 1}],
                train_fns=TrainStepFns(train_step=train_step),
                async_config=cfg,
            )
        )
        assert result == 0

    def test_training_runs_one_step_per_batch(self):
        cfg = AsyncConfig(max_steps_off_policy=0, groups_per_batch=2)
        rows = [{"i": i} for i in range(4)]
        train_calls: list[int] = []

        def train_step(step, groups, stats):
            train_calls.append(step)
            assert len(groups) == 2
            return step + 1, {}

        async def sample_fn(row):
            return _empty_prompt_group(reward=float(row["i"]))

        result = _run(
            run_rl_loop_async(
                sample_fn=sample_fn,
                rows=rows,
                train_fns=TrainStepFns(train_step=train_step),
                async_config=cfg,
            )
        )
        assert result == 2
        assert train_calls == [0, 1]

    def test_sample_fn_exceptions_counted_as_sample_fails(self):
        cfg = AsyncConfig(max_steps_off_policy=0, groups_per_batch=1)
        rows = [{"i": i} for i in range(4)]
        captured_stats: list[dict] = []

        def train_step(step, groups, stats):
            captured_stats.append(stats)
            return step + 1, {}

        async def sample_fn(row):
            if row["i"] == 1:
                raise RuntimeError("boom")
            return _empty_prompt_group()

        _run(
            run_rl_loop_async(
                sample_fn=sample_fn, rows=rows,
                train_fns=TrainStepFns(train_step=train_step),
                async_config=cfg,
            )
        )
        # One of the four rows explodes -> it shows up as a sample_fail.
        assert sum(s.get("sample_fails", 0) for s in captured_stats) == 1

    def test_dynamic_filter_drops_groups(self):
        cfg = AsyncConfig(max_steps_off_policy=0, groups_per_batch=1)
        rows = [{"i": i} for i in range(4)]
        call_count = [0]

        def train_step(step, groups, stats):
            call_count[0] += 1
            return step + 1, {}

        async def sample_fn(row):
            return _empty_prompt_group(reward=float(row["i"]))

        _run(
            run_rl_loop_async(
                sample_fn=sample_fn, rows=rows,
                train_fns=TrainStepFns(train_step=train_step),
                async_config=cfg,
                dynamic_filter_fn=lambda pg: pg.rewards[0] >= 2.0,
            )
        )
        # i=0,1 get filtered; i=2,3 pass -> exactly 2 train calls.
        assert call_count[0] == 2

    def test_generation_step_stamped_on_prompt_group(self):
        cfg = AsyncConfig(max_steps_off_policy=0, groups_per_batch=1)
        rows = [{"i": i} for i in range(2)]
        seen: list[int | None] = []

        def train_step(step, groups, stats):
            seen.append(groups[0].generation_step)
            return step + 1, {}

        async def sample_fn(row):
            return _empty_prompt_group()

        _run(
            run_rl_loop_async(
                sample_fn=sample_fn, rows=rows,
                train_fns=TrainStepFns(train_step=train_step),
                async_config=cfg,
            )
        )
        assert all(s is not None and s >= 0 for s in seen)


async def _async_return(value):
    return value
