"""Tests for the async off-policy RL training loop."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from training.utils.rl.losses import PromptGroup
from training.utils.rl.train import TrainStepFns
from training.utils.rl.train_async import AsyncConfig, run_rl_loop_async


def _make_prompt_group(reward: float = 1.0) -> PromptGroup:
    target = SimpleNamespace(shape=[3], data=[11, 12, 13])
    datum = SimpleNamespace(loss_fn_inputs={"target_tokens": target})
    return PromptGroup(
        data=[datum],
        advantages=[reward],
        ref_logprobs=[[0.1, 0.1, 0.1]],
        prompt_len=2,
        rewards=[reward],
        inf_logprobs=[[-0.2, -0.3, -0.4]],
        completion_lens=[2],
        truncated=[False],
    )


class TestAsyncConfig:
    def test_default_values(self):
        cfg = AsyncConfig()
        assert cfg.max_steps_off_policy == 2
        assert cfg.groups_per_batch == 8

    def test_max_staleness_cap(self):
        with pytest.raises(ValueError, match="exceeds the supported limit"):
            AsyncConfig(max_steps_off_policy=3)

    def test_negative_staleness(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            AsyncConfig(max_steps_off_policy=-1)

    def test_zero_groups(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            AsyncConfig(groups_per_batch=0)

    def test_valid_configs(self):
        AsyncConfig(max_steps_off_policy=0, groups_per_batch=1)
        AsyncConfig(max_steps_off_policy=1, groups_per_batch=4)
        AsyncConfig(max_steps_off_policy=2, groups_per_batch=16)


class TestRunRlLoopAsync:
    @pytest.fixture
    def simple_config(self) -> AsyncConfig:
        return AsyncConfig(max_steps_off_policy=2, groups_per_batch=2)

    def _make_train_step(self, train_log: list[dict]):
        def train_step(
            step: int,
            prompt_groups: list[PromptGroup],
            loop_stats: dict | None = None,
        ) -> tuple[int, dict]:
            train_log.append({
                "step": step,
                "n_groups": len(prompt_groups),
                "loop_stats": loop_stats,
            })
            return step + 1, {}

        return train_step

    def test_basic_training(self, simple_config: AsyncConfig):
        """4 rows with groups_per_batch=2 should produce 2 training steps."""
        rows = [{"id": i} for i in range(4)]

        async def sample_fn(row: dict) -> PromptGroup:
            return _make_prompt_group(reward=1.0)

        train_log: list[dict] = []
        train_fns = TrainStepFns(train_step=self._make_train_step(train_log))

        final_step = asyncio.run(run_rl_loop_async(
            sample_fn=sample_fn,
            rows=rows,
            train_fns=train_fns,
            async_config=simple_config,
        ))

        assert final_step == 2
        assert len(train_log) == 2
        assert all(entry["n_groups"] == 2 for entry in train_log)

    def test_remainder_rows_ignored(self, simple_config: AsyncConfig):
        """5 rows with groups_per_batch=2 should produce 2 steps, ignoring the remainder."""
        rows = [{"id": i} for i in range(5)]

        async def sample_fn(row: dict) -> PromptGroup:
            return _make_prompt_group()

        train_log: list[dict] = []
        train_fns = TrainStepFns(train_step=self._make_train_step(train_log))

        final_step = asyncio.run(run_rl_loop_async(
            sample_fn=sample_fn,
            rows=rows,
            train_fns=train_fns,
            async_config=simple_config,
        ))

        assert final_step == 2

    def test_empty_rows(self, simple_config: AsyncConfig):
        """0 rows should return the initial step."""
        async def sample_fn(row: dict) -> PromptGroup:
            return _make_prompt_group()

        train_log: list[dict] = []
        train_fns = TrainStepFns(train_step=self._make_train_step(train_log))

        final_step = asyncio.run(run_rl_loop_async(
            sample_fn=sample_fn,
            rows=[],
            train_fns=train_fns,
            async_config=simple_config,
            global_step=5,
        ))

        assert final_step == 5
        assert len(train_log) == 0

    def test_sample_failures_skipped(self, simple_config: AsyncConfig):
        """sample_fn returning None should be skipped, not counted as a valid group."""
        call_count = 0

        async def sample_fn(row: dict) -> PromptGroup | None:
            nonlocal call_count
            call_count += 1
            if row["id"] % 2 == 0:
                return None
            return _make_prompt_group()

        rows = [{"id": i} for i in range(8)]
        train_log: list[dict] = []
        train_fns = TrainStepFns(train_step=self._make_train_step(train_log))

        final_step = asyncio.run(run_rl_loop_async(
            sample_fn=sample_fn,
            rows=rows,
            train_fns=train_fns,
            async_config=simple_config,
        ))

        assert final_step == 2
        assert len(train_log) == 2

    def test_dynamic_filter(self, simple_config: AsyncConfig):
        """dynamic_filter_fn should reject groups; rejected groups don't count toward batch."""
        async def sample_fn(row: dict) -> PromptGroup:
            return _make_prompt_group(reward=float(row["id"]))

        def reject_low_reward(pg: PromptGroup) -> bool:
            return pg.rewards[0] > 2.0

        # 10 rows, groups_per_batch=2 → total_steps=5.
        # Rows 0,1,2 have reward <=2.0 → filtered.  Rows 3-9 pass.
        # Workers fill groups from the valid ones; filtered ones consume
        # rows but not training slots.  All 5 steps should complete since
        # 7 rows pass the filter (enough for 5 steps of 2 = 10, plus extras).
        rows = [{"id": i} for i in range(10)]
        train_log: list[dict] = []
        train_fns = TrainStepFns(train_step=self._make_train_step(train_log))

        final_step = asyncio.run(run_rl_loop_async(
            sample_fn=sample_fn,
            rows=rows,
            train_fns=train_fns,
            async_config=simple_config,
            dynamic_filter_fn=reject_low_reward,
        ))

        # Workers exhaust all 10 rows. Training tries 5 steps but may not
        # fill all 5 if too many are filtered. At least 3 steps should complete
        # (rows 3-9 give 7 valid groups → 3 full batches of 2, 1 leftover).
        assert final_step >= 3
        assert all(entry["n_groups"] <= 2 for entry in train_log)

    def test_generation_step_set(self, simple_config: AsyncConfig):
        """PromptGroups should have generation_step stamped by workers."""
        captured_groups: list[PromptGroup] = []

        def train_step(
            step: int,
            prompt_groups: list[PromptGroup],
            loop_stats: dict | None = None,
        ) -> tuple[int, dict]:
            captured_groups.extend(prompt_groups)
            return step + 1, {}

        async def sample_fn(row: dict) -> PromptGroup:
            return _make_prompt_group()

        rows = [{"id": i} for i in range(4)]
        train_fns = TrainStepFns(train_step=train_step)

        asyncio.run(run_rl_loop_async(
            sample_fn=sample_fn,
            rows=rows,
            train_fns=train_fns,
            async_config=simple_config,
        ))

        assert len(captured_groups) == 4
        for pg in captured_groups:
            assert pg.generation_step is not None
            assert pg.generation_step >= 0

    def test_metrics_callback(self, simple_config: AsyncConfig):
        """metrics_callback should be called with staleness info."""
        all_metrics: list[dict] = []

        async def sample_fn(row: dict) -> PromptGroup:
            return _make_prompt_group()

        def train_step(
            step: int,
            prompt_groups: list[PromptGroup],
            loop_stats: dict | None = None,
        ) -> tuple[int, dict]:
            return step + 1, {}

        rows = [{"id": i} for i in range(4)]
        train_fns = TrainStepFns(train_step=train_step)

        asyncio.run(run_rl_loop_async(
            sample_fn=sample_fn,
            rows=rows,
            train_fns=train_fns,
            async_config=simple_config,
            metrics_callback=all_metrics.append,
        ))

        assert len(all_metrics) == 2
        for m in all_metrics:
            assert "train/step" in m
            assert "version/sample_staleness_avg" in m

    def test_global_step_offset(self, simple_config: AsyncConfig):
        """global_step should start from the provided offset."""
        async def sample_fn(row: dict) -> PromptGroup:
            return _make_prompt_group()

        train_log: list[dict] = []
        train_fns = TrainStepFns(train_step=self._make_train_step(train_log))

        final_step = asyncio.run(run_rl_loop_async(
            sample_fn=sample_fn,
            rows=[{"id": i} for i in range(4)],
            train_fns=train_fns,
            async_config=simple_config,
            global_step=10,
        ))

        assert final_step == 12
        assert train_log[0]["step"] == 10
        assert train_log[1]["step"] == 11

    def test_sample_exception_treated_as_none(self, simple_config: AsyncConfig):
        """Exceptions from sample_fn should be caught and treated as None."""
        async def sample_fn(row: dict) -> PromptGroup:
            if row["id"] == 0:
                raise RuntimeError("simulated failure")
            return _make_prompt_group()

        # 6 rows, groups_per_batch=2 → total_steps=3.
        # Row 0 fails, rows 1-5 succeed (5 valid groups).
        # Training runs up to 3 steps: needs 6 valid groups but only 5 exist,
        # so it completes 2 full steps and then a partial 3rd if workers
        # are done. At least 2 steps should complete.
        rows = [{"id": i} for i in range(6)]
        train_log: list[dict] = []
        train_fns = TrainStepFns(train_step=self._make_train_step(train_log))

        final_step = asyncio.run(run_rl_loop_async(
            sample_fn=sample_fn,
            rows=rows,
            train_fns=train_fns,
            async_config=simple_config,
        ))

        assert final_step >= 2
        assert len(train_log) >= 2
