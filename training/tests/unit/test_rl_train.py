"""Tests for training.utils.rl.train — pipeline RL loop."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from training.utils.rl.losses import PromptGroup
from training.utils.rl.train import TrainStepFns, run_rl_loop


def _make_pg(reward: float = 1.0) -> PromptGroup:
    return PromptGroup(
        data=[], advantages=[0.5], ref_logprobs=[], prompt_len=1,
        rewards=[reward],
    )


async def _sample_ok(reward: float = 1.0) -> PromptGroup | None:
    return _make_pg(reward)


async def _sample_none() -> PromptGroup | None:
    return None


def _make_train_step(log: list):
    """Return a train_step callback that records calls and increments step."""
    def train_step(step, groups, stats=None):
        log.append({"step": step, "n_groups": len(groups)})
        return step + 1, {"loss": 0.1}
    return train_step


# ---------------------------------------------------------------------------
# Basic pipeline behavior
# ---------------------------------------------------------------------------


class TestRunRlLoopBasic:
    def test_empty_coroutines_returns_initial_step(self):
        result = asyncio.run(run_rl_loop(
            sample_fns=[],
            train_fns=TrainStepFns(train_step=lambda *a: (0, {})),
        ))
        assert result == 0

    def test_single_step_trains_once(self):
        log = []
        result = asyncio.run(run_rl_loop(
            sample_fns=[_sample_ok()],
            train_fns=TrainStepFns(train_step=_make_train_step(log)),
            prompt_groups_per_step=1,
        ))
        assert result == 1
        assert len(log) == 1
        assert log[0] == {"step": 0, "n_groups": 1}

    def test_multiple_steps(self):
        log = []
        result = asyncio.run(run_rl_loop(
            sample_fns=[_sample_ok() for _ in range(6)],
            train_fns=TrainStepFns(train_step=_make_train_step(log)),
            prompt_groups_per_step=2,
        ))
        assert result == 3
        assert len(log) == 3

    def test_filtered_groups_are_skipped(self):
        log = []
        result = asyncio.run(run_rl_loop(
            sample_fns=[_sample_ok(0.0), _sample_ok(0.0)],
            train_fns=TrainStepFns(train_step=_make_train_step(log)),
            prompt_groups_per_step=2,
            dynamic_filter_fn=lambda pg: False,
        ))
        assert result == 0
        assert len(log) == 0

    def test_none_samples_counted_as_failures(self):
        log = []
        result = asyncio.run(run_rl_loop(
            sample_fns=[_sample_none(), _sample_ok()],
            train_fns=TrainStepFns(train_step=_make_train_step(log)),
            prompt_groups_per_step=2,
        ))
        assert result == 1
        assert log[0]["n_groups"] == 1

    def test_metrics_callback_fires(self):
        metrics_log = []
        asyncio.run(run_rl_loop(
            sample_fns=[_sample_ok()],
            train_fns=TrainStepFns(train_step=_make_train_step([])),
            prompt_groups_per_step=1,
            metrics_callback=lambda m: metrics_log.append(m),
        ))
        assert len(metrics_log) == 1
        assert metrics_log[0]["train/step"] == 1


# ---------------------------------------------------------------------------
# Weight sync coordination
# ---------------------------------------------------------------------------


class TestWeightSyncCoordination:
    def test_weight_sync_fn_called_at_window_boundary(self):
        log = []
        sync_log = []
        asyncio.run(run_rl_loop(
            sample_fns=[_sample_ok() for _ in range(8)],
            train_fns=TrainStepFns(train_step=_make_train_step(log)),
            prompt_groups_per_step=1,
            weight_sync_fn=lambda step: sync_log.append(step),
            weight_sync_interval=4,
        ))
        assert len(log) == 8
        assert sync_log == [4, 8]

    def test_weight_sync_interval_1_syncs_every_step(self):
        log = []
        sync_log = []
        asyncio.run(run_rl_loop(
            sample_fns=[_sample_ok() for _ in range(3)],
            train_fns=TrainStepFns(train_step=_make_train_step(log)),
            prompt_groups_per_step=1,
            weight_sync_fn=lambda step: sync_log.append(step),
            weight_sync_interval=1,
        ))
        assert len(log) == 3
        assert sync_log == [1, 2, 3]

    def test_weight_sync_interval_0_no_syncs(self):
        log = []
        sync_log = []
        asyncio.run(run_rl_loop(
            sample_fns=[_sample_ok() for _ in range(4)],
            train_fns=TrainStepFns(train_step=_make_train_step(log)),
            prompt_groups_per_step=1,
            weight_sync_fn=lambda step: sync_log.append(step),
            weight_sync_interval=0,
        ))
        assert len(log) == 4
        assert sync_log == []

    def test_no_weight_sync_fn_skips_syncs(self):
        log = []
        asyncio.run(run_rl_loop(
            sample_fns=[_sample_ok() for _ in range(4)],
            train_fns=TrainStepFns(train_step=_make_train_step(log)),
            prompt_groups_per_step=1,
            weight_sync_fn=None,
            weight_sync_interval=2,
        ))
        assert len(log) == 4

    def test_weight_sync_exception_propagates(self):
        def bad_sync(step):
            raise RuntimeError("hotload failed")

        with pytest.raises(RuntimeError, match="hotload failed"):
            asyncio.run(run_rl_loop(
                sample_fns=[_sample_ok()],
                train_fns=TrainStepFns(train_step=_make_train_step([])),
                prompt_groups_per_step=1,
                weight_sync_fn=bad_sync,
                weight_sync_interval=1,
            ))


# ---------------------------------------------------------------------------
# Pipeline overlap behavior
# ---------------------------------------------------------------------------


class TestPipelineOverlap:
    def test_overlap_within_window(self):
        """With weight_sync_interval > 1, sampling for step K+1 should
        overlap with training for step K within the same window."""
        timestamps = []

        def slow_train_step(step, groups, stats=None):
            timestamps.append(("train_start", step, time.monotonic()))
            time.sleep(0.1)
            timestamps.append(("train_end", step, time.monotonic()))
            return step + 1, {}

        async def timed_sample(idx):
            timestamps.append(("sample_start", idx, time.monotonic()))
            await asyncio.sleep(0.05)
            timestamps.append(("sample_end", idx, time.monotonic()))
            return _make_pg()

        asyncio.run(run_rl_loop(
            sample_fns=[timed_sample(i) for i in range(4)],
            train_fns=TrainStepFns(train_step=slow_train_step),
            prompt_groups_per_step=1,
            weight_sync_fn=lambda step: None,
            weight_sync_interval=4,
        ))

        train_starts = [t for name, _, t in timestamps if name == "train_start"]
        sample_starts = [t for name, _, t in timestamps if name == "sample_start"]
        train_ends = [t for name, _, t in timestamps if name == "train_end"]

        # With overlap, sample[1] should start before train[0] ends
        # (within the same window)
        if len(sample_starts) >= 2 and len(train_ends) >= 1:
            assert sample_starts[1] < train_ends[0], (
                "Sampling for step 2 should overlap with training for step 1"
            )

    def test_no_overlap_with_interval_1(self):
        """With weight_sync_interval=1, each step is its own window.
        No overlap should occur."""
        order = []

        def train_step(step, groups, stats=None):
            order.append(("train", step))
            return step + 1, {}

        async def sample(idx):
            order.append(("sample", idx))
            return _make_pg()

        asyncio.run(run_rl_loop(
            sample_fns=[sample(i) for i in range(3)],
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=1,
            weight_sync_fn=lambda step: order.append(("sync", step)),
            weight_sync_interval=1,
        ))

        # With interval=1, each window has 1 sample + 1 train + 1 sync
        # Sampling for step N+1 should NOT start before sync for step N
        for i in range(len(order) - 1):
            if order[i][0] == "sync":
                if i + 1 < len(order):
                    assert order[i + 1][0] == "sample", (
                        f"After sync, next action should be sample, got {order[i + 1]}"
                    )
