"""Unit tests for training.utils.rl.async_train.run_async_rl_loop.

Covers the gate formula, version tagging, and metric emission paths.
No GPU / network / fireworks SDK required.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from training.utils.rl.async_train import run_async_rl_loop
from training.utils.rl.losses import PromptGroup
from training.utils.rl.train import TrainStepFns


def _run(coro):
    return asyncio.run(coro)


def _pg(reward: float = 0.0) -> PromptGroup:
    return PromptGroup(
        data=[],
        advantages=[0.0],
        ref_logprobs=None,
        prompt_len=0,
        rewards=[reward],
    )


def _noop_train_step(step, groups, extra):
    return step + 1, {}


class TestArgValidation:
    def test_rejects_invalid_prompt_groups_per_step(self):
        with pytest.raises(ValueError, match="prompt_groups_per_step"):
            _run(run_async_rl_loop(
                sample_fns=iter([]),
                train_fns=TrainStepFns(train_step=_noop_train_step),
                prompt_groups_per_step=0,
                max_head_offpolicy_versions=0,
            ))

    def test_rejects_negative_offpolicy(self):
        with pytest.raises(ValueError, match="max_head_offpolicy_versions"):
            _run(run_async_rl_loop(
                sample_fns=iter([]),
                train_fns=TrainStepFns(train_step=_noop_train_step),
                prompt_groups_per_step=1,
                max_head_offpolicy_versions=-1,
            ))

    def test_rejects_zero_weight_sync_interval(self):
        with pytest.raises(ValueError, match="weight_sync_interval"):
            _run(run_async_rl_loop(
                sample_fns=iter([]),
                train_fns=TrainStepFns(train_step=_noop_train_step),
                prompt_groups_per_step=1,
                max_head_offpolicy_versions=0,
                weight_sync_interval=0,
            ))

    def test_rejects_zero_max_concurrent(self):
        with pytest.raises(ValueError, match="max_concurrent"):
            _run(run_async_rl_loop(
                sample_fns=iter([]),
                train_fns=TrainStepFns(train_step=_noop_train_step),
                prompt_groups_per_step=1,
                max_head_offpolicy_versions=0,
                max_concurrent=0,
            ))


class TestHappyPath:
    def test_strict_onpolicy_runs_all_steps(self):
        """budget=0, gpb=2, 4 rows -> 2 training steps (requires version bumps)."""
        n_rows = 4
        calls = []

        def train_step(step, groups, extra):
            calls.append((step, len(groups), extra.get("async/version_offset_max")))
            return step + 1, {}

        async def one_sample(i):
            return _pg(float(i))

        sample_fns = (one_sample(i) for i in range(n_rows))

        result = _run(run_async_rl_loop(
            sample_fns=sample_fns,
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            max_head_offpolicy_versions=0,
            weight_sync_fn=lambda _step: None,
        ))
        assert result == 2
        assert len(calls) == 2
        assert all(offset == 0 for _, _, offset in calls)

    def test_sample_failures_skip_batch_not_charge_gate(self):
        """Sample-fn returning None counts as failure, doesn't block refill."""
        n_rows = 6

        async def one_sample(i):
            return None if i % 2 == 0 else _pg()

        sample_fns = (one_sample(i) for i in range(n_rows))
        calls = []

        def train_step(step, groups, extra):
            calls.append(extra["sample_fails"])
            return step + 1, {}

        _run(run_async_rl_loop(
            sample_fns=sample_fns,
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            max_head_offpolicy_versions=2,
        ))
        # 3 Nones were returned across the run -- at least one shows up in
        # the stats (monotonic counter).
        assert any(fails >= 1 for fails in calls)

    def test_filter_drops_exclude_from_buffer(self):
        """dynamic_filter_fn=False drops groups, doesn't train on them."""
        n_rows = 6

        async def one_sample(i):
            return _pg(reward=float(i))

        def filter_fn(pg):
            return pg.rewards[0] >= 3.0  # keep last 3 of 6

        trained_groups = []

        def train_step(step, groups, extra):
            trained_groups.extend(groups)
            return step + 1, {}

        _run(run_async_rl_loop(
            sample_fns=(one_sample(i) for i in range(n_rows)),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=3,
            max_head_offpolicy_versions=2,
            dynamic_filter_fn=filter_fn,
        ))
        # Exactly 3 groups survived filter -> one step of 3.
        assert len(trained_groups) == 3
        assert all(g.rewards[0] >= 3.0 for g in trained_groups)

    def test_iterator_exhausted_drops_partial_final_step(self):
        """With 5 rows and gpb=2, only 2 steps fire; the 5th row is dropped."""
        call_count = [0]

        def train_step(step, groups, extra):
            call_count[0] += 1
            assert len(groups) == 2
            return step + 1, {}

        async def one_sample(i):
            return _pg()

        _run(run_async_rl_loop(
            sample_fns=(one_sample(i) for i in range(5)),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            max_head_offpolicy_versions=0,
            weight_sync_fn=lambda _step: None,
        ))
        assert call_count[0] == 2


class TestVersionTracking:
    def test_weight_sync_increments_version(self):
        """Each weight_sync_fn call bumps current_version by 1."""
        sync_calls = []

        def weight_sync(step):
            sync_calls.append(step)

        async def one_sample(i):
            return _pg()

        _run(run_async_rl_loop(
            sample_fns=(one_sample(i) for i in range(6)),
            train_fns=TrainStepFns(train_step=_noop_train_step),
            prompt_groups_per_step=2,
            max_head_offpolicy_versions=0,
            weight_sync_fn=weight_sync,
            weight_sync_interval=1,
        ))
        assert sync_calls == [1, 2, 3]

    def test_weight_sync_interval_skips(self):
        """interval=2 -> sync only on even steps. Budget must be >= interval so
        the gate doesn't deadlock between syncs."""
        sync_calls = []

        def weight_sync(step):
            sync_calls.append(step)

        async def one_sample(i):
            return _pg()

        _run(run_async_rl_loop(
            sample_fns=(one_sample(i) for i in range(8)),
            train_fns=TrainStepFns(train_step=_noop_train_step),
            prompt_groups_per_step=2,
            max_head_offpolicy_versions=2,
            weight_sync_fn=weight_sync,
            weight_sync_interval=2,
        ))
        assert sync_calls == [2, 4]

    def test_version_offset_emitted(self):
        """Under strict on-policy, offset=0 for every step."""
        metric_snapshots = []

        def cb(m):
            metric_snapshots.append(m)

        async def one_sample(i):
            return _pg()

        _run(run_async_rl_loop(
            sample_fns=(one_sample(i) for i in range(4)),
            train_fns=TrainStepFns(train_step=_noop_train_step),
            prompt_groups_per_step=2,
            max_head_offpolicy_versions=0,
            weight_sync_fn=lambda _step: None,
            metrics_callback=cb,
        ))
        assert len(metric_snapshots) == 2
        for m in metric_snapshots:
            assert m["async/version_offset_mean"] == 0
            assert m["async/version_offset_max"] == 0
            assert m["async/version_offset_min"] == 0


class TestMaxConcurrent:
    def test_caps_in_flight_even_with_budget(self):
        """max_concurrent=2 even when the gate would allow more."""
        peak_in_flight = [0]

        def train_step(step, groups, extra):
            peak_in_flight[0] = max(peak_in_flight[0], extra["async/in_flight"])
            return step + 1, {}

        async def slow_sample(i):
            # Small sleep so multiple tasks stack up in in-flight.
            await asyncio.sleep(0.01)
            return _pg()

        # gpb=5, budget=10, max_concurrent=2 -> never more than 2 in flight.
        _run(run_async_rl_loop(
            sample_fns=(slow_sample(i) for i in range(10)),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=5,
            max_head_offpolicy_versions=10,
            max_concurrent=2,
        ))
        assert peak_in_flight[0] <= 2


class TestTaskExceptions:
    def test_exception_in_sample_counted_as_failure(self):
        """Task raising doesn't blow up the loop; counted as sample_fail."""

        async def sample(i):
            if i == 1:
                raise RuntimeError("boom")
            return _pg()

        stats = []

        def train_step(step, groups, extra):
            stats.append(extra["sample_fails"])
            return step + 1, {}

        _run(run_async_rl_loop(
            sample_fns=(sample(i) for i in range(4)),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=1,
            max_head_offpolicy_versions=2,
        ))
        # At least one sample_fail observed across the three training steps.
        assert any(s >= 1 for s in stats)
