"""Unit tests for training.utils.rl.async_train.run_async_rl_loop.

Exercises the per-sample API: rows fan out to N samples, the
GroupAssembler joins them by row id, and the loop trains on assembled
PromptGroups.  No GPU / network / fireworks SDK required.
"""

from __future__ import annotations

import asyncio

import pytest

from training.utils.rl.async_train import (
    RowRequest,
    _StalenessController,
    run_async_rl_loop,
)
from training.utils.rl.rollout import RolloutSample
from training.utils.rl.train import TrainStepFns


def _run(coro):
    return asyncio.run(coro)


def _sample(reward: float = 0.0) -> RolloutSample:
    """Minimal valid RolloutSample for loop-level tests."""
    return RolloutSample(
        tokens=[1, 2],
        logprobs=[0.0, -0.1],
        loss_mask=[0, 1],
        reward=reward,
    )


def _passthrough_advantages(rewards):
    """REINFORCE-style: raw reward as advantage.  Safe on N=1 groups."""
    return list(rewards)


def _row(
    row_id: int,
    *,
    reward: float = 0.0,
    fail_indices: tuple[int, ...] = (),
    on_resolved=None,
    row_meta=None,
) -> RowRequest:
    """Build a RowRequest whose sample_factory returns a fresh sample
    per sub_index (or ``None`` for indices in ``fail_indices``)."""

    async def factory(sub_index: int):
        if sub_index in fail_indices:
            return None
        return _sample(reward=reward)

    return RowRequest(
        row_id=row_id,
        sample_factory=factory,
        row_meta=row_meta,
        on_resolved=on_resolved,
    )


def _noop_train_step(step, groups, extra):
    return step + 1, {}


# Sensible defaults for tests that don't care about the new knobs.
_DEFAULTS = dict(
    completions_per_prompt=1,
    advantage_fn=_passthrough_advantages,
)


class TestArgValidation:
    def test_rejects_invalid_prompt_groups_per_step(self):
        with pytest.raises(ValueError, match="prompt_groups_per_step"):
            _run(run_async_rl_loop(
                rows=iter([]),
                train_fns=TrainStepFns(train_step=_noop_train_step),
                prompt_groups_per_step=0,
                max_head_offpolicy_versions=0,
                **_DEFAULTS,
            ))

    def test_rejects_invalid_completions_per_prompt(self):
        with pytest.raises(ValueError, match="completions_per_prompt"):
            _run(run_async_rl_loop(
                rows=iter([]),
                train_fns=TrainStepFns(train_step=_noop_train_step),
                prompt_groups_per_step=1,
                max_head_offpolicy_versions=0,
                completions_per_prompt=0,
                advantage_fn=_passthrough_advantages,
            ))

    def test_rejects_min_group_size_above_completions(self):
        with pytest.raises(ValueError, match="min_group_size"):
            _run(run_async_rl_loop(
                rows=iter([]),
                train_fns=TrainStepFns(train_step=_noop_train_step),
                prompt_groups_per_step=1,
                max_head_offpolicy_versions=0,
                completions_per_prompt=2,
                min_group_size=3,
                advantage_fn=_passthrough_advantages,
            ))

    def test_rejects_negative_offpolicy(self):
        with pytest.raises(ValueError, match="max_head_offpolicy_versions"):
            _run(run_async_rl_loop(
                rows=iter([]),
                train_fns=TrainStepFns(train_step=_noop_train_step),
                prompt_groups_per_step=1,
                max_head_offpolicy_versions=-1,
                **_DEFAULTS,
            ))

    def test_rejects_zero_weight_sync_interval(self):
        with pytest.raises(ValueError, match="weight_sync_interval"):
            _run(run_async_rl_loop(
                rows=iter([]),
                train_fns=TrainStepFns(train_step=_noop_train_step),
                prompt_groups_per_step=1,
                max_head_offpolicy_versions=0,
                weight_sync_interval=0,
                **_DEFAULTS,
            ))

    def test_rejects_zero_max_concurrent(self):
        with pytest.raises(ValueError, match="max_concurrent"):
            _run(run_async_rl_loop(
                rows=iter([]),
                train_fns=TrainStepFns(train_step=_noop_train_step),
                prompt_groups_per_step=1,
                max_head_offpolicy_versions=0,
                max_concurrent=0,
                **_DEFAULTS,
            ))

    def test_rejects_sync_interval_gt_offpolicy_plus_one(self):
        """``weight_sync_interval > 1`` + ``max_head_offpolicy_versions == 0``
        is mathematically guaranteed to stall."""
        with pytest.raises(ValueError, match="max_head_offpolicy_versions"):
            _run(run_async_rl_loop(
                rows=iter([]),
                train_fns=TrainStepFns(train_step=_noop_train_step),
                prompt_groups_per_step=1,
                max_head_offpolicy_versions=0,
                weight_sync_interval=5,
                **_DEFAULTS,
            ))

    def test_accepts_balanced_sync_interval_and_offpolicy(self):
        result = _run(run_async_rl_loop(
            rows=iter([]),
            train_fns=TrainStepFns(train_step=_noop_train_step),
            prompt_groups_per_step=1,
            max_head_offpolicy_versions=2,
            weight_sync_interval=3,
            **_DEFAULTS,
        ))
        assert isinstance(result, int)


class TestStalenessController:
    def test_capacity_matches_areal_formula(self):
        # cpp=1 collapses sample-level math to batch/group units, so the
        # values below double as a reference for the AReaL row-level form.
        ctl = _StalenessController(
            batch_size_samples=4,
            completions_per_prompt=1,
            max_staleness=1,
            max_concurrent_samples=3,
        )
        assert ctl.capacity() == 3

        ctl.submit()
        ctl.submit()
        assert ctl.capacity() == 1

        ctl.accept()
        ctl.accept()
        assert ctl.capacity() == 3

        ctl.advance_version()
        assert ctl.capacity() == 3

    def test_reject_frees_running_capacity_without_charging_accepted(self):
        ctl = _StalenessController(
            batch_size_samples=2,
            completions_per_prompt=1,
            max_staleness=0,
            max_concurrent_samples=2,
        )
        ctl.submit()
        ctl.submit()
        assert ctl.capacity() == 0

        ctl.reject("none")
        assert ctl.accepted_samples == 0
        assert ctl.sample_fails == 1
        assert ctl.capacity() == 1


class TestHappyPath:
    def test_strict_onpolicy_runs_all_steps(self):
        """budget=0, gpb=2, 4 rows -> 2 training steps."""
        n_rows = 4
        calls = []

        def train_step(step, groups, extra):
            calls.append((step, len(groups), extra.get("async/version_offset_max")))
            return step + 1, {}

        result = _run(run_async_rl_loop(
            rows=(_row(i, reward=float(i)) for i in range(n_rows)),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            max_head_offpolicy_versions=0,
            weight_sync_fn=lambda _step: None,
            **_DEFAULTS,
        ))
        assert result == 2
        assert len(calls) == 2
        assert all(offset == 0 for _, _, offset in calls)

    def test_sample_failures_skip_batch_not_charge_gate(self):
        """Sample factory returning None drops the row from training."""
        n_rows = 6
        calls = []

        def train_step(step, groups, extra):
            calls.append(extra["sample_fails"])
            return step + 1, {}

        # Even rows fail (single sample == row drops).
        rows = (
            _row(i, fail_indices=(0,) if i % 2 == 0 else ())
            for i in range(n_rows)
        )

        _run(run_async_rl_loop(
            rows=rows,
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            max_head_offpolicy_versions=2,
            **_DEFAULTS,
        ))
        assert any(fails >= 1 for fails in calls)

    def test_filter_drops_exclude_from_buffer(self):
        """dynamic_filter_fn=False drops groups, doesn't train on them."""
        n_rows = 6

        def filter_fn(pg):
            return pg.rewards[0] >= 3.0

        trained_groups = []

        def train_step(step, groups, extra):
            trained_groups.extend(groups)
            return step + 1, {}

        _run(run_async_rl_loop(
            rows=(_row(i, reward=float(i)) for i in range(n_rows)),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=3,
            max_head_offpolicy_versions=2,
            dynamic_filter_fn=filter_fn,
            **_DEFAULTS,
        ))
        assert len(trained_groups) == 3
        assert all(g.rewards[0] >= 3.0 for g in trained_groups)

    def test_iterator_exhausted_flushes_partial_final_step(self):
        sizes: list[int] = []

        def train_step(step, groups, extra):
            sizes.append(len(groups))
            return step + 1, {}

        final_step = _run(run_async_rl_loop(
            rows=(_row(i) for i in range(5)),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            max_head_offpolicy_versions=0,
            weight_sync_fn=lambda _step: None,
            **_DEFAULTS,
        ))
        assert sizes == [2, 2, 1]
        assert final_step == 3

    def test_partial_only_buffer_still_trains(self):
        sizes: list[int] = []

        def train_step(step, groups, extra):
            sizes.append(len(groups))
            return step + 1, {}

        _run(run_async_rl_loop(
            rows=(_row(i) for i in range(2)),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=4,
            max_head_offpolicy_versions=0,
            weight_sync_fn=lambda _step: None,
            **_DEFAULTS,
        ))
        assert sizes == [2]

    def test_stalled_gate_does_not_consume_remaining_iterator(self):
        """A stalled strict-on-policy loop must not drain a lazy row iterator
        just to report how many rows remain."""

        class Rows:
            def __init__(self):
                self.consumed = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.consumed += 1
                if self.consumed > 4:
                    raise StopIteration
                return _row(self.consumed)

        rows = Rows()
        _run(run_async_rl_loop(
            rows=rows,
            train_fns=TrainStepFns(train_step=_noop_train_step),
            prompt_groups_per_step=2,
            max_head_offpolicy_versions=0,
            weight_sync_fn=None,
            **_DEFAULTS,
        ))
        assert rows.consumed == 2


class TestVersionTracking:
    def test_weight_sync_increments_version(self):
        sync_calls = []

        def weight_sync(step):
            sync_calls.append(step)

        _run(run_async_rl_loop(
            rows=(_row(i) for i in range(6)),
            train_fns=TrainStepFns(train_step=_noop_train_step),
            prompt_groups_per_step=2,
            max_head_offpolicy_versions=0,
            weight_sync_fn=weight_sync,
            weight_sync_interval=1,
            **_DEFAULTS,
        ))
        assert sync_calls == [1, 2, 3]

    def test_weight_sync_interval_skips(self):
        sync_calls = []

        def weight_sync(step):
            sync_calls.append(step)

        _run(run_async_rl_loop(
            rows=(_row(i) for i in range(8)),
            train_fns=TrainStepFns(train_step=_noop_train_step),
            prompt_groups_per_step=2,
            max_head_offpolicy_versions=2,
            weight_sync_fn=weight_sync,
            weight_sync_interval=2,
            **_DEFAULTS,
        ))
        assert sync_calls == [2, 4]

    def test_version_offset_emitted(self):
        metric_snapshots = []

        def train_step(step, groups, extra):
            metric_snapshots.append(dict(extra))
            return step + 1, {}

        _run(run_async_rl_loop(
            rows=(_row(i) for i in range(4)),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            max_head_offpolicy_versions=0,
            weight_sync_fn=lambda _step: None,
            **_DEFAULTS,
        ))
        assert len(metric_snapshots) == 2
        for m in metric_snapshots:
            assert m["async/version_offset_mean"] == 0
            assert m["async/version_offset_max"] == 0
            assert m["async/version_offset_min"] == 0


class TestMaxConcurrent:
    def test_caps_in_flight_even_with_budget(self):
        """max_concurrent counts in-flight SAMPLES; with cpp=1 the row
        and sample bookkeeping coincide."""
        peak_in_flight = [0]

        def train_step(step, groups, extra):
            peak_in_flight[0] = max(peak_in_flight[0], extra["async/in_flight"])
            return step + 1, {}

        async def slow_factory(_sub):
            await asyncio.sleep(0.01)
            return _sample()

        rows = (
            RowRequest(row_id=i, sample_factory=slow_factory)
            for i in range(10)
        )

        _run(run_async_rl_loop(
            rows=rows,
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=5,
            max_head_offpolicy_versions=10,
            max_concurrent=2,
            **_DEFAULTS,
        ))
        assert peak_in_flight[0] <= 2

    def test_caps_in_flight_at_sample_level_with_cpp_gt_1(self):
        """cpp=4, max_concurrent=8: never more than 8 samples in flight.

        Pins the unit semantic for max_concurrent: it counts samples
        (LLM calls), not rows.  Each row submits cpp=4 sample tasks
        atomically, so under an 8-sample cap at most 2 rows are admitted.
        """
        peak_samples = [0]

        async def slow_factory(_sub):
            await asyncio.sleep(0.005)
            return _sample()

        def train_step(step, groups, extra):
            peak_samples[0] = max(peak_samples[0], extra["async/in_flight"])
            return step + 1, {}

        rows = (
            RowRequest(row_id=i, sample_factory=slow_factory)
            for i in range(6)
        )

        _run(run_async_rl_loop(
            rows=rows,
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            max_head_offpolicy_versions=10,
            completions_per_prompt=4,
            max_concurrent=8,
            weight_sync_fn=lambda _step: None,
            advantage_fn=_passthrough_advantages,
        ))
        assert peak_samples[0] <= 8

    def test_rejects_max_concurrent_below_cpp(self):
        """max_concurrent < cpp would deadlock the gate (can't admit a row)."""
        with pytest.raises(ValueError, match="max_concurrent"):
            _run(run_async_rl_loop(
                rows=iter([]),
                train_fns=TrainStepFns(train_step=_noop_train_step),
                prompt_groups_per_step=1,
                max_head_offpolicy_versions=2,
                completions_per_prompt=4,
                max_concurrent=3,
                advantage_fn=_passthrough_advantages,
            ))


class TestTaskExceptions:
    def test_exception_in_sample_propagates(self):
        """Sample-task exceptions abort the run rather than getting silently
        counted as ``sample_fails``."""

        async def factory(sub_index):
            raise RuntimeError("boom")

        bad_row = RowRequest(row_id=99, sample_factory=factory)

        with pytest.raises(RuntimeError, match="boom"):
            _run(run_async_rl_loop(
                rows=iter([_row(0), bad_row, _row(2), _row(3)]),
                train_fns=TrainStepFns(train_step=_noop_train_step),
                prompt_groups_per_step=1,
                max_head_offpolicy_versions=2,
                **_DEFAULTS,
            ))


class TestRowResolutionHooks:
    def test_on_resolved_drives_external_cursor(self):
        cursor = {"value": 0}

        def make_row(i):
            return _row(i, on_resolved=lambda _reason, i=i: cursor.update(value=i + 1))

        final_step, final = _run(run_async_rl_loop(
            rows=(make_row(i) for i in range(3)),
            train_fns=TrainStepFns(train_step=_noop_train_step),
            prompt_groups_per_step=1,
            max_head_offpolicy_versions=0,
            weight_sync_fn=lambda _step: None,
            resolved_rows_fn=lambda: cursor["value"],
            return_final_stats=True,
            **_DEFAULTS,
        ))
        assert final_step == 3
        assert final["resolved_rows"] == 3

    def test_final_stats_include_tail_sample_failures(self):
        """Rows where the only sample fails count as sample_fails in stats."""

        rows = (
            _row(i, fail_indices=(0,) if i >= 3 else ())
            for i in range(5)
        )

        final_step, final = _run(run_async_rl_loop(
            rows=rows,
            train_fns=TrainStepFns(train_step=_noop_train_step),
            prompt_groups_per_step=2,
            max_head_offpolicy_versions=0,
            weight_sync_fn=lambda _step: None,
            return_final_stats=True,
            **_DEFAULTS,
        ))
        # Rows 0,1 train as step 1; row 2 trains as a partial step 2.
        assert final_step == 2
        assert final["sample_fails"] == 2
        assert final["total_accepted"] == 3
        assert final["resolved_rows"] == 5

    def test_filter_drops_recorded_in_final_stats(self):
        def filter_fn(_pg):
            return False  # drop everything

        final_step, final = _run(run_async_rl_loop(
            rows=(_row(i) for i in range(3)),
            train_fns=TrainStepFns(train_step=_noop_train_step),
            prompt_groups_per_step=1,
            max_head_offpolicy_versions=0,
            dynamic_filter_fn=filter_fn,
            return_final_stats=True,
            **_DEFAULTS,
        ))
        assert final_step == 0
        assert final["filter_drops"] == 3
        assert final["sample_fails"] == 0
        assert final["resolved_rows"] == 3

    def test_resolved_rows_offset_applies_to_metrics(self):
        snapshots = []

        def train_step(step, groups, extra):
            snapshots.append(dict(extra))
            return step + 1, {}

        final_step, final = _run(run_async_rl_loop(
            rows=(_row(i) for i in range(2)),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=1,
            max_head_offpolicy_versions=1,
            resolved_rows_offset=10,
            return_final_stats=True,
            **_DEFAULTS,
        ))
        assert final_step == 2
        assert [m["resolved_rows"] for m in snapshots] == [12, 12]
        assert final["resolved_rows"] == 12


class TestPerSampleFanout:
    def test_grpo_fanout_assembles_n_samples_per_row(self):
        """completions_per_prompt=4 with default GRPO advantages: every row
        produces a 4-sample group."""
        trained_groups = []

        def train_step(step, groups, extra):
            trained_groups.extend(groups)
            return step + 1, {}

        # Two rows -> 1 step at gpb=2.  Each row fans out to 4 samples.
        _run(run_async_rl_loop(
            rows=iter([_row(0, reward=1.0), _row(1, reward=2.0)]),
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            max_head_offpolicy_versions=0,
            completions_per_prompt=4,
            weight_sync_fn=lambda _step: None,
        ))
        assert len(trained_groups) == 2
        # Each PromptGroup has 4 datums (one per sample).
        assert all(len(g.data) == 4 for g in trained_groups)

    def test_partial_group_emits_when_min_group_size_satisfied(self):
        """If 1 of 4 samples fails but min_group_size=2, the row still emits."""
        trained_groups = []

        def train_step(step, groups, extra):
            trained_groups.extend(groups)
            return step + 1, {}

        # Fail 2 of 4 samples per row -> 2 surviving samples meets min=2.
        rows = iter([
            _row(0, reward=0.5, fail_indices=(0, 2)),
            _row(1, reward=1.5, fail_indices=(1, 3)),
        ])
        _run(run_async_rl_loop(
            rows=rows,
            train_fns=TrainStepFns(train_step=train_step),
            prompt_groups_per_step=2,
            max_head_offpolicy_versions=0,
            completions_per_prompt=4,
            min_group_size=2,
            weight_sync_fn=lambda _step: None,
        ))
        assert len(trained_groups) == 2
        assert all(len(g.data) == 2 for g in trained_groups)

    def test_row_fully_dropped_when_below_min_group_size(self):
        """When more samples fail than min_group_size allows, the row drops
        entirely and counts as sample_fail."""
        rows = iter([
            _row(0, fail_indices=(0, 1, 2, 3)),  # all fail
            _row(1),                              # all succeed
            _row(2),
        ])
        final_step, final = _run(run_async_rl_loop(
            rows=rows,
            train_fns=TrainStepFns(train_step=_noop_train_step),
            prompt_groups_per_step=2,
            max_head_offpolicy_versions=0,
            completions_per_prompt=4,
            min_group_size=2,
            weight_sync_fn=lambda _step: None,
            return_final_stats=True,
        ))
        # Only 2 surviving rows -> 1 step at gpb=2.
        assert final_step == 1
        assert final["sample_fails"] == 1
        assert final["total_accepted"] == 2
