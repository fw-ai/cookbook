"""Unit tests for training.utils.rl.rollout.group_assembler.GroupAssembler.

Covers row-level fan-in semantics: full-group emit, partial-group on
quorum, single-run dropped under default advantages, custom advantage_fn
passthrough, row_meta propagation, and drain-at-shutdown.
"""

from __future__ import annotations

from training.utils.rl.rollout import (
    GroupAssembler,
    RolloutRun,
    RolloutSample,
)


def _sample(reward: float = 0.0) -> RolloutSample:
    return RolloutSample(
        tokens=[1, 2, 3],
        logprobs=[0.0, -0.1, -0.2],
        loss_mask=[0, 1, 1],
        reward=reward,
    )


def _run(reward: float = 0.0) -> RolloutRun:
    return RolloutRun(segments=[_sample(reward)])


def _passthrough(rewards):
    return list(rewards)


class TestSettlement:
    def test_full_group_emits_after_n_runs(self):
        asm = GroupAssembler(completions_per_prompt=3)
        for _ in range(3):
            asm.note_started("row-A", submit_version=0)
        for r in (0.5, 1.5, 2.5):
            res = asm.add_run("row-A", _run(reward=r))
        # Last add_run should return resolution; earlier two return None.
        assert res is not None
        assert res.pg is not None
        assert len(res.pg.data) == 3
        assert res.pg.rewards == [0.5, 1.5, 2.5]

    def test_intermediate_calls_return_none(self):
        asm = GroupAssembler(completions_per_prompt=3)
        for _ in range(3):
            asm.note_started("r", submit_version=0)
        assert asm.add_run("r", _run(0.0)) is None
        assert asm.add_run("r", _run(1.0)) is None
        # Only the third call settles.
        out = asm.add_run("r", _run(2.0))
        assert out is not None

    def test_drops_when_all_runs_fail(self):
        asm = GroupAssembler(completions_per_prompt=2)
        asm.note_started("r", submit_version=0)
        asm.note_started("r", submit_version=0)
        assert asm.note_dropped("r") is None
        out = asm.note_dropped("r")
        assert out is not None
        assert out.pg is None  # settled-but-empty

    def test_min_group_size_drops_below_threshold(self):
        asm = GroupAssembler(completions_per_prompt=4, min_group_size=3)
        for _ in range(4):
            asm.note_started("r", submit_version=0)
        asm.add_run("r", _run(0.0))
        asm.add_run("r", _run(1.0))
        asm.note_dropped("r")
        out = asm.note_dropped("r")
        # 2 surviving < min=3 -> settled-but-empty.
        assert out is not None
        assert out.pg is None

    def test_partial_group_emits_when_min_satisfied(self):
        asm = GroupAssembler(
            completions_per_prompt=4,
            min_group_size=2,
            advantage_fn=_passthrough,
        )
        for _ in range(4):
            asm.note_started("r", submit_version=0)
        asm.add_run("r", _run(0.5))
        asm.add_run("r", _run(1.5))
        asm.note_dropped("r")
        out = asm.note_dropped("r")
        assert out is not None
        assert out.pg is not None
        assert len(out.pg.data) == 2
        assert out.pg.rewards == [0.5, 1.5]


class TestVersionTracking:
    def test_min_submit_version_returned(self):
        asm = GroupAssembler(completions_per_prompt=2, advantage_fn=_passthrough)
        asm.note_started("r", submit_version=5)
        asm.note_started("r", submit_version=7)
        asm.add_run("r", _run(0.0))
        out = asm.add_run("r", _run(1.0))
        assert out is not None
        # Stale-as-its-stalest-run: oldest version dominates.
        assert out.min_submit_version == 5


class TestAdvantageFn:
    def test_default_grpo_drops_singleton(self):
        """Default GRPO z-score is NaN on N=1 -- the row drops."""
        asm = GroupAssembler(completions_per_prompt=1)
        asm.note_started("r", submit_version=0)
        out = asm.add_run("r", _run(1.0))
        assert out is not None
        assert out.pg is None

    def test_custom_passthrough_keeps_singleton(self):
        """REINFORCE-style custom advantage_fn is well-defined on N=1."""
        asm = GroupAssembler(completions_per_prompt=1, advantage_fn=_passthrough)
        asm.note_started("r", submit_version=3)
        out = asm.add_run("r", _run(0.7))
        assert out is not None
        assert out.pg is not None
        assert out.pg.rewards == [0.7]
        assert out.pg.advantages == [0.7]
        assert out.min_submit_version == 3


class TestRowMeta:
    def test_row_meta_threaded_to_prompt_group(self):
        asm = GroupAssembler(completions_per_prompt=2, advantage_fn=_passthrough)
        meta = {"row_id": "abc", "extra": 1}
        asm.note_started("r", submit_version=0, row_meta=meta)
        asm.note_started("r", submit_version=0)  # second call ignores row_meta
        asm.add_run("r", _run(0.0))
        out = asm.add_run("r", _run(1.0))
        assert out is not None
        assert out.pg is not None
        assert out.pg.row_meta == meta
        # Defensive copy.
        assert out.pg.row_meta is not meta


class TestMultipleRows:
    def test_independent_row_assembly(self):
        asm = GroupAssembler(completions_per_prompt=2, advantage_fn=_passthrough)
        asm.note_started("A", submit_version=0)
        asm.note_started("A", submit_version=0)
        asm.note_started("B", submit_version=0)
        asm.note_started("B", submit_version=0)

        # Interleave runs across rows.
        assert asm.add_run("A", _run(1.0)) is None
        assert asm.add_run("B", _run(2.0)) is None
        out_a = asm.add_run("A", _run(3.0))
        out_b = asm.add_run("B", _run(4.0))

        assert out_a is not None and out_a.pg is not None
        assert out_b is not None and out_b.pg is not None
        assert out_a.pg.rewards == [1.0, 3.0]
        assert out_b.pg.rewards == [2.0, 4.0]

    def test_pending_rows_count(self):
        asm = GroupAssembler(completions_per_prompt=2)
        asm.note_started("A", submit_version=0)
        asm.note_started("B", submit_version=0)
        assert asm.pending_rows() == 2
        asm.note_started("A", submit_version=0)
        assert asm.pending_rows() == 2  # still A and B
        asm.add_run("A", _run(0.0))
        asm.add_run("A", _run(0.0))
        assert asm.pending_rows() == 1


class TestDrain:
    def test_drain_emits_partial_groups_meeting_min(self):
        asm = GroupAssembler(
            completions_per_prompt=4,
            min_group_size=2,
            advantage_fn=_passthrough,
        )
        asm.note_started("A", submit_version=0)
        asm.note_started("A", submit_version=0)
        asm.note_started("A", submit_version=0)
        asm.note_started("A", submit_version=0)
        asm.add_run("A", _run(1.0))
        asm.add_run("A", _run(2.0))
        # Two more never landed; drain emits the partial.
        drained = asm.drain()
        assert len(drained) == 1
        assert drained[0].pg is not None
        assert drained[0].pg.rewards == [1.0, 2.0]

    def test_drain_skips_below_min(self):
        asm = GroupAssembler(
            completions_per_prompt=4,
            min_group_size=3,
        )
        asm.note_started("A", submit_version=0)
        asm.note_started("A", submit_version=0)
        asm.note_started("A", submit_version=0)
        asm.note_started("A", submit_version=0)
        asm.add_run("A", _run(0.0))
        # Only 1 run; below min=3.
        assert asm.drain() == []

    def test_multi_segment_run_counts_as_one_run(self):
        asm = GroupAssembler(completions_per_prompt=2, advantage_fn=_passthrough)
        asm.note_started("r", submit_version=0)
        asm.note_started("r", submit_version=0)

        first_run = RolloutRun(segments=[_sample(1.0), _sample(1.0)])
        assert asm.add_run("r", first_run) is None
        out = asm.add_run("r", _run(0.0))

        assert out is not None
        assert out.pg is not None
        assert out.pg.rewards == [1.0, 0.0]
        assert len(out.pg.data) == 3
        assert out.pg.advantages == [1.0, 1.0, 0.0]
