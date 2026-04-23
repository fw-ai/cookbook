"""Unit tests for training.utils.rl.trajectory."""

from __future__ import annotations

import pytest

from training.utils.rl.trajectory import (
    CompletionSegment,
    Trajectory,
    trajectory_to_prompt_group,
)


def _seg(tokens, logprobs=None, version=0, finish_reason="stop", text="x"):
    return CompletionSegment(
        tokens=list(tokens),
        inference_logprobs=list(tokens) if logprobs is None else list(logprobs),
        version=version,
        finish_reason=finish_reason,
        text=text,
    )


def _traj(prompt_tokens, completions, rewards):
    return Trajectory(
        prompt_tokens=list(prompt_tokens),
        completions=completions,
        rewards=list(rewards),
    )


class TestValidation:
    def test_rejects_missing_rewards(self):
        traj = Trajectory(
            prompt_tokens=[1, 2],
            completions=[[_seg([3, 4])]],
            rewards=None,  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="rewards must be populated"):
            trajectory_to_prompt_group(traj)

    def test_rejects_rewards_length_mismatch(self):
        traj = _traj([1, 2], [[_seg([3, 4])], [_seg([5, 6])]], [0.0])
        with pytest.raises(ValueError, match="rewards length"):
            trajectory_to_prompt_group(traj)

    def test_rejects_empty_completion(self):
        traj = _traj([1, 2], [[]], [0.0])
        with pytest.raises(ValueError, match="no segments"):
            trajectory_to_prompt_group(traj)

    def test_rejects_logprob_length_mismatch(self):
        seg = CompletionSegment(
            tokens=[3, 4, 5],
            inference_logprobs=[-0.1, -0.2],  # len != len(tokens)
            version=0,
        )
        traj = _traj([1, 2], [[seg]], [0.0])
        with pytest.raises(ValueError, match="len\\(tokens\\)"):
            trajectory_to_prompt_group(traj)


class TestSingleTurnPacking:
    def test_single_completion_produces_one_datum(self):
        traj = _traj(
            prompt_tokens=[1, 2, 3],
            completions=[[_seg([10, 11], logprobs=[-0.1, -0.2])]],
            rewards=[1.0],
        )
        pg = trajectory_to_prompt_group(traj)
        assert pg is not None
        assert len(pg.data) == 1
        assert pg.prompt_len == 3
        assert pg.rewards == [1.0]
        assert pg.completion_lens == [2]
        assert len(pg.advantages) == 1

    def test_n_completions_produce_n_datums(self):
        traj = _traj(
            prompt_tokens=[1, 2],
            completions=[
                [_seg([10, 11], logprobs=[-0.1, -0.2])],
                [_seg([20, 21], logprobs=[-0.3, -0.4])],
            ],
            rewards=[1.0, 0.0],
        )
        pg = trajectory_to_prompt_group(traj)
        assert pg is not None
        assert len(pg.data) == 2
        assert pg.rewards == [1.0, 0.0]
        # Advantages should be centered (non-zero variance).
        assert sum(pg.advantages) == pytest.approx(0.0, abs=1e-5)

    def test_with_reference_produces_ref_data(self):
        traj = _traj(
            prompt_tokens=[1, 2],
            completions=[[_seg([10, 11], logprobs=[-0.1, -0.2])]],
            rewards=[0.5],
        )
        pg = trajectory_to_prompt_group(traj, with_reference=True)
        assert pg is not None
        assert len(pg.ref_data) == 1

    def test_without_reference_ref_data_empty(self):
        traj = _traj(
            prompt_tokens=[1, 2],
            completions=[[_seg([10, 11], logprobs=[-0.1, -0.2])]],
            rewards=[0.5],
        )
        pg = trajectory_to_prompt_group(traj, with_reference=False)
        assert pg is not None
        assert pg.ref_data == []

    def test_truncated_flag_from_last_segment(self):
        traj = _traj(
            prompt_tokens=[1, 2],
            completions=[
                [_seg([10, 11], logprobs=[-0.1, -0.2], finish_reason="length")],
                [_seg([20, 21], logprobs=[-0.3, -0.4], finish_reason="stop")],
            ],
            rewards=[0.0, 1.0],
        )
        pg = trajectory_to_prompt_group(traj)
        assert pg is not None
        assert pg.truncated == [True, False]

    def test_inf_logprob_alignment(self):
        """Completion logprobs land at the correct training-step index."""
        traj = _traj(
            prompt_tokens=[1, 2, 3],  # prompt_len = 3
            completions=[[_seg([10, 11], logprobs=[-0.5, -0.6])]],
            rewards=[1.0],
        )
        pg = trajectory_to_prompt_group(traj)
        assert pg is not None
        model_input_len = 3 + 2 - 1  # len(full) - 1
        aligned = pg.inf_logprobs[0]
        assert len(aligned) == model_input_len
        # response_start = prompt_len - 1 = 2, so first 2 are zeros, then lp.
        assert aligned[0] == 0.0
        assert aligned[1] == 0.0
        assert aligned[2] == -0.5
        assert aligned[3] == -0.6


class TestMultiTurnPacking:
    def test_segments_concatenate_into_single_datum(self):
        traj = _traj(
            prompt_tokens=[1, 2],
            completions=[[
                _seg([10, 11], logprobs=[-0.1, -0.2], version=0),
                _seg([20, 21, 22], logprobs=[-0.3, -0.4, -0.5], version=1),
            ]],
            rewards=[1.0],
        )
        pg = trajectory_to_prompt_group(traj)
        assert pg is not None
        assert len(pg.data) == 1  # single datum per completion regardless of turns.
        assert pg.completion_lens == [5]  # 2 + 3

    def test_per_segment_versions_preserved_on_trajectory(self):
        """PromptGroup drops per-segment metadata; Trajectory keeps it."""
        traj = _traj(
            prompt_tokens=[1],
            completions=[[
                _seg([10], logprobs=[-0.1], version=0),
                _seg([20], logprobs=[-0.2], version=5),
            ]],
            rewards=[1.0],
        )
        assert traj.completions[0][0].version == 0
        assert traj.completions[0][1].version == 5


class TestDegenerate:
    def test_empty_completion_tokens_drop_completion(self):
        traj = _traj(
            prompt_tokens=[1, 2],
            completions=[
                [_seg([], logprobs=[])],
                [_seg([10, 11], logprobs=[-0.1, -0.2])],
            ],
            rewards=[0.0, 1.0],
        )
        pg = trajectory_to_prompt_group(traj)
        assert pg is not None
        assert len(pg.data) == 1
        assert pg.rewards == [1.0]

    def test_all_completions_empty_returns_none(self):
        traj = _traj(
            prompt_tokens=[1, 2],
            completions=[[_seg([], logprobs=[])]],
            rewards=[0.0],
        )
        assert trajectory_to_prompt_group(traj) is None

    def test_persist_raw_copies_prompt_and_completions(self):
        traj = Trajectory(
            prompt_tokens=[1, 2],
            completions=[[_seg([10, 11], logprobs=[-0.1, -0.2], text="hi")]],
            rewards=[1.0],
            prompt_messages=[{"role": "user", "content": "q"}],
            row_meta={"ground_truth": "gt"},
        )
        pg = trajectory_to_prompt_group(traj, persist_raw=True)
        assert pg is not None
        assert pg.prompt == [{"role": "user", "content": "q"}]
        assert pg.completions == ["hi"]
        assert pg.row_meta == {"ground_truth": "gt"}

    def test_persist_raw_false_drops_raw(self):
        traj = Trajectory(
            prompt_tokens=[1, 2],
            completions=[[_seg([10, 11], logprobs=[-0.1, -0.2], text="hi")]],
            rewards=[1.0],
            prompt_messages=[{"role": "user", "content": "q"}],
            row_meta={"ground_truth": "gt"},
        )
        pg = trajectory_to_prompt_group(traj, persist_raw=False)
        assert pg is not None
        assert pg.prompt is None
        assert pg.completions is None
        assert pg.row_meta is None
