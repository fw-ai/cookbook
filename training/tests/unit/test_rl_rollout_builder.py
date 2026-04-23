"""Unit tests for training.utils.rl.rollout_builder."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from training.utils.rl.env import Trajectory, Transition
from training.utils.rl.rollout_builder import trajectories_to_prompt_group


def _make_transition(
    *,
    prompt_tokens=(1, 2, 3),
    completion_tokens=(10, 11),
    reward=1.0,
    inference_logprobs=(-0.1, -0.2),
    finish_reason="stop",
):
    return Transition(
        prompt_tokens=list(prompt_tokens),
        completion_tokens=list(completion_tokens),
        completion_text="x",
        inference_logprobs=list(inference_logprobs) if inference_logprobs is not None else None,
        assistant_message={"role": "assistant", "content": "x"},
        reward=float(reward),
        episode_done=True,
        finish_reason=finish_reason,
    )


def _make_trajectory(*transitions):
    return Trajectory(transitions=list(transitions))


class TestTrajectoriesToPromptGroup:
    def test_empty_input_returns_none(self):
        assert trajectories_to_prompt_group([], need_reference=False, tokenizer=None) is None

    def test_single_turn_group_packs_prompt_group(self):
        trajs = [
            _make_trajectory(_make_transition(reward=1.0)),
            _make_trajectory(_make_transition(reward=0.0)),
        ]
        pg = trajectories_to_prompt_group(trajs, need_reference=False, tokenizer=None)
        assert pg is not None
        assert len(pg.data) == 2
        assert pg.rewards == [1.0, 0.0]
        assert pg.ref_data == []
        # Advantages centered.
        assert sum(pg.advantages) == pytest.approx(0.0, abs=1e-5)
        # Completion-length accounting matches the transition.
        assert pg.completion_lens == [2, 2]
        assert pg.truncated == [False, False]

    def test_need_reference_builds_reference_datums(self):
        trajs = [_make_trajectory(_make_transition())]
        pg = trajectories_to_prompt_group(trajs, need_reference=True, tokenizer=None)
        assert pg is not None
        assert len(pg.ref_data) == len(pg.data) == 1

    def test_truncated_flag_propagates(self):
        trajs = [_make_trajectory(_make_transition(finish_reason="length"))]
        pg = trajectories_to_prompt_group(trajs, need_reference=False, tokenizer=None)
        assert pg is not None
        assert pg.truncated == [True]

    def test_drops_too_short_trajectories(self):
        """Trajectories with <2 total tokens are unusable (need a shift-pair)."""
        t = Transition(
            prompt_tokens=[],
            completion_tokens=[42],
            completion_text="",
            inference_logprobs=[-0.5],
            assistant_message={"role": "assistant", "content": ""},
            reward=1.0,
            episode_done=True,
        )
        trajs = [_make_trajectory(t)]
        assert trajectories_to_prompt_group(
            trajs, need_reference=False, tokenizer=None,
        ) is None

    def test_auto_prefill_when_logprobs_missing(self):
        trajs = [_make_trajectory(_make_transition(inference_logprobs=None))]
        with patch(
            "training.utils.rl.rollout_builder.get_prefill_logprobs",
            return_value=[-0.5, -0.6, -0.7],
        ) as mock:
            pg = trajectories_to_prompt_group(
                trajs, need_reference=False, tokenizer=None,
                inference_url="https://x", api_key="k", model="m",
            )
        assert pg is not None
        assert mock.call_count == 1

    def test_rejects_missing_logprobs_without_inference_url(self):
        trajs = [_make_trajectory(_make_transition(inference_logprobs=None))]
        with pytest.raises(RuntimeError, match="prefill"):
            trajectories_to_prompt_group(
                trajs, need_reference=False, tokenizer=None,
            )
