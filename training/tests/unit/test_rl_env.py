"""Unit tests for training.utils.rl.env types."""

from __future__ import annotations

import pytest

from training.utils.rl.env import (
    MessageEnv,
    MessageStepResult,
    Trajectory,
    Transition,
)


def _make_transition(
    *, reward: float = 1.0, done: bool = True, finish: str = "stop", valid: bool = True
) -> Transition:
    return Transition(
        prompt_tokens=[1, 2, 3],
        completion_tokens=[4, 5],
        completion_text="hi",
        inference_logprobs=[-0.1, -0.2],
        assistant_message={"role": "assistant", "content": "hi"},
        reward=reward,
        episode_done=done,
        finish_reason=finish,
        is_reward_valid=valid,
    )


class TestMessageStepResult:
    def test_defaults_are_empty(self):
        result = MessageStepResult(reward=1.0, episode_done=True)
        assert result.next_messages == []
        assert result.metrics == {}
        assert result.is_reward_valid is True


class TestTrajectory:
    def test_empty_trajectory_is_not_complete(self):
        traj = Trajectory()
        assert traj.transitions == []
        assert traj.is_complete is False
        assert traj.total_reward == 0.0

    def test_single_turn_trajectory(self):
        traj = Trajectory(transitions=[_make_transition(reward=0.7)])
        assert traj.is_complete is True
        assert traj.total_reward == pytest.approx(0.7)
        assert traj.any_truncated is False
        assert traj.all_rewards_valid is True

    def test_multi_turn_reward_sum(self):
        traj = Trajectory(
            transitions=[
                _make_transition(reward=0.5, done=False),
                _make_transition(reward=0.25, done=True),
            ]
        )
        assert traj.total_reward == pytest.approx(0.75)
        assert traj.is_complete is True

    def test_is_complete_false_when_last_turn_not_done(self):
        traj = Trajectory(transitions=[_make_transition(done=False)])
        assert traj.is_complete is False

    def test_any_truncated_picks_up_length_finish(self):
        traj = Trajectory(
            transitions=[
                _make_transition(done=False, finish="stop"),
                _make_transition(done=True, finish="length"),
            ]
        )
        assert traj.any_truncated is True

    def test_all_rewards_valid_false_on_invalid_turn(self):
        traj = Trajectory(transitions=[_make_transition(valid=False)])
        assert traj.all_rewards_valid is False

    def test_add_turn_reward_default_targets_last_turn(self):
        traj = Trajectory(
            transitions=[
                _make_transition(reward=0.1),
                _make_transition(reward=0.2),
            ]
        )
        traj.add_turn_reward(0.5)
        assert traj.transitions[-1].reward == pytest.approx(0.7)
        assert traj.transitions[0].reward == pytest.approx(0.1)

    def test_add_turn_reward_specific_index(self):
        traj = Trajectory(transitions=[_make_transition(reward=0.1), _make_transition(reward=0.2)])
        traj.add_turn_reward(0.3, turn_index=0)
        assert traj.transitions[0].reward == pytest.approx(0.4)
        assert traj.transitions[1].reward == pytest.approx(0.2)

    def test_add_turn_reward_on_empty_trajectory_raises(self):
        with pytest.raises(ValueError, match="empty trajectory"):
            Trajectory().add_turn_reward(1.0)


class TestMessageEnv:
    def test_abstract_methods_must_be_implemented(self):
        with pytest.raises(TypeError):
            MessageEnv()  # type: ignore[abstract]

    def test_concrete_subclass_is_instantiable(self):
        class _Impl(MessageEnv):
            async def initial_messages(self):
                return []

            async def step(self, assistant_message):
                return MessageStepResult(reward=0.0, episode_done=True)

        env = _Impl()
        assert isinstance(env, MessageEnv)
