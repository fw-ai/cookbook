"""Unit tests for training.utils.rl.rollout_runner."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from training.utils.rl.env import MessageEnv, MessageStepResult
from training.utils.rl.env_adapters import SingleTurnEnv
from training.utils.rl.rollout_runner import (
    run_env_group_to_trajectories,
    run_env_to_trajectory,
)


def _run(coro):
    return asyncio.run(coro)


@dataclass
class _FakeSampled:
    text: str
    full_tokens: list[int]
    prompt_len: int
    finish_reason: str = "stop"
    inference_logprobs: list[float] | None = None
    routing_matrices: Any | None = None


@dataclass
class _FakeGroupSampler:
    """Records calls; returns ``n`` fixed completions per sample_with_tokens."""

    completions: list[_FakeSampled] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    async def sample_with_tokens(self, *, messages, n, **kwargs):
        self.calls.append({"messages": list(messages), "n": n, "kwargs": kwargs})
        return self.completions[:n]


class TestRunEnvGroupToTrajectories:
    def test_single_turn_single_api_call_for_n_completions(self):
        sampler = _FakeGroupSampler(
            completions=[
                _FakeSampled(text="a", full_tokens=[1, 2, 10], prompt_len=2),
                _FakeSampled(text="b", full_tokens=[1, 2, 20], prompt_len=2),
                _FakeSampled(text="c", full_tokens=[1, 2, 30], prompt_len=2),
                _FakeSampled(text="d", full_tokens=[1, 2, 40], prompt_len=2),
            ]
        )

        def reward(completion, row):
            return 1.0 if completion == "a" else 0.0

        envs = [
            SingleTurnEnv(
                row={"messages": [{"role": "user", "content": "hi"}]}, reward_fn=reward,
            )
            for _ in range(4)
        ]

        trajs = _run(
            run_env_group_to_trajectories(
                envs, sampler, completions_per_prompt=4, max_turns=1,
            )
        )
        assert trajs is not None
        assert len(trajs) == 4
        # One single api call (n=4) -- matches sync loop's group batching.
        assert len(sampler.calls) == 1
        assert sampler.calls[0]["n"] == 4
        # Rewards reflect the per-completion reward_fn.
        assert trajs[0].total_reward == 1.0
        assert sum(t.total_reward for t in trajs) == 1.0
        for t in trajs:
            assert t.is_complete
            assert len(t.transitions) == 1

    def test_returns_none_when_sampler_returns_fewer_than_n(self):
        sampler = _FakeGroupSampler(
            completions=[_FakeSampled(text="a", full_tokens=[1, 2, 3], prompt_len=2)]
        )
        envs = [
            SingleTurnEnv(row={"messages": [{"role": "user", "content": "x"}]}, reward_fn=lambda c, r: 0.0)
            for _ in range(4)
        ]
        assert _run(
            run_env_group_to_trajectories(envs, sampler, completions_per_prompt=4)
        ) is None

    def test_rejects_mismatched_n_and_env_count(self):
        sampler = _FakeGroupSampler()
        envs = [
            SingleTurnEnv(row={"messages": [{"role": "user", "content": "x"}]}, reward_fn=lambda c, r: 0.0)
            for _ in range(3)
        ]
        import pytest

        with pytest.raises(ValueError, match="must match"):
            _run(run_env_group_to_trajectories(envs, sampler, completions_per_prompt=4))

    def test_multi_turn_fans_out_after_turn_one(self):
        """Turn 1 is batched, turn 2 issues per-env calls concurrently."""

        class _TwoTurnEnv(MessageEnv):
            def __init__(self, finish_on_second: bool):
                self.finish_on_second = finish_on_second
                self.turns = 0

            async def initial_messages(self):
                return [{"role": "user", "content": "go"}]

            async def step(self, msg):
                self.turns += 1
                if self.turns == 1 and self.finish_on_second:
                    return MessageStepResult(
                        reward=0.5, episode_done=False,
                        next_messages=[{"role": "user", "content": "again"}],
                    )
                return MessageStepResult(reward=1.0, episode_done=True)

        sampler = _FakeGroupSampler(
            completions=[
                _FakeSampled(text="x", full_tokens=[1, 2, 3], prompt_len=2),
                _FakeSampled(text="y", full_tokens=[1, 2, 4], prompt_len=2),
            ]
        )
        envs = [_TwoTurnEnv(finish_on_second=True), _TwoTurnEnv(finish_on_second=False)]

        trajs = _run(
            run_env_group_to_trajectories(
                envs, sampler, completions_per_prompt=2, max_turns=3,
            )
        )
        assert trajs is not None
        # Env 0 runs two turns (0.5 + 1.0), env 1 runs one (1.0).
        assert len(trajs[0].transitions) == 2
        assert len(trajs[1].transitions) == 1
        assert trajs[0].total_reward == 1.5
        # First call was n=2 (batched); second call was per-env with n=1.
        assert sampler.calls[0]["n"] == 2
        assert sampler.calls[1]["n"] == 1


class TestRunEnvToTrajectory:
    def test_runs_single_env_single_turn(self):
        sampler = _FakeGroupSampler(
            completions=[_FakeSampled(text="ok", full_tokens=[1, 2, 7], prompt_len=2)]
        )
        env = SingleTurnEnv(
            row={"messages": [{"role": "user", "content": "hi"}]},
            reward_fn=lambda c, r: 0.42,
        )
        traj = _run(run_env_to_trajectory(env, sampler, max_turns=1))
        assert traj is not None and len(traj.transitions) == 1
        assert traj.transitions[0].reward == 0.42

    def test_returns_none_on_empty_initial_messages(self):
        sampler = _FakeGroupSampler()
        env = SingleTurnEnv(row={}, reward_fn=lambda c, r: 0.0)
        assert _run(run_env_to_trajectory(env, sampler, max_turns=1)) is None
