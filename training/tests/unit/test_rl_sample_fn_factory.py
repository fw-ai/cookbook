"""Unit tests for training.utils.rl.sample_fn_factory."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from training.utils.rl.env import Trajectory, Transition
from training.utils.rl.sample_fn_factory import (
    build_sample_fn,
    validate_rollout_regime,
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
class _FakeSampler:
    completions: list[_FakeSampled] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    async def sample_with_tokens(self, *, messages, n, **kwargs):
        self.calls.append({"n": n, "messages": list(messages)})
        return self.completions[:n]


class TestValidateRolloutRegime:
    def test_requires_exactly_one(self):
        with pytest.raises(ValueError, match="one of"):
            validate_rollout_regime(reward_fn=None, env_builder=None, rollout_source=None)

    def test_rejects_multiple(self):
        with pytest.raises(ValueError, match="exactly one"):
            validate_rollout_regime(
                reward_fn=lambda c, r: 0.0,
                env_builder=lambda r: object(),
                rollout_source=None,
            )


class TestBuildSampleFn:
    def _make_sampler(self) -> _FakeSampler:
        return _FakeSampler(
            completions=[
                _FakeSampled(
                    text=f"c{i}",
                    full_tokens=[1, 2, 10 + i, 11 + i],
                    prompt_len=2,
                    inference_logprobs=[-0.1, -0.2],
                )
                for i in range(4)
            ]
        )

    def test_reward_fn_path_produces_prompt_group(self):
        sampler = self._make_sampler()
        sample_fn = build_sample_fn(
            reward_fn=lambda completion, row: 1.0 if "c0" in completion else 0.0,
            sampler=sampler,
            completions_per_prompt=4,
            sample_kwargs={},
            tokenizer=object(),
        )
        row = {"messages": [{"role": "user", "content": "hi"}]}
        pg = _run(sample_fn(row))
        assert pg is not None
        assert sum(pg.rewards) == pytest.approx(1.0)
        assert len(pg.advantages) == len(pg.data)

    def test_env_builder_path(self):
        from training.utils.rl.env_adapters import SingleTurnEnv

        sampler = self._make_sampler()

        def env_builder(row):
            return SingleTurnEnv(row=row, reward_fn=lambda c, r: 0.5)

        sample_fn = build_sample_fn(
            env_builder=env_builder,
            sampler=sampler,
            completions_per_prompt=4,
            sample_kwargs={},
            tokenizer=object(),
        )
        pg = _run(sample_fn({"messages": [{"role": "user", "content": "y"}]}))
        assert pg is not None
        assert pg.rewards == [0.5, 0.5, 0.5, 0.5]

    def test_rollout_source_path_with_provided_logprobs(self):
        async def rollout_source(row, *, n):
            return [
                Trajectory(
                    transitions=[
                        Transition(
                            prompt_tokens=[1, 2],
                            completion_tokens=[3, 4],
                            completion_text="x",
                            inference_logprobs=[-0.1, -0.2],
                            assistant_message={"role": "assistant", "content": "x"},
                            reward=float(i),
                            episode_done=True,
                        )
                    ]
                )
                for i in range(n)
            ]

        sample_fn = build_sample_fn(
            rollout_source=rollout_source,
            sampler=object(),  # unused on this path
            completions_per_prompt=3,
            sample_kwargs={},
            tokenizer=object(),
        )
        pg = _run(sample_fn({}))
        assert pg is not None
        assert pg.rewards == [0.0, 1.0, 2.0]

    def test_rollout_source_auto_prefill_when_logprobs_missing(self):
        async def rollout_source(row, *, n):
            return [
                Trajectory(
                    transitions=[
                        Transition(
                            prompt_tokens=[1, 2],
                            completion_tokens=[3, 4],
                            completion_text="x",
                            inference_logprobs=None,
                            assistant_message={"role": "assistant", "content": "x"},
                            reward=1.0,
                            episode_done=True,
                        )
                    ]
                )
                for _ in range(n)
            ]

        with patch(
            "training.utils.rl.rollout_builder.get_prefill_logprobs",
            return_value=[-0.1, -0.2, -0.3],
        ) as mock:
            sample_fn = build_sample_fn(
                rollout_source=rollout_source,
                sampler=object(),
                completions_per_prompt=2,
                sample_kwargs={},
                tokenizer=object(),
                inference_url="https://x",
                api_key="k",
                model="m",
            )
            pg = _run(sample_fn({}))
        assert pg is not None
        # get_prefill_logprobs should have been called once per trajectory.
        assert mock.call_count == 2

    def test_group_reward_fn_folds_into_final_reward(self):
        sampler = self._make_sampler()

        async def group_reward_fn(trajectories, row):
            return [0.1] * len(trajectories)

        sample_fn = build_sample_fn(
            reward_fn=lambda c, r: 0.0,
            sampler=sampler,
            completions_per_prompt=4,
            sample_kwargs={},
            tokenizer=object(),
            group_reward_fn=group_reward_fn,
        )
        row = {"messages": [{"role": "user", "content": "q"}]}
        pg = _run(sample_fn(row))
        assert pg is not None
        # All rewards should reflect the +0.1 delta.
        for r in pg.rewards:
            assert r == pytest.approx(0.1)
