"""Unit tests for training.utils.rl.text_rollout.

Token-native only: every turn carries ``token_ids`` and every assistant
turn carries ``logprobs``.  Re-tokenizing text post-hoc silently
misaligns the loss mask and inference logprobs (slime/AReaL refuse to
do it; this packer follows the same rule), so the path doesn't exist
here either.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from training.utils.rl.rollout import Rollout
from training.utils.rl.rollout_service import RolloutPayload, TurnRecord
from training.utils.rl import text_rollout as tr


@dataclass
class _FakeCtx:
    completions_per_prompt: int = 2
    sample_kwargs: dict = field(default_factory=dict)
    inference_url: str = "https://example.invalid/inference/v1"
    api_key: str = "sk-test"
    model: str = "accounts/test/models/x"
    _version: int = 3

    def current_version(self) -> int:
        return self._version


# ---------------------------------------------------------------------------
# Token-native packing
# ---------------------------------------------------------------------------


class TestTokenNative:
    @pytest.mark.asyncio
    async def test_single_turn_assistant(self):
        payload = RolloutPayload(
            turns=[
                TurnRecord(role="user", text="hi", token_ids=[1, 2, 3]),
                TurnRecord(
                    role="assistant", text="ok",
                    token_ids=[4, 5], logprobs=[-0.1, -0.2],
                    finish_reason="stop",
                ),
            ],
            total_reward=0.75,
        )
        sample = await tr.pack_payload_to_sample(
            payload, ctx=_FakeCtx(), version=7,
        )
        assert sample.tokens == [1, 2, 3, 4, 5]
        assert sample.logprobs == [0.0, 0.0, 0.0, -0.1, -0.2]
        assert sample.loss_mask == [0, 0, 0, 1, 1]
        assert sample.reward == 0.75
        assert sample.versions == [7] * 5
        assert sample.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_multi_turn_interleaved_mask(self):
        payload = RolloutPayload(
            turns=[
                TurnRecord(role="user", text="q1", token_ids=[1, 2]),
                TurnRecord(
                    role="assistant", text="a1",
                    token_ids=[3, 4], logprobs=[-0.1, -0.2],
                ),
                TurnRecord(role="tool", text="obs", token_ids=[5, 6, 7]),
                TurnRecord(
                    role="assistant", text="a2",
                    token_ids=[8, 9], logprobs=[-0.3, -0.4],
                    finish_reason="stop",
                ),
            ],
            total_reward=1.0,
        )
        sample = await tr.pack_payload_to_sample(
            payload, ctx=_FakeCtx(), version=0,
        )
        assert sample.tokens == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert sample.loss_mask == [0, 0, 1, 1, 0, 0, 0, 1, 1]
        assert sample.logprobs == [0.0, 0.0, -0.1, -0.2, 0.0, 0.0, 0.0, -0.3, -0.4]

    @pytest.mark.asyncio
    async def test_assistant_missing_logprobs_rejected(self):
        payload = RolloutPayload(
            turns=[
                TurnRecord(role="user", token_ids=[1]),
                TurnRecord(role="assistant", token_ids=[2, 3], logprobs=None),
            ],
            total_reward=0.0,
        )
        with pytest.raises(tr._PackError, match="per-token logprobs"):
            await tr.pack_payload_to_sample(
                payload, ctx=_FakeCtx(), version=0,
            )

    @pytest.mark.asyncio
    async def test_text_only_turn_rejected(self):
        """A turn that lacks ``token_ids`` is a hard error: the packer
        refuses to re-tokenize text after the fact."""
        payload = RolloutPayload(
            turns=[
                TurnRecord(role="user", token_ids=[1, 2]),
                TurnRecord(role="assistant", text="only text"),
            ],
            total_reward=0.5,
        )
        with pytest.raises(tr._PackError, match="missing token_ids"):
            await tr.pack_payload_to_sample(
                payload, ctx=_FakeCtx(), version=0,
            )


# ---------------------------------------------------------------------------
# Reward plumbing
# ---------------------------------------------------------------------------


def _toy_payload(reward):
    return RolloutPayload(
        turns=[
            TurnRecord(role="user", token_ids=[1]),
            TurnRecord(role="assistant", token_ids=[2], logprobs=[-0.1]),
        ],
        total_reward=reward,
    )


class TestReward:
    @pytest.mark.asyncio
    async def test_server_reward_wins(self):
        async def reward_fn(row, payload):
            return 99.0  # should not be called

        sample = await tr.pack_payload_to_sample(
            _toy_payload(0.5), ctx=_FakeCtx(), version=0, reward_fn=reward_fn,
        )
        assert sample.reward == 0.5

    @pytest.mark.asyncio
    async def test_reward_fn_called_when_unset(self):
        captured: dict = {}

        async def reward_fn(row, payload):
            captured["row"] = row
            captured["payload"] = payload
            return 0.8

        payload = _toy_payload(None)
        sample = await tr.pack_payload_to_sample(
            payload, ctx=_FakeCtx(), version=0, reward_fn=reward_fn,
            row={"id": "r1"},
        )
        assert sample.reward == 0.8
        assert captured["row"] == {"id": "r1"}
        assert captured["payload"] is payload

    @pytest.mark.asyncio
    async def test_missing_reward_raises(self):
        with pytest.raises(tr._PackError, match="total_reward is None"):
            await tr.pack_payload_to_sample(
                _toy_payload(None), ctx=_FakeCtx(), version=0,
            )


# ---------------------------------------------------------------------------
# make_text_rollout_fn end-to-end
# ---------------------------------------------------------------------------


class TestMakeTextRolloutFn:
    @pytest.mark.asyncio
    async def test_builds_rollout_from_service(self):
        async def service(messages, *, n, sample_kwargs, row):
            return [_toy_payload(float(i)) for i in range(n)]

        rollout_fn = tr.make_text_rollout_fn(service)
        ctx = _FakeCtx(completions_per_prompt=3)
        row = {"messages": [{"role": "user", "content": "q"}], "id": "r0"}

        rollout = await rollout_fn(row, ctx)
        assert isinstance(rollout, Rollout)
        assert len(rollout.samples) == 3
        assert [s.reward for s in rollout.samples] == [0.0, 1.0, 2.0]

    @pytest.mark.asyncio
    async def test_service_failure_returns_none(self):
        async def service(messages, *, n, sample_kwargs, row):
            raise RuntimeError("upstream down")

        rollout_fn = tr.make_text_rollout_fn(service)
        out = await rollout_fn(
            {"messages": [{"role": "user", "content": "q"}]}, _FakeCtx(),
        )
        assert out is None

    @pytest.mark.asyncio
    async def test_empty_messages_returns_none(self):
        async def service(messages, *, n, sample_kwargs, row):
            return []  # should not be called

        rollout_fn = tr.make_text_rollout_fn(service)
        out = await rollout_fn({"messages": []}, _FakeCtx())
        assert out is None

    @pytest.mark.asyncio
    async def test_accepts_class_based_service(self):
        class S:
            async def rollout(self, messages, *, n, sample_kwargs, row):
                return [_toy_payload(1.0) for _ in range(n)]

        rollout_fn = tr.make_text_rollout_fn(S())
        rollout = await rollout_fn(
            {"messages": [{"role": "user", "content": "q"}]},
            _FakeCtx(completions_per_prompt=2),
        )
        assert rollout is not None
        assert len(rollout.samples) == 2
