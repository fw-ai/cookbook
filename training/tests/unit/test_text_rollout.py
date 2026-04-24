"""Unit tests for training.utils.rl.text_rollout.

Covers the two supported paths:
  * token-native fast path (payload carries token_ids + logprobs),
  * text-only single-turn fallback (echo re-score mocked).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import pytest

from training.utils.rl.rollout import Rollout
from training.utils.rl.rollout_service import RolloutPayload, TurnRecord
from training.utils.rl import text_rollout as tr


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Deterministic, prefix-preserving chat template.

    Each message contributes ``[role_tok, 200, *[ord(c) for c in content]]``
    so header = 2 tokens, content = one token per character.
    ``add_generation_prompt=True`` appends an assistant header
    ``[ROLE_ASSISTANT, 200]`` with no content.  This is monotonic across
    turns: ``apply_chat_template(m[:i])`` is always a prefix of
    ``apply_chat_template(m[:i+1])``, which is what the multi-turn
    packer relies on.
    """

    _ROLE = {"user": 100, "assistant": 101, "tool": 102, "system": 103}

    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=True):
        assert tokenize
        out: list[int] = []
        for m in messages:
            out.append(self._ROLE[m["role"]])
            out.append(200)
            out.extend(ord(c) for c in m["content"])
        if add_generation_prompt:
            out.append(self._ROLE["assistant"])
            out.append(200)
        return out


class _DriftingTokenizer(_FakeTokenizer):
    """Like ``_FakeTokenizer`` but rewrites the first token on the second
    assistant turn -- simulates a tokenizer that isn't prefix-preserving
    across turns (e.g. the Qwen3/GLM quirks the issue calls out)."""

    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=True):
        out = super().apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt,
        )
        assistant_count = sum(1 for m in messages if m["role"] == "assistant")
        if assistant_count >= 2 and out:
            out[0] = 999  # retroactive drift
        return out


@dataclass
class _FakeCtx:
    tokenizer: _FakeTokenizer = field(default_factory=_FakeTokenizer)
    completions_per_prompt: int = 2
    sample_kwargs: dict = field(default_factory=dict)
    inference_url: str = "https://example.invalid/inference/v1"
    api_key: str = "sk-test"
    model: str = "accounts/test/models/x"
    _version: int = 3

    def current_version(self) -> int:
        return self._version


@pytest.fixture
def patch_echo(monkeypatch):
    """Replace ``_echo_rescore`` with a deterministic stub so tests don't
    hit the network."""
    calls: list[list[int]] = []

    def fake_echo(tokens, *, inference_url, api_key, model):
        calls.append(list(tokens))
        return [-0.5] * len(tokens)

    monkeypatch.setattr(tr, "_echo_rescore", fake_echo)
    return calls


# ---------------------------------------------------------------------------
# Token-native path
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
            payload, prompt_messages=[{"role": "user", "content": "hi"}],
            ctx=_FakeCtx(), version=7,
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
            payload, prompt_messages=[], ctx=_FakeCtx(), version=0,
        )
        assert sample.tokens == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert sample.loss_mask == [0, 0, 1, 1, 0, 0, 0, 1, 1]
        # assistant logprobs flow through; non-assistant positions get 0.0.
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
                payload, prompt_messages=[], ctx=_FakeCtx(), version=0,
            )

    @pytest.mark.asyncio
    async def test_mixed_token_and_text_rejected(self):
        payload = RolloutPayload(
            turns=[
                TurnRecord(role="user", token_ids=[1, 2]),
                TurnRecord(role="assistant", text="only text"),
            ],
            total_reward=0.5,
        )
        with pytest.raises(tr._PackError, match="mixes token-native"):
            await tr.pack_payload_to_sample(
                payload, prompt_messages=[], ctx=_FakeCtx(), version=0,
            )


# ---------------------------------------------------------------------------
# Text-only single-turn path
# ---------------------------------------------------------------------------


class TestTextOnly:
    @pytest.mark.asyncio
    async def test_single_assistant_turn_packs(self, patch_echo):
        payload = RolloutPayload(
            turns=[TurnRecord(role="assistant", text="abc", finish_reason="stop")],
            total_reward=0.25,
        )
        sample = await tr.pack_payload_to_sample(
            payload,
            prompt_messages=[{"role": "user", "content": "hi"}],
            ctx=_FakeCtx(), version=5,
        )
        # prompt [user, 'h', 'i'] + assistant header + 'a','b','c'
        expected = [100, 200, ord("h"), ord("i"), 101, 200, ord("a"), ord("b"), ord("c")]
        assert sample.tokens == expected
        # header (6 tokens) masked out, content (3 tokens) masked in.
        assert sample.loss_mask == [0] * 6 + [1] * 3
        assert sample.logprobs == [-0.5] * len(expected)
        assert sample.reward == 0.25
        assert patch_echo == [expected]

    @pytest.mark.asyncio
    async def test_multi_turn_text_interleaved_mask(self, patch_echo):
        """a1 / tool / a2 / tool / a3 -- mask must hit every assistant
        content span and nothing else."""
        payload = RolloutPayload(
            turns=[
                TurnRecord(role="assistant", text="a1"),
                TurnRecord(role="tool", text="obs1"),
                TurnRecord(role="assistant", text="ok"),
                TurnRecord(role="tool", text="o2"),
                TurnRecord(role="assistant", text="done"),
            ],
            total_reward=1.0,
        )
        sample = await tr.pack_payload_to_sample(
            payload,
            prompt_messages=[{"role": "user", "content": "hi"}],
            ctx=_FakeCtx(), version=0,
        )
        # Masked-in positions must all decode to assistant content chars,
        # not role tokens (100-103) or the 200 boundary marker.
        assistant_chars = set(ord(c) for c in "a1okdone")
        trained = [tok for tok, m in zip(sample.tokens, sample.loss_mask) if m]
        assert all(t in assistant_chars for t in trained)
        assert len(trained) == len("a1") + len("ok") + len("done")
        # Total tokens == header(2) + "hi"(2) + 3*(asst header 2) + contents +
        #                 2*(tool header 2 + content).
        # Sanity-check via re-tokenisation:
        assert len(sample.tokens) == 2 + 2 + (2 + 2) + (2 + 4) + (2 + 2) + (2 + 2) + (2 + 4)
        # Echo re-score called once on the full token list.
        assert patch_echo == [sample.tokens]

    @pytest.mark.asyncio
    async def test_multi_turn_text_tokenizer_drift_rejected(self, patch_echo):
        """When the tokenizer isn't prefix-preserving across turns the
        packer must error loud with the offending turn index, not silently
        produce a wrong mask."""
        ctx = _FakeCtx(tokenizer=_DriftingTokenizer())
        payload = RolloutPayload(
            turns=[
                TurnRecord(role="assistant", text="a1"),
                TurnRecord(role="user", text="followup"),
                TurnRecord(role="assistant", text="a2"),
            ],
            total_reward=1.0,
        )
        with pytest.raises(tr._PackError, match="prefix-preserving"):
            await tr.pack_payload_to_sample(
                payload,
                prompt_messages=[{"role": "user", "content": "hi"}],
                ctx=ctx, version=0,
            )

    @pytest.mark.asyncio
    async def test_missing_final_assistant_rejected(self, patch_echo):
        payload = RolloutPayload(
            turns=[
                TurnRecord(role="assistant", text="a1"),
                TurnRecord(role="user", text="followup"),
            ],
            total_reward=0.0,
        )
        with pytest.raises(tr._PackError, match="end with an assistant"):
            await tr.pack_payload_to_sample(
                payload,
                prompt_messages=[{"role": "user", "content": "q"}],
                ctx=_FakeCtx(), version=0,
            )

    @pytest.mark.asyncio
    async def test_empty_assistant_rejected(self, patch_echo):
        payload = RolloutPayload(
            turns=[TurnRecord(role="assistant", text="")], total_reward=0.0,
        )
        with pytest.raises(tr._PackError, match="empty text"):
            await tr.pack_payload_to_sample(
                payload,
                prompt_messages=[{"role": "user", "content": "q"}],
                ctx=_FakeCtx(), version=0,
            )


# ---------------------------------------------------------------------------
# Reward plumbing
# ---------------------------------------------------------------------------


class TestReward:
    @pytest.mark.asyncio
    async def test_server_reward_wins(self, patch_echo):
        async def reward_fn(row, payload):
            return 99.0  # should not be called

        payload = RolloutPayload(
            turns=[TurnRecord(role="assistant", text="a")],
            total_reward=0.5,
        )
        sample = await tr.pack_payload_to_sample(
            payload, prompt_messages=[{"role": "user", "content": "q"}],
            ctx=_FakeCtx(), version=0, reward_fn=reward_fn,
        )
        assert sample.reward == 0.5

    @pytest.mark.asyncio
    async def test_reward_fn_called_when_unset(self, patch_echo):
        captured: dict = {}

        async def reward_fn(row, payload):
            captured["row"] = row
            captured["payload"] = payload
            return 0.8

        payload = RolloutPayload(
            turns=[TurnRecord(role="assistant", text="a")], total_reward=None,
        )
        sample = await tr.pack_payload_to_sample(
            payload,
            prompt_messages=[{"role": "user", "content": "q"}],
            ctx=_FakeCtx(), version=0, reward_fn=reward_fn,
            row={"id": "r1"},
        )
        assert sample.reward == 0.8
        assert captured["row"] == {"id": "r1"}
        assert captured["payload"] is payload

    @pytest.mark.asyncio
    async def test_missing_reward_raises(self, patch_echo):
        payload = RolloutPayload(
            turns=[TurnRecord(role="assistant", text="a")], total_reward=None,
        )
        with pytest.raises(tr._PackError, match="total_reward is None"):
            await tr.pack_payload_to_sample(
                payload, prompt_messages=[{"role": "user", "content": "q"}],
                ctx=_FakeCtx(), version=0,
            )


# ---------------------------------------------------------------------------
# make_text_rollout_fn end-to-end
# ---------------------------------------------------------------------------


class TestMakeTextRolloutFn:
    @pytest.mark.asyncio
    async def test_builds_rollout_from_service(self, patch_echo):
        async def service(messages, *, n, sample_kwargs, row):
            return [
                RolloutPayload(
                    turns=[TurnRecord(role="assistant", text="x")],
                    total_reward=float(i),
                )
                for i in range(n)
            ]

        rollout_fn = tr.make_text_rollout_fn(service)
        ctx = _FakeCtx(completions_per_prompt=3)
        row = {"messages": [{"role": "user", "content": "q"}], "id": "r0"}

        rollout = await rollout_fn(row, ctx)
        assert isinstance(rollout, Rollout)
        assert len(rollout.samples) == 3
        assert [s.reward for s in rollout.samples] == [0.0, 1.0, 2.0]

    @pytest.mark.asyncio
    async def test_service_failure_returns_none(self, patch_echo):
        async def service(messages, *, n, sample_kwargs, row):
            raise RuntimeError("upstream down")

        rollout_fn = tr.make_text_rollout_fn(service)
        out = await rollout_fn(
            {"messages": [{"role": "user", "content": "q"}]}, _FakeCtx(),
        )
        assert out is None

    @pytest.mark.asyncio
    async def test_empty_messages_returns_none(self, patch_echo):
        async def service(messages, *, n, sample_kwargs, row):
            return []  # should not be called

        rollout_fn = tr.make_text_rollout_fn(service)
        out = await rollout_fn({"messages": []}, _FakeCtx())
        assert out is None

    @pytest.mark.asyncio
    async def test_accepts_class_based_service(self, patch_echo):
        class S:
            async def rollout(self, messages, *, n, sample_kwargs, row):
                return [
                    RolloutPayload(
                        turns=[TurnRecord(role="assistant", text="ok")],
                        total_reward=1.0,
                    )
                    for _ in range(n)
                ]

        rollout_fn = tr.make_text_rollout_fn(S())
        rollout = await rollout_fn(
            {"messages": [{"role": "user", "content": "q"}]},
            _FakeCtx(completions_per_prompt=2),
        )
        assert rollout is not None
        assert len(rollout.samples) == 2
