"""Unit tests for the NeMo Gym example's recording proxy (no network / deployment)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from training.examples.rl.nemo_gym.proxy import (
    RecordingProxy,
    RolloutSession,
    _finish_reason,
    _fix_tool_calls_for_openai,
    _is_user_simulator_call,
)
from training.utils.rl.agent.trajectory import TurnRecord


class _StubRenderer:
    def prompt_tokens(self, *, messages, tools, system_prompt):
        return [1, 2, 3] + [len(messages)] * len(messages)

    def stop_sequences(self):
        return []

    def parse_completion(self, tokens):
        return {"role": "assistant", "content": "ok"}


class _StubSampler:
    def __init__(self, finish_reason=None):
        self.calls = []
        self._finish_reason = finish_reason

    async def sample_with_prompt_tokens(self, prompt_ids, **kwargs):
        self.calls.append(kwargs)
        out = [7, 8]
        return [
            SimpleNamespace(
                prompt_len=len(prompt_ids),
                full_tokens=[*prompt_ids, *out],
                sampling_logprobs=[-0.1, -0.2],
                logprobs_echoed=False,
                finish_reason=self._finish_reason,
            )
        ]


def _proxy(sampler, **kwargs) -> RecordingProxy:
    proxy = RecordingProxy.__new__(RecordingProxy)
    # Bypass __init__ so no tokenizer/renderer build is needed.
    from aiohttp import web

    proxy._sampler = sampler
    proxy._renderer = _StubRenderer()
    proxy._max_sample_tokens = 64
    proxy._temperature = 0.7
    proxy._exclude_user_simulator = kwargs.get("exclude_user_simulator", False)
    proxy._sessions = {}
    proxy.app = web.Application()
    proxy.app.router.add_post("/v1/chat/completions", proxy._chat_completions)
    return proxy


async def _post(proxy, body):
    async with TestClient(TestServer(proxy.app)) as client:
        return await client.post("/v1/chat/completions", json=body)


def _body(**extra):
    return {"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}], **extra}


def test_missing_user_field_is_rejected_not_shared():
    proxy = _proxy(_StubSampler())
    resp = asyncio.run(_post(proxy, _body()))
    assert resp.status == 400
    assert proxy.active_rollout_ids() == []


def test_request_cannot_override_temperature_or_raise_max_tokens():
    sampler = _StubSampler()
    proxy = _proxy(sampler)
    resp = asyncio.run(_post(proxy, _body(user="0-0", temperature=0.0, max_tokens=10_000)))
    assert resp.status == 200
    assert sampler.calls[0]["temperature"] == 0.7
    assert sampler.calls[0]["max_tokens"] == 64
    assert sampler.calls[0]["user"]  # session affinity key forwarded


def test_finish_reason_length_is_reported():
    assert _finish_reason(SimpleNamespace(finish_reason="length"), {}) == "length"
    assert _finish_reason(SimpleNamespace(finish_reason="stop"), {"tool_calls": [1]}) == "tool_calls"
    assert _finish_reason(SimpleNamespace(), {}) == "stop"


def test_turns_chain_append_only_and_session_pops():
    sampler = _StubSampler()
    proxy = _proxy(sampler)

    async def run():
        async with TestClient(TestServer(proxy.app)) as client:
            for _ in range(2):
                assert (await client.post("/v1/chat/completions", json=_body(user="1-0"))).status == 200

    asyncio.run(run())
    session = proxy.pop_session("1-0")
    assert session is not None and session.turn_count == 2
    assert proxy.pop_session("1-0") is None


def test_to_samples_keeps_every_segment():
    session = RolloutSession(rollout_id="x")
    tree = session.training.tree
    # Second prompt is NOT a token-prefix of turn 1's prompt+output -> two segments.
    n0 = tree.add_turn(TurnRecord(prompt_ids=[1, 2], output_ids=[3], finish_reason="stop", output_log_probs=[-0.5]))
    n1 = tree.add_turn(
        TurnRecord(prompt_ids=[9, 9, 9], output_ids=[4], finish_reason="stop", output_log_probs=[-0.5]),
        parent_id=n0.node_id,
    )
    session.leaf_id = n1.node_id
    samples = session.to_samples(reward=1.0)
    assert len(samples) >= 2
    assert all(any(s.loss_mask) and s.reward == 1.0 for s in samples)
    assert RolloutSession(rollout_id="empty").to_samples(reward=1.0) == []


def test_user_simulator_filter_is_opt_in():
    no_tools = {"messages": [{"role": "user", "content": "hi"}], "user": "2-0"}
    off = _proxy(_StubSampler())
    asyncio.run(_post(off, no_tools))
    assert off.pop_session("2-0").turn_count == 1  # tool-less agent turn is trained on

    on = _proxy(_StubSampler(), exclude_user_simulator=True)
    asyncio.run(_post(on, no_tools))
    assert on.pop_session("2-0").turn_count == 0
    assert _is_user_simulator_call([])
    assert _is_user_simulator_call([{"function": {"name": "end_conversation"}}])
    assert not _is_user_simulator_call([{"function": {"name": "a"}}, {"function": {"name": "b"}}])


def test_tool_calls_coerced_to_openai_shape():
    msg = {"tool_calls": [{"id": None, "function": {"name": "f", "arguments": {"a": 1}}}]}
    _fix_tool_calls_for_openai(msg)
    call = msg["tool_calls"][0]
    assert call["id"].startswith("call_") and call["function"]["arguments"] == '{"a": 1}'
