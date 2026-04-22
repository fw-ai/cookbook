"""Unit tests for training.utils.rl.env_adapters."""

from __future__ import annotations

import asyncio

import pytest

from training.utils.rl.env import MessageEnv, MessageStepResult
from training.utils.rl.env_adapters import SingleTurnEnv, _extract_text, wrap_reward_fn


def _run(coro):
    return asyncio.run(coro)


class TestExtractText:
    def test_none_returns_empty_string(self):
        assert _extract_text(None) == ""

    def test_str_passthrough(self):
        assert _extract_text("hello") == "hello"

    def test_list_of_parts_concatenated(self):
        content = [{"type": "text", "text": "foo "}, {"type": "text", "text": "bar"}]
        assert _extract_text(content) == "foo bar"

    def test_list_with_non_dict_parts(self):
        assert _extract_text(["a", "b", "c"]) == "abc"

    def test_other_types_stringified(self):
        assert _extract_text(42) == "42"


class TestSingleTurnEnv:
    def test_is_a_message_env(self):
        env = SingleTurnEnv(row={}, reward_fn=lambda c, r: 0.0)
        assert isinstance(env, MessageEnv)

    def test_initial_messages_returns_row_messages(self):
        msgs = [{"role": "user", "content": "hi"}]
        env = SingleTurnEnv(row={"messages": msgs}, reward_fn=lambda c, r: 0.0)
        result = _run(env.initial_messages())
        assert result == msgs
        # Defensive copy so mutating row doesn't mutate env output.
        assert result is not msgs

    def test_initial_messages_defaults_to_empty(self):
        env = SingleTurnEnv(row={}, reward_fn=lambda c, r: 0.0)
        assert _run(env.initial_messages()) == []

    def test_custom_messages_field(self):
        msgs = [{"role": "user", "content": "x"}]
        env = SingleTurnEnv(row={"convo": msgs}, reward_fn=lambda c, r: 0.0, messages_field="convo")
        assert _run(env.initial_messages()) == msgs

    def test_step_calls_sync_reward_fn_with_text_and_row(self):
        captured: dict = {}

        def reward(completion, row):
            captured["completion"] = completion
            captured["row"] = row
            return 0.75

        env = SingleTurnEnv(row={"answer": 42, "messages": []}, reward_fn=reward)
        result = _run(env.step({"role": "assistant", "content": "the answer"}))

        assert isinstance(result, MessageStepResult)
        assert result.reward == pytest.approx(0.75)
        assert result.episode_done is True
        assert result.next_messages == []
        assert captured["completion"] == "the answer"
        assert captured["row"] == {"answer": 42, "messages": []}

    def test_step_awaits_async_reward_fn(self):
        async def reward(completion, row):
            return 1.0 if "yes" in completion else 0.0

        env = SingleTurnEnv(row={}, reward_fn=reward)

        assert _run(env.step({"role": "assistant", "content": "yes sir"})).reward == 1.0
        assert _run(env.step({"role": "assistant", "content": "nope"})).reward == 0.0

    def test_step_extracts_text_from_list_content(self):
        seen: dict = {}

        def reward(completion, row):
            seen["text"] = completion
            return 0.0

        env = SingleTurnEnv(row={}, reward_fn=reward)
        _run(
            env.step(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
                }
            )
        )
        assert seen["text"] == "AB"

    def test_step_coerces_reward_to_float(self):
        env = SingleTurnEnv(row={}, reward_fn=lambda c, r: True)
        assert _run(env.step({"role": "assistant", "content": ""})).reward == 1.0


class TestWrapRewardFn:
    def test_returns_env_builder_producing_single_turn_envs(self):
        builder = wrap_reward_fn(lambda c, r: 0.5)
        env = builder({"messages": [{"role": "user", "content": "q"}]})
        assert isinstance(env, SingleTurnEnv)
        result = _run(env.step({"role": "assistant", "content": "a"}))
        assert result.reward == pytest.approx(0.5)

    def test_builder_produces_fresh_env_each_call(self):
        builder = wrap_reward_fn(lambda c, r: 0.0)
        e1 = builder({})
        e2 = builder({})
        assert e1 is not e2

    def test_rejects_non_callable(self):
        with pytest.raises(TypeError, match="callable"):
            wrap_reward_fn("not a function")  # type: ignore[arg-type]

    def test_forwards_messages_field_kwarg(self):
        builder = wrap_reward_fn(lambda c, r: 0.0, messages_field="convo")
        msgs = [{"role": "user", "content": "x"}]
        env = builder({"convo": msgs})
        assert _run(env.initial_messages()) == msgs
