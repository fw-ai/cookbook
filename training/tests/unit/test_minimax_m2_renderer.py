"""Verify ``MiniMaxM2Renderer`` matches HuggingFace ``apply_chat_template``.

Run from ``cookbook/training`` with:
    PYTHONPATH=../.. .venv/bin/python -m pytest tests/unit/test_minimax_m2_renderer.py -v -s
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import transformers

from training.renderer.minimax_m2 import MiniMaxM2Renderer
from tinker_cookbook.renderers.base import RenderContext, ToolCall, TrainOnWhat

_LOCAL_PATH = "/shared/MiniMax-M2"
_HF_REPO = "MiniMaxAI/MiniMax-M2"
TOKENIZER_MODEL = _LOCAL_PATH if Path(_LOCAL_PATH).exists() else _HF_REPO


@pytest.fixture(scope="module")
def tokenizer() -> transformers.PreTrainedTokenizerBase:
    """Load the MiniMax tokenizer from a local snapshot when available."""
    return transformers.AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL,
        trust_remote_code=True,
    )


@pytest.fixture(scope="module")
def renderer(
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> MiniMaxM2Renderer:
    """Create the renderer with HF-matching history truncation enabled."""
    return MiniMaxM2Renderer(tokenizer, strip_thinking_from_history=True)


def _hf_tokens(
    tokenizer: transformers.PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(result, "input_ids"):
        return list(result.input_ids)
    return list(result)


def _renderer_tokens(
    renderer: MiniMaxM2Renderer,
    messages: list[dict[str, Any]],
) -> list[int]:
    model_input = renderer.build_generation_prompt(messages, role="assistant")
    return list(model_input.to_ints())


def _renderer_supervised_tokens(
    renderer: MiniMaxM2Renderer,
    messages: list[dict[str, Any]],
    *,
    train_on_what: TrainOnWhat,
) -> tuple[list[int], list[float]]:
    model_input, weights = renderer.build_supervised_example(
        messages,
        train_on_what=train_on_what,
    )
    return list(model_input.to_ints()), weights.tolist()


def _hf_supervised_tokens(
    tokenizer: transformers.PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
) -> list[int]:
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    if hasattr(result, "input_ids"):
        return list(result.input_ids)
    return list(result)


def _assert_tokens_match(
    tokenizer: transformers.PreTrainedTokenizerBase,
    expected: list[int],
    actual: list[int],
) -> None:
    expected_text = tokenizer.decode(expected)
    actual_text = tokenizer.decode(actual)
    assert expected == actual, (
        f"Token mismatch:\n--- expected ---\n{expected_text}\n\n"
        f"--- actual ---\n{actual_text}"
    )


def _make_tool_call(name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(
        function=ToolCall.FunctionBody(
            name=name,
            arguments=json.dumps(arguments),
        )
    )


def _message_output_slices(
    renderer: MiniMaxM2Renderer,
    messages: list[dict[str, Any]],
) -> list[tuple[str, slice]]:
    processed = renderer._preprocess_messages(messages)
    last_user_index = max(
        (idx for idx, message in enumerate(processed) if message["role"] == "user"),
        default=-1,
    )

    offset = len(renderer._bos_tokens)
    slices: list[tuple[str, slice]] = []
    for idx, message in enumerate(processed):
        ctx = RenderContext(
            idx=idx,
            is_last=(idx == len(processed) - 1),
            prev_message=processed[idx - 1] if idx > 0 else None,
            last_user_index=last_user_index,
        )
        rendered = renderer.render_message(message, ctx)
        header_len = 0 if rendered.header is None else rendered.header.length
        output_len = sum(chunk.length for chunk in rendered.output)
        offset += header_len
        slices.append((message["role"], slice(offset, offset + output_len)))
        offset += output_len
    return slices


def test_single_turn_no_thinking(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """Single-turn user/assistant rendering should match HF exactly."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    expected = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_single_turn_with_thinking(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """Assistant think blocks should be normalized to the MiniMax multiline form."""
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>Let me calculate</think>4"},
    ]
    expected = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_with_system_message(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """Explicit system prompts should override the default injected prompt."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    expected = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_default_system_message_is_injected(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """Missing system prompts should render the tokenizer's default system block."""
    messages = [{"role": "user", "content": "Hello"}]
    expected = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_multi_turn_history_truncation(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """Historical assistant reasoning should be stripped before the last user turn."""
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>2+2=4</think>The answer is 4."},
        {"role": "user", "content": "What about 3+3?"},
        {"role": "assistant", "content": "<think>3+3=6</think>The answer is 6."},
    ]
    expected = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_assistant_after_last_user_keeps_thinking(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """Assistant turns after the last user message should keep thinking intact."""
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "<think>thought 1</think>Reply 1"},
        {"role": "assistant", "content": "<think>thought 2</think>Reply 2"},
    ]
    expected = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_generation_prompt_suffix(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """The generation suffix should end with ``]~b]ai\\n<think>\\n``."""
    messages = [{"role": "user", "content": "Hello"}]
    expected = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_tool_call_multiple_params(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """Tool calls with multiple scalar parameters should match HF exactly."""
    messages = [
        {"role": "user", "content": "Book a flight"},
        {
            "role": "assistant",
            "content": "<think>booking</think>Booking now.",
            "tool_calls": [
                _make_tool_call(
                    "book_flight",
                    {
                        "origin": "SFO",
                        "destination": "LAX",
                        "date": "2026-04-01",
                    },
                )
            ],
        },
    ]
    hf_messages = [
        {"role": "user", "content": "Book a flight"},
        {
            "role": "assistant",
            "content": "<think>booking</think>Booking now.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "book_flight",
                        "arguments": {
                            "origin": "SFO",
                            "destination": "LAX",
                            "date": "2026-04-01",
                        },
                    },
                }
            ],
        },
    ]
    expected = _hf_tokens(tokenizer, hf_messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_tool_call_multiple_calls(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """Multiple tool calls should render as consecutive ``<invoke>`` blocks."""
    messages = [
        {"role": "user", "content": "Compare weather in SF and NYC"},
        {
            "role": "assistant",
            "content": "<think>check both cities</think>Let me check both.",
            "tool_calls": [
                _make_tool_call("get_weather", {"city": "San Francisco"}),
                _make_tool_call("get_weather", {"city": "New York"}),
            ],
        },
    ]
    hf_messages = [
        {"role": "user", "content": "Compare weather in SF and NYC"},
        {
            "role": "assistant",
            "content": "<think>check both cities</think>Let me check both.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "San Francisco"},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "New York"},
                    },
                },
            ],
        },
    ]
    expected = _hf_tokens(tokenizer, hf_messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_tool_response_multiple_grouped(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """Consecutive tool responses should share one ``tool`` header."""
    messages = [
        {"role": "user", "content": "Compare weather"},
        {
            "role": "assistant",
            "content": "<think>checking</think>Checking.",
            "tool_calls": [
                _make_tool_call("get_weather", {"city": "SF"}),
                _make_tool_call("get_weather", {"city": "NYC"}),
            ],
        },
        {"role": "tool", "content": "72F"},
        {"role": "tool", "content": "55F"},
        {"role": "assistant", "content": "<think>done</think>Compared."},
    ]
    hf_messages = [
        {"role": "user", "content": "Compare weather"},
        {
            "role": "assistant",
            "content": "<think>checking</think>Checking.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "SF"}},
                },
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "NYC"}},
                },
            ],
        },
        {"role": "tool", "content": "72F"},
        {"role": "tool", "content": "55F"},
        {"role": "assistant", "content": "<think>done</think>Compared."},
    ]
    expected = _hf_tokens(tokenizer, hf_messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_tool_call_history_truncation(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """Historical tool-calling assistants should lose thinking but keep the tool call."""
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "<think>I should check</think>Let me check.",
            "tool_calls": [_make_tool_call("get_weather", {"city": "SF"})],
        },
        {"role": "tool", "content": "72F"},
        {"role": "assistant", "content": "<think>got result</think>It's 72F!"},
        {"role": "user", "content": "Thanks! And in NYC?"},
    ]
    hf_messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "<think>I should check</think>Let me check.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "SF"}},
                }
            ],
        },
        {"role": "tool", "content": "72F"},
        {"role": "assistant", "content": "<think>got result</think>It's 72F!"},
        {"role": "user", "content": "Thanks! And in NYC?"},
    ]
    expected = _hf_tokens(tokenizer, hf_messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_supervised_example_basic(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """Supervised tokenization should match HF without a generation suffix."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>calculate</think>4"},
    ]
    expected = _hf_supervised_tokens(tokenizer, messages)
    actual, _weights = _renderer_supervised_tokens(
        renderer,
        messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    _assert_tokens_match(tokenizer, expected, actual)


def test_supervised_weights_last_assistant_message(
    renderer: MiniMaxM2Renderer,
) -> None:
    """Only the final assistant output should be trainable in LAST_ASSISTANT_MESSAGE mode."""
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    _tokens, weights = _renderer_supervised_tokens(
        renderer,
        messages,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    output_slices = _message_output_slices(renderer, messages)
    assistant_slices = [slice_ for role, slice_ in output_slices if role == "assistant"]

    assert len(assistant_slices) == 2
    assert all(weight == 0 for weight in weights[assistant_slices[0]])
    assert all(weight == 1 for weight in weights[assistant_slices[1]])


def test_supervised_weights_mask_non_assistant(
    renderer: MiniMaxM2Renderer,
) -> None:
    """Only assistant outputs should be trainable in ALL_ASSISTANT_MESSAGES mode."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    _tokens, weights = _renderer_supervised_tokens(
        renderer,
        messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    output_slices = _message_output_slices(renderer, messages)
    assistant_slices = [slice_ for role, slice_ in output_slices if role == "assistant"]
    non_assistant_slices = [
        slice_ for role, slice_ in output_slices if role != "assistant"
    ]

    assert len(assistant_slices) == 1
    for assistant_slice in assistant_slices:
        assert all(weight == 1 for weight in weights[assistant_slice])
    for non_assistant_slice in non_assistant_slices:
        assert all(weight == 0 for weight in weights[non_assistant_slice])


def test_empty_system_message(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """An explicit empty system message should still render the default system block."""
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Hello"},
    ]
    expected = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_multi_turn_no_thinking_history(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """Historical assistant messages without think tags should still match HF."""
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
    ]
    expected = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_tool_call_single(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """A single tool call should match the HF XML rendering."""
    messages = [
        {"role": "user", "content": "What's the weather in SF?"},
        {
            "role": "assistant",
            "content": "<think>I should check the weather</think>Let me check.",
            "tool_calls": [_make_tool_call("get_weather", {"city": "San Francisco"})],
        },
    ]
    hf_messages = [
        {"role": "user", "content": "What's the weather in SF?"},
        {
            "role": "assistant",
            "content": "<think>I should check the weather</think>Let me check.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "San Francisco"},
                    },
                }
            ],
        },
    ]
    expected = _hf_tokens(tokenizer, hf_messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_tool_call_no_thinking(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """Tool calls without explicit reasoning should still match HF."""
    messages = [
        {"role": "user", "content": "Search for python docs"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_make_tool_call("search", {"query": "python"})],
        },
    ]
    hf_messages = [
        {"role": "user", "content": "Search for python docs"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "search", "arguments": {"query": "python"}},
                }
            ],
        },
    ]
    expected = _hf_tokens(tokenizer, hf_messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_tool_response_single(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """A single tool response round-trip should match HF exactly."""
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "<think>check weather</think>Checking.",
            "tool_calls": [_make_tool_call("get_weather", {"city": "SF"})],
        },
        {"role": "tool", "content": "72F and sunny"},
        {
            "role": "assistant",
            "content": "<think>got it</think>It is 72F and sunny in SF!",
        },
    ]
    hf_messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "<think>check weather</think>Checking.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "SF"}},
                }
            ],
        },
        {"role": "tool", "content": "72F and sunny"},
        {
            "role": "assistant",
            "content": "<think>got it</think>It is 72F and sunny in SF!",
        },
    ]
    expected = _hf_tokens(tokenizer, hf_messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_tool_response_three_consecutive(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """Three grouped tool responses should share one tool block like HF."""
    messages = [
        {"role": "user", "content": "Get weather for three cities"},
        {
            "role": "assistant",
            "content": "<think>checking all three</think>Let me check.",
            "tool_calls": [
                _make_tool_call("get_weather", {"city": "San Francisco"}),
                _make_tool_call("get_weather", {"city": "New York"}),
                _make_tool_call("get_weather", {"city": "Chicago"}),
            ],
        },
        {"role": "tool", "content": "72F and sunny"},
        {"role": "tool", "content": "55F and cloudy"},
        {"role": "tool", "content": "45F and windy"},
        {"role": "assistant", "content": "<think>got all three</think>Done."},
    ]
    hf_messages = [
        {"role": "user", "content": "Get weather for three cities"},
        {
            "role": "assistant",
            "content": "<think>checking all three</think>Let me check.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "San Francisco"},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "New York"},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Chicago"},
                    },
                },
            ],
        },
        {"role": "tool", "content": "72F and sunny"},
        {"role": "tool", "content": "55F and cloudy"},
        {"role": "tool", "content": "45F and windy"},
        {"role": "assistant", "content": "<think>got all three</think>Done."},
    ]
    expected = _hf_tokens(tokenizer, hf_messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_tool_call_nested_json_args(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """Nested list/dict tool args should serialize exactly like HF."""
    messages = [
        {"role": "user", "content": "Create an event"},
        {
            "role": "assistant",
            "content": "<think>creating</think>Creating the event.",
            "tool_calls": [
                _make_tool_call(
                    "create_event",
                    {
                        "title": "Team Meeting",
                        "attendees": ["alice", "bob"],
                        "location": {"building": "HQ", "room": "101"},
                    },
                )
            ],
        },
    ]
    hf_messages = [
        {"role": "user", "content": "Create an event"},
        {
            "role": "assistant",
            "content": "<think>creating</think>Creating the event.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "create_event",
                        "arguments": {
                            "title": "Team Meeting",
                            "attendees": ["alice", "bob"],
                            "location": {"building": "HQ", "room": "101"},
                        },
                    },
                }
            ],
        },
    ]
    expected = _hf_tokens(tokenizer, hf_messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_generation_prompt_after_tool_response(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """Generation after a tool response should match HF's assistant suffix."""
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "<think>checking</think>Let me check.",
            "tool_calls": [_make_tool_call("get_weather", {"city": "SF"})],
        },
        {"role": "tool", "content": "72F and sunny"},
    ]
    hf_messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "<think>checking</think>Let me check.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "SF"}},
                }
            ],
        },
        {"role": "tool", "content": "72F and sunny"},
    ]
    expected = _hf_tokens(tokenizer, hf_messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_multi_turn_multiple_tool_roundtrips(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """Two tool-use exchanges should preserve truncation behavior across turns."""
    messages = [
        {"role": "user", "content": "What's the weather in SF?"},
        {
            "role": "assistant",
            "content": "<think>check SF weather</think>Checking SF.",
            "tool_calls": [_make_tool_call("get_weather", {"city": "SF"})],
        },
        {"role": "tool", "content": "72F and sunny"},
        {
            "role": "assistant",
            "content": "<think>got SF</think>It is 72F and sunny in SF.",
        },
        {"role": "user", "content": "Now check NYC"},
        {
            "role": "assistant",
            "content": "<think>check NYC</think>Checking NYC.",
            "tool_calls": [_make_tool_call("get_weather", {"city": "NYC"})],
        },
        {"role": "tool", "content": "55F and rainy"},
        {"role": "assistant", "content": "<think>got NYC</think>NYC is 55F and rainy."},
    ]
    hf_messages = [
        {"role": "user", "content": "What's the weather in SF?"},
        {
            "role": "assistant",
            "content": "<think>check SF weather</think>Checking SF.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "SF"}},
                }
            ],
        },
        {"role": "tool", "content": "72F and sunny"},
        {
            "role": "assistant",
            "content": "<think>got SF</think>It is 72F and sunny in SF.",
        },
        {"role": "user", "content": "Now check NYC"},
        {
            "role": "assistant",
            "content": "<think>check NYC</think>Checking NYC.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "NYC"}},
                }
            ],
        },
        {"role": "tool", "content": "55F and rainy"},
        {"role": "assistant", "content": "<think>got NYC</think>NYC is 55F and rainy."},
    ]
    expected = _hf_tokens(tokenizer, hf_messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_empty_assistant_content(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """An empty assistant message should still round-trip to HF tokens."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": ""},
    ]
    expected = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_supervised_with_tool_calls(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """Supervised tokenization with a tool round-trip should match HF."""
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "<think>check weather</think>Checking.",
            "tool_calls": [_make_tool_call("get_weather", {"city": "SF"})],
        },
        {"role": "tool", "content": "72F and sunny"},
        {
            "role": "assistant",
            "content": "<think>got it</think>It is 72F and sunny in SF!",
        },
    ]
    hf_messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "<think>check weather</think>Checking.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "SF"}},
                }
            ],
        },
        {"role": "tool", "content": "72F and sunny"},
        {
            "role": "assistant",
            "content": "<think>got it</think>It is 72F and sunny in SF!",
        },
    ]
    expected = _hf_supervised_tokens(tokenizer, hf_messages)
    actual, _weights = _renderer_supervised_tokens(
        renderer,
        messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    _assert_tokens_match(tokenizer, expected, actual)


def test_supervised_multi_turn_thinking_weights(
    renderer: MiniMaxM2Renderer,
) -> None:
    """LAST_ASSISTANT_MESSAGE should only weight the final assistant turn."""
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>calculating</think>4"},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "<think>adding</think>6"},
    ]
    _tokens, weights = _renderer_supervised_tokens(
        renderer,
        messages,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    output_slices = _message_output_slices(renderer, messages)
    assistant_slices = [slice_ for role, slice_ in output_slices if role == "assistant"]

    assert len(assistant_slices) == 2
    assert all(weight == 0 for weight in weights[assistant_slices[0]])
    assert all(weight == 1 for weight in weights[assistant_slices[1]])


def test_supervised_no_system_message(
    tokenizer: transformers.PreTrainedTokenizerBase,
    renderer: MiniMaxM2Renderer,
) -> None:
    """Supervised rendering should auto-inject the default system prompt like HF."""
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    expected = _hf_supervised_tokens(tokenizer, messages)
    actual, _weights = _renderer_supervised_tokens(
        renderer,
        messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    _assert_tokens_match(tokenizer, expected, actual)
