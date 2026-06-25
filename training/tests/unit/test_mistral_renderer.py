"""Verify ``MistralRenderer`` matches HuggingFace ``apply_chat_template``.

Targets the ``mistralai/Ministral-3-3B-Instruct-2512`` chat template, which is
representative of the Tekken-style template shipped by recent Mistral /
Ministral instruct checkpoints. Every test compares renderer output against the
tokenizer's own ``apply_chat_template`` so we catch any drift the moment Mistral
ships a new template revision.

Run from ``cookbook/training`` with::

    PYTHONPATH=../.. .venv/bin/python -m pytest tests/unit/test_mistral_renderer.py -v -s
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import transformers

from training.renderer.mistral import MistralRenderer
from tinker_cookbook.renderers.base import RenderContext, ToolCall, TrainOnWhat


_LOCAL_PATH = "/shared/Ministral-3-3B-Instruct-2512"
_HF_REPO = "mistralai/Ministral-3-3B-Instruct-2512"
TOKENIZER_MODEL = _LOCAL_PATH if Path(_LOCAL_PATH).exists() else _HF_REPO


@pytest.fixture(scope="module")
def tokenizer() -> transformers.PreTrainedTokenizerBase:
    return transformers.AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL,
        trust_remote_code=True,
    )


@pytest.fixture(scope="module")
def renderer(
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> MistralRenderer:
    return MistralRenderer(tokenizer)


def _hf_tokens(
    tokenizer: transformers.PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] | None = None,
) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        kwargs["tools"] = tools
    result = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(result, "input_ids"):
        return list(result.input_ids)
    return list(result)


def _renderer_tokens(
    renderer: MistralRenderer,
    messages: list[dict[str, Any]],
) -> list[int]:
    model_input = renderer.build_generation_prompt(messages, role="assistant")
    return list(model_input.to_ints())


def _renderer_supervised_tokens(
    renderer: MistralRenderer,
    messages: list[dict[str, Any]],
    *,
    train_on_what: TrainOnWhat,
) -> tuple[list[int], list[float]]:
    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=train_on_what
    )
    return list(model_input.to_ints()), weights.tolist()


def _hf_supervised_tokens(
    tokenizer: transformers.PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
) -> list[int]:
    result = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    if hasattr(result, "input_ids"):
        return list(result.input_ids)
    return list(result)


def _assert_tokens_match(
    tokenizer: transformers.PreTrainedTokenizerBase,
    expected: list[int],
    actual: list[int],
) -> None:
    if expected != actual:
        expected_text = tokenizer.decode(expected)
        actual_text = tokenizer.decode(actual)
        raise AssertionError(
            "Token mismatch:\n"
            f"--- expected ({len(expected)} toks) ---\n{expected_text}\n\n"
            f"--- actual ({len(actual)} toks) ---\n{actual_text}"
        )


def _make_tool_call(name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(
        function=ToolCall.FunctionBody(
            name=name,
            arguments=json.dumps(arguments),
        )
    )


def _message_output_slices(
    renderer: MistralRenderer,
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


# --- Generation prompt parity ----------------------------------------


def test_single_turn_no_system(
    tokenizer: transformers.PreTrainedTokenizerBase, renderer: MistralRenderer
) -> None:
    """Conversations with no system message should auto-inject the default."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    expected = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_single_turn_with_system(
    tokenizer: transformers.PreTrainedTokenizerBase, renderer: MistralRenderer
) -> None:
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    expected = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_explicit_empty_system_renders_empty_block(
    tokenizer: transformers.PreTrainedTokenizerBase, renderer: MistralRenderer
) -> None:
    """Explicit empty system content stays empty (no default injection)."""
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Yo"},
    ]
    expected = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_multi_turn(
    tokenizer: transformers.PreTrainedTokenizerBase, renderer: MistralRenderer
) -> None:
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "Good"},
    ]
    expected = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_generation_prompt_after_user(
    tokenizer: transformers.PreTrainedTokenizerBase, renderer: MistralRenderer
) -> None:
    """Generation prompt has no extra suffix for Mistral templates."""
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Hi"},
    ]
    expected = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_generation_prompt_with_top_level_tools_matches_hf(
    tokenizer: transformers.PreTrainedTokenizerBase, renderer: MistralRenderer
) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "weather in SF?"},
    ]
    prefix = renderer.create_conversation_prefix_with_tools(
        [tools[0]["function"]],
        system_prompt="Be terse.",
    )

    expected = _hf_tokens(
        tokenizer,
        messages,
        add_generation_prompt=True,
        tools=tools,
    )
    actual = _renderer_tokens(renderer, prefix + messages[1:])

    _assert_tokens_match(tokenizer, expected, actual)


# --- Tool calls / tool responses ------------------------------------


def test_tool_call_single(
    tokenizer: transformers.PreTrainedTokenizerBase, renderer: MistralRenderer
) -> None:
    cookbook_messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [_make_tool_call("get_weather", {"city": "SF"})],
        },
    ]
    hf_messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "SF"},
                    },
                }
            ],
        },
    ]
    expected = _hf_tokens(tokenizer, hf_messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, cookbook_messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_tool_call_multi(
    tokenizer: transformers.PreTrainedTokenizerBase, renderer: MistralRenderer
) -> None:
    cookbook_messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "u"},
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [
                _make_tool_call("f1", {"x": 1}),
                _make_tool_call("f2", {"y": "z"}),
            ],
        },
    ]
    hf_messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "u"},
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [
                {"type": "function", "function": {"name": "f1", "arguments": {"x": 1}}},
                {
                    "type": "function",
                    "function": {"name": "f2", "arguments": {"y": "z"}},
                },
            ],
        },
    ]
    expected = _hf_tokens(tokenizer, hf_messages, add_generation_prompt=True)
    actual = _renderer_tokens(renderer, cookbook_messages)
    _assert_tokens_match(tokenizer, expected, actual)


def test_tool_response_round_trip(
    tokenizer: transformers.PreTrainedTokenizerBase, renderer: MistralRenderer
) -> None:
    cookbook_messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [_make_tool_call("get_weather", {"city": "SF"})],
        },
        {"role": "tool", "content": "72F"},
        {"role": "assistant", "content": "It's 72F!"},
    ]
    hf_messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "SF"},
                    },
                }
            ],
        },
        {"role": "tool", "content": "72F"},
        {"role": "assistant", "content": "It's 72F!"},
    ]
    expected = _hf_tokens(tokenizer, hf_messages, add_generation_prompt=False)
    actual = _renderer_tokens(renderer, cookbook_messages)
    _assert_tokens_match(tokenizer, expected, actual)


# --- Supervised tokenization parity --------------------------------


def test_supervised_basic(
    tokenizer: transformers.PreTrainedTokenizerBase, renderer: MistralRenderer
) -> None:
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    expected = _hf_supervised_tokens(tokenizer, messages)
    actual, _weights = _renderer_supervised_tokens(
        renderer, messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    _assert_tokens_match(tokenizer, expected, actual)


def test_supervised_no_system_injects_default(
    tokenizer: transformers.PreTrainedTokenizerBase, renderer: MistralRenderer
) -> None:
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    expected = _hf_supervised_tokens(tokenizer, messages)
    actual, _weights = _renderer_supervised_tokens(
        renderer, messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    _assert_tokens_match(tokenizer, expected, actual)


def test_supervised_weights_last_assistant_message(
    renderer: MistralRenderer,
) -> None:
    """LAST_ASSISTANT_MESSAGE only weights the final assistant turn."""
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    _tokens, weights = _renderer_supervised_tokens(
        renderer, messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
    )
    output_slices = _message_output_slices(renderer, messages)
    assistant_slices = [s for role, s in output_slices if role == "assistant"]
    assert len(assistant_slices) == 2
    assert all(w == 0 for w in weights[assistant_slices[0]])
    assert all(w == 1 for w in weights[assistant_slices[1]])


def test_supervised_weights_mask_non_assistant(
    renderer: MistralRenderer,
) -> None:
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    _tokens, weights = _renderer_supervised_tokens(
        renderer, messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    output_slices = _message_output_slices(renderer, messages)
    for role, sl in output_slices:
        if role == "assistant":
            assert all(w == 1 for w in weights[sl])
        else:
            assert all(w == 0 for w in weights[sl])


# --- Response parsing -----------------------------------------------


def test_parse_response_plain_text(
    tokenizer: transformers.PreTrainedTokenizerBase, renderer: MistralRenderer
) -> None:
    response = tokenizer.encode("Hello world!", add_special_tokens=False) + [
        tokenizer.eos_token_id
    ]
    message, termination = renderer.parse_response(response)
    assert message["role"] == "assistant"
    assert message["content"] == "Hello world!"
    assert termination.is_clean


def test_parse_response_tool_call(
    tokenizer: transformers.PreTrainedTokenizerBase, renderer: MistralRenderer
) -> None:
    raw = 'Sure thing.[TOOL_CALLS]get_weather[ARGS]{"city": "SF"}'
    response = tokenizer.encode(raw, add_special_tokens=False) + [
        tokenizer.eos_token_id
    ]
    message, termination = renderer.parse_response(response)
    assert termination.is_clean
    assert message["content"].rstrip() == "Sure thing."
    tool_calls = message.get("tool_calls") or []
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_calls[0].function.arguments) == {"city": "SF"}
