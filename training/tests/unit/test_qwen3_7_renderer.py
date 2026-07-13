"""Focused Qwen3.7 thinking, multiturn, and tool-call renderer coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

import training.renderer  # noqa: F401 - registers cookbook-local renderers
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import TrainOnWhat
from tinker_cookbook.tokenizer_utils import get_tokenizer
from training.utils.supervised import (
    normalize_messages,
    prepare_messages_with_tools,
)


_TOKENIZER_MODEL = (
    "/shared/qwen3p7-plus-llm-think/hf"
    if Path("/shared/qwen3p7-plus-llm-think/hf/tokenizer_config.json").exists()
    else "Qwen/Qwen3.6-27B"
)
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def _tokenizer():
    try:
        tokenizer = get_tokenizer(_TOKENIZER_MODEL)
    except (OSError, ValueError, RuntimeError) as exc:
        pytest.skip(f"tokenizer unavailable for {_TOKENIZER_MODEL!r}: {exc}")
    if not getattr(tokenizer, "chat_template", None):
        pytest.skip(f"{_TOKENIZER_MODEL!r} has no chat_template")
    return tokenizer


def _assert_hf_parity(
    *,
    renderer_name: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    **template_kwargs,
) -> None:
    tokenizer = _tokenizer()
    renderer = get_renderer(renderer_name, tokenizer)
    normalized = prepare_messages_with_tools(
        messages,
        renderer=renderer,
        tools=tools,
    )
    renderer_tokens = list(
        renderer.build_generation_prompt(normalized, role="assistant").to_ints()
    )
    hf_result = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        **template_kwargs,
    )
    hf_tokens = list(
        hf_result.input_ids if hasattr(hf_result, "input_ids") else hf_result
    )
    assert renderer_tokens == hf_tokens

    renderer_text = tokenizer.decode(
        renderer_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    hf_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    assert renderer_text.encode("utf-8") == hf_text.encode("utf-8")


def test_qwen3_7_variants_are_registered_with_expected_extension_property():
    tokenizer = _tokenizer()

    default = get_renderer("qwen3_7", tokenizer)
    disabled = get_renderer("qwen3_7_disable_thinking", tokenizer)
    preserved = get_renderer("qwen3_7_preserve_thinking", tokenizer)

    assert default.has_extension_property is False
    assert disabled.has_extension_property is False
    assert preserved.has_extension_property is True


@pytest.mark.parametrize(
    ("renderer_name", "messages", "template_kwargs"),
    [
        (
            "qwen3_7",
            [{"role": "user", "content": "What is 2+2?"}],
            {},
        ),
        (
            "qwen3_7_disable_thinking",
            [{"role": "user", "content": "What is 2+2?"}],
            {"enable_thinking": False},
        ),
        (
            "qwen3_7_preserve_thinking",
            [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "reasoning_content": "Two plus two is four.",
                    "content": "4",
                },
                {"role": "user", "content": "And 3+3?"},
            ],
            {"preserve_thinking": True},
        ),
    ],
)
@pytest.mark.timeout(180)
def test_thinking_modes_are_byte_identical_to_hf_template(
    renderer_name: str,
    messages: list[dict],
    template_kwargs: dict,
):
    _assert_hf_parity(
        renderer_name=renderer_name,
        messages=messages,
        **template_kwargs,
    )


@pytest.mark.timeout(180)
def test_empty_content_tool_call_keeps_exact_thinking_separator():
    messages = [
        {"role": "user", "content": "Check the weather in SF."},
        {
            "role": "assistant",
            "reasoning_content": "I need current data.",
            "content": "",
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
        {"role": "tool", "content": "72F"},
    ]

    _assert_hf_parity(
        renderer_name="qwen3_7",
        messages=messages,
        tools=_TOOLS,
    )


@pytest.mark.timeout(180)
def test_parallel_tool_group_closes_before_following_user():
    messages = [
        {"role": "user", "content": "Compare SF and NYC weather."},
        {
            "role": "assistant",
            "reasoning_content": "I need two calls.",
            "content": "Checking.",
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
        {"role": "tool", "content": "SF: 72F"},
        {"role": "tool", "content": "NYC: 68F"},
        {"role": "user", "content": "Which is warmer?"},
    ]

    _assert_hf_parity(
        renderer_name="qwen3_7_preserve_thinking",
        messages=messages,
        tools=_TOOLS,
        preserve_thinking=True,
    )


@pytest.mark.timeout(180)
def test_parser_round_trip_keeps_thinking_and_structured_tool_call():
    tokenizer = _tokenizer()
    renderer = get_renderer("qwen3_7", tokenizer)
    response_text = (
        "Need current data.\n</think>\n\n"
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=city>\nSan Francisco\n</parameter>\n"
        "</function>\n"
        "</tool_call><|im_end|>"
    )
    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)

    parsed, termination = renderer.parse_response(response_tokens)

    assert termination.is_clean
    assert parsed["content"][0] == {
        "type": "thinking",
        "thinking": "Need current data.",
    }
    assert parsed["tool_calls"][0].function.name == "get_weather"
    assert parsed["tool_calls"][0].function.arguments == '{"city": "San Francisco"}'

    # This is the verifier's parser -> normalizer hand-off that previously
    # raised TypeError for tinker_cookbook.renderers.base.ToolCall.
    normalized = normalize_messages([parsed])
    assert normalized[0]["tool_calls"] == parsed["tool_calls"]


@pytest.mark.timeout(180)
def test_tool_trajectory_supervised_tokens_and_extension_property():
    tokenizer = _tokenizer()
    renderer = get_renderer("qwen3_7_preserve_thinking", tokenizer)
    messages = [
        {"role": "user", "content": "Compare SF and NYC weather."},
        {
            "role": "assistant",
            "reasoning_content": "I need two calls.",
            "content": "Checking.",
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
        {"role": "tool", "content": "SF: 72F"},
        {"role": "tool", "content": "NYC: 68F"},
        {
            "role": "assistant",
            "reasoning_content": "SF is four degrees warmer.",
            "content": "San Francisco is warmer.",
        },
    ]
    normalized = prepare_messages_with_tools(
        messages,
        renderer=renderer,
        tools=_TOOLS,
    )

    model_input, weights = renderer.build_supervised_example(
        normalized,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    renderer_tokens = list(model_input.to_ints())
    hf_result = tokenizer.apply_chat_template(
        messages,
        tools=_TOOLS,
        tokenize=True,
        add_generation_prompt=False,
        preserve_thinking=True,
    )
    hf_tokens = list(
        hf_result.input_ids if hasattr(hf_result, "input_ids") else hf_result
    )

    assert renderer_tokens == hf_tokens
    renderer_text = tokenizer.decode(
        renderer_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    hf_text = tokenizer.apply_chat_template(
        messages,
        tools=_TOOLS,
        tokenize=False,
        add_generation_prompt=False,
        preserve_thinking=True,
    )
    assert renderer_text.encode("utf-8") == hf_text.encode("utf-8")
    assert len(weights) == len(renderer_tokens)
    assert weights.max().item() == 1
    assert weights.min().item() == 0
    newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
    assert renderer_tokens[-len(newline_tokens) :] == newline_tokens
    assert weights[-len(newline_tokens) :].tolist() == [0] * len(newline_tokens)

    # Sampling at each assistant turn must keep the previous prompt as an
    # exact token prefix, including across a grouped parallel-tool response.
    first_prompt = list(
        renderer.build_generation_prompt(normalized[:2], role="assistant").to_ints()
    )
    second_prompt = list(
        renderer.build_generation_prompt(normalized[:5], role="assistant").to_ints()
    )
    assert second_prompt[: len(first_prompt)] == first_prompt
