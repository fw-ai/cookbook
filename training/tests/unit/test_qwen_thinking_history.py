"""Pinned-template regression tests for Qwen thinking-history rendering.

These are synthetic mechanism tests built from the official Qwen chat-template
state machine.  They specifically cover the distinction between the last real
user query and the last assistant message, including a multi-assistant tool
trajectory in one user turn.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import pytest
from tinker_cookbook.renderers import ToolCall, TrainOnWhat, get_renderer
from transformers import AutoTokenizer

import training.renderer._qwen3_split  # noqa: F401  (renderer registration)
from training.utils.supervised import normalize_messages


_QWEN_INTERLEAVED_RENDERERS = [
    pytest.param(
        "qwen3_5_interleaved",
        "Qwen/Qwen3.5-35B-A3B",
        "59d61f3ce65a6d9863b86d2e96597125219dc754",
        {},
        id="qwen3.5-interleaved",
    ),
    pytest.param(
        "qwen3_5_disable_thinking_interleaved",
        "Qwen/Qwen3.5-35B-A3B",
        "59d61f3ce65a6d9863b86d2e96597125219dc754",
        {"enable_thinking": False},
        id="qwen3.5-interleaved-thinking-disabled",
    ),
    pytest.param(
        "qwen3_6_interleaved",
        "Qwen/Qwen3.6-27B",
        "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        {},
        id="qwen3.6-interleaved",
    ),
    pytest.param(
        "qwen3_6_disable_thinking_interleaved",
        "Qwen/Qwen3.6-27B",
        "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        {"enable_thinking": False},
        id="qwen3.6-interleaved-thinking-disabled",
    ),
]


@lru_cache(maxsize=None)
def _load_pinned_tokenizer(model: str, revision: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"pinned tokenizer unavailable for {model}@{revision}: {exc}")
    if not getattr(tokenizer, "chat_template", None):
        pytest.skip(f"{model}@{revision} has no chat template")
    return tokenizer


def _tool_call(name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(
        function=ToolCall.FunctionBody(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        )
    )


def _renderer_trajectory(*, prewrapped_tool_response: bool) -> list[dict[str, Any]]:
    tool_result = (
        {
            "role": "user",
            "content": "<tool_response>\ntool result\n</tool_response>",
        }
        if prewrapped_tool_response
        else {"role": "tool", "content": "tool result"}
    )
    return [
        {"role": "user", "content": "u1"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "thinking1"},
                {"type": "text", "text": "a1"},
            ],
        },
        {"role": "user", "content": "u2"},
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "thinking2"}],
            "tool_calls": [_tool_call("lookup", {"query": "value"})],
        },
        tool_result,
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "thinking3"},
                {"type": "text", "text": "a3"},
            ],
        },
    ]


def _hf_trajectory(*, prewrapped_tool_response: bool) -> list[dict[str, Any]]:
    tool_result = (
        {
            "role": "user",
            "content": "<tool_response>\ntool result\n</tool_response>",
        }
        if prewrapped_tool_response
        else {"role": "tool", "content": "tool result"}
    )
    return [
        {"role": "user", "content": "u1"},
        {
            "role": "assistant",
            "reasoning_content": "thinking1",
            "content": "a1",
        },
        {"role": "user", "content": "u2"},
        {
            "role": "assistant",
            "reasoning_content": "thinking2",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "lookup",
                        "arguments": {"query": "value"},
                    }
                }
            ],
        },
        tool_result,
        {
            "role": "assistant",
            "reasoning_content": "thinking3",
            "content": "a3",
        },
    ]


def _renderer_tokens(tokenizer, renderer_name: str, messages: list[dict]) -> list[int]:
    model_input = get_renderer(renderer_name, tokenizer).build_generation_prompt(
        messages, role="assistant"
    )
    return [int(token) for token in model_input.to_ints()]


def _hf_tokens(tokenizer, messages: list[dict], **kwargs: Any) -> list[int]:
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **kwargs,
    )
    input_ids = result.input_ids if hasattr(result, "input_ids") else result
    input_ids = list(input_ids)
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token) for token in input_ids]


def _assert_token_parity(tokenizer, actual: list[int], expected: list[int]) -> None:
    if actual == expected:
        return
    shared = min(len(actual), len(expected))
    first_diff = next(
        (index for index in range(shared) if actual[index] != expected[index]),
        shared,
    )
    pytest.fail(
        "renderer diverged from pinned official template at token "
        f"{first_diff}:\n"
        f"renderer={tokenizer.decode(actual, skip_special_tokens=False)!r}\n"
        f"official={tokenizer.decode(expected, skip_special_tokens=False)!r}"
    )


@pytest.mark.parametrize(
    ("renderer_name", "model", "revision"),
    [
        (
            "qwen3_5_interleaved",
            "Qwen/Qwen3.5-35B-A3B",
            "59d61f3ce65a6d9863b86d2e96597125219dc754",
        ),
        (
            "qwen3_6_interleaved",
            "Qwen/Qwen3.6-27B",
            "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        ),
    ],
)
def test_empty_reasoning_does_not_hide_reasoning_content(
    renderer_name: str,
    model: str,
    revision: str,
) -> None:
    tokenizer = _load_pinned_tokenizer(model, revision)
    raw_messages = [
        {"role": "user", "content": "u1"},
        {
            "role": "assistant",
            "reasoning": "",
            "reasoning_content": "QWEN_REASONING_CONTENT",
            "content": "a1",
        },
    ]

    actual = _renderer_tokens(
        tokenizer,
        renderer_name,
        normalize_messages(raw_messages),
    )
    expected = _hf_tokens(tokenizer, raw_messages)

    _assert_token_parity(tokenizer, actual, expected)
    assert "QWEN_REASONING_CONTENT" in tokenizer.decode(
        actual, skip_special_tokens=False
    )


@pytest.mark.parametrize(
    ("renderer_name", "model", "revision", "template_kwargs"),
    _QWEN_INTERLEAVED_RENDERERS,
)
@pytest.mark.parametrize("prewrapped_tool_response", [False, True])
@pytest.mark.timeout(180)
def test_clear_matches_last_real_user_boundary_and_pinned_template(
    renderer_name: str,
    model: str,
    revision: str,
    template_kwargs: dict[str, bool],
    prewrapped_tool_response: bool,
) -> None:
    """INTERLEAVED keeps every assistant in the post-last-user tool trajectory."""

    tokenizer = _load_pinned_tokenizer(model, revision)
    actual = _renderer_tokens(
        tokenizer,
        renderer_name,
        _renderer_trajectory(
            prewrapped_tool_response=prewrapped_tool_response,
        ),
    )
    expected = _hf_tokens(
        tokenizer,
        _hf_trajectory(prewrapped_tool_response=prewrapped_tool_response),
        **template_kwargs,
    )

    _assert_token_parity(tokenizer, actual, expected)
    decoded = tokenizer.decode(actual, skip_special_tokens=False)
    assert "thinking1" not in decoded
    assert "thinking2" in decoded
    assert "thinking3" in decoded


def _weighted_text(tokenizer, example) -> tuple[str, str]:
    model_input, weights = example
    token_ids = [int(token) for token in model_input.to_ints()]
    weight_values = weights.tolist()
    trained = [
        token
        for token, weight in zip(token_ids, weight_values, strict=True)
        if weight > 0
    ]
    masked = [
        token
        for token, weight in zip(token_ids, weight_values, strict=True)
        if weight == 0
    ]
    return (
        tokenizer.decode(trained, skip_special_tokens=False),
        tokenizer.decode(masked, skip_special_tokens=False),
    )


@pytest.mark.parametrize(
    ("renderer_name", "model", "revision", "template_kwargs"),
    _QWEN_INTERLEAVED_RENDERERS,
)
@pytest.mark.parametrize("prewrapped_tool_response", [False, True])
@pytest.mark.timeout(180)
def test_clear_unrolls_at_real_user_boundaries_without_splitting_tool_trajectory(
    renderer_name: str,
    model: str,
    revision: str,
    template_kwargs: dict[str, bool],
    prewrapped_tool_response: bool,
) -> None:
    del template_kwargs  # Only generation rendering consumes this vendor flag.
    tokenizer = _load_pinned_tokenizer(model, revision)
    renderer = get_renderer(renderer_name, tokenizer)
    examples = renderer.build_supervised_examples(
        _renderer_trajectory(
            prewrapped_tool_response=prewrapped_tool_response,
        ),
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    assert renderer.has_extension_property is False
    assert len(examples) == 2

    first_trained, first_masked = _weighted_text(tokenizer, examples[0])
    second_trained, second_masked = _weighted_text(tokenizer, examples[1])
    assert all(value in first_trained for value in ("thinking1", "a1"))
    assert "u1" in first_masked

    assert all(
        value in second_trained for value in ("thinking2", "lookup", "thinking3", "a3")
    )
    assert all(value not in second_trained for value in ("thinking1", "a1"))
    assert all(value in second_masked for value in ("u2", "tool result"))


@pytest.mark.parametrize(
    ("enabled_renderer", "disabled_renderer", "model", "revision"),
    [
        pytest.param(
            "qwen3_5_interleaved",
            "qwen3_5_disable_thinking_interleaved",
            "Qwen/Qwen3.5-35B-A3B",
            "59d61f3ce65a6d9863b86d2e96597125219dc754",
            id="qwen3.5",
        ),
        pytest.param(
            "qwen3_6_interleaved",
            "qwen3_6_disable_thinking_interleaved",
            "Qwen/Qwen3.6-27B",
            "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
            id="qwen3.6",
        ),
    ],
)
@pytest.mark.timeout(180)
def test_thinking_enablement_is_orthogonal_to_clear_history(
    enabled_renderer: str,
    disabled_renderer: str,
    model: str,
    revision: str,
) -> None:
    tokenizer = _load_pinned_tokenizer(model, revision)
    messages = _renderer_trajectory(prewrapped_tool_response=True)
    enabled = get_renderer(enabled_renderer, tokenizer)
    disabled = get_renderer(disabled_renderer, tokenizer)

    enabled_input, enabled_weights = enabled.build_supervised_example(
        messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    disabled_input, disabled_weights = disabled.build_supervised_example(
        messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    assert list(enabled_input.to_ints()) == list(disabled_input.to_ints())
    assert enabled_weights.tolist() == disabled_weights.tolist()

    enabled_text = tokenizer.decode(
        _renderer_tokens(tokenizer, enabled_renderer, messages),
        skip_special_tokens=False,
    )
    disabled_text = tokenizer.decode(
        _renderer_tokens(tokenizer, disabled_renderer, messages),
        skip_special_tokens=False,
    )
    enabled_suffix = "\n<|im_start|>assistant\n<think>\n"
    disabled_suffix = "\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert enabled_text.endswith(enabled_suffix)
    assert disabled_text.endswith(disabled_suffix)
    assert enabled_text.removesuffix(enabled_suffix) == disabled_text.removesuffix(
        disabled_suffix
    )


@pytest.mark.parametrize("prewrapped_tool_response", [False, True])
@pytest.mark.timeout(180)
def test_qwen36_preserve_is_prefix_extending_and_does_not_unroll(
    prewrapped_tool_response: bool,
) -> None:
    tokenizer = _load_pinned_tokenizer(
        "Qwen/Qwen3.6-27B",
        "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    )
    renderer = get_renderer("qwen3_6_preserved", tokenizer)
    messages = _renderer_trajectory(
        prewrapped_tool_response=prewrapped_tool_response,
    )

    actual = _renderer_tokens(tokenizer, "qwen3_6_preserved", messages)
    expected = _hf_tokens(
        tokenizer,
        _hf_trajectory(prewrapped_tool_response=prewrapped_tool_response),
        preserve_thinking=True,
    )
    _assert_token_parity(tokenizer, actual, expected)

    examples = renderer.build_supervised_examples(
        messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    assert renderer.has_extension_property is True
    assert len(examples) == 1

    first_turn = renderer.build_supervised_example(
        messages[:2],
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )[0].to_ints()
    full = renderer.build_supervised_example(
        messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )[0].to_ints()
    assert list(full[: len(first_turn)]) == list(first_turn)
    decoded = tokenizer.decode(full, skip_special_tokens=False)
    assert all(value in decoded for value in ("thinking1", "thinking2", "thinking3"))


@pytest.mark.parametrize(
    ("renderer_name", "template_kwargs"),
    [
        pytest.param("qwen3_6_interleaved", {}, id="interleaved"),
        pytest.param(
            "qwen3_6_preserved",
            {"preserve_thinking": True},
            id="preserved",
        ),
        pytest.param(
            "qwen3_6_disable_thinking_interleaved",
            {"enable_thinking": False},
            id="thinking-disabled-separate-axis",
        ),
    ],
)
@pytest.mark.timeout(180)
def test_qwen36_non_string_tool_arguments_use_json_serialization(
    renderer_name: str,
    template_kwargs: dict[str, bool],
) -> None:
    tokenizer = _load_pinned_tokenizer(
        "Qwen/Qwen3.6-27B",
        "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    )
    arguments = {
        "string": "café",
        "boolean": True,
        "null": None,
        "list": ["café", 1],
        "object": {"enabled": False},
    }
    renderer_messages = [
        {"role": "user", "content": "call the tool"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "inspect each argument"},
                {"type": "text", "text": "calling"},
            ],
            "tool_calls": [_tool_call("inspect", arguments)],
        },
    ]
    hf_messages = [
        {"role": "user", "content": "call the tool"},
        {
            "role": "assistant",
            "reasoning_content": "inspect each argument",
            "content": "calling",
            "tool_calls": [
                {
                    "function": {
                        "name": "inspect",
                        "arguments": arguments,
                    }
                }
            ],
        },
    ]

    actual = _renderer_tokens(tokenizer, renderer_name, renderer_messages)
    expected = _hf_tokens(tokenizer, hf_messages, **template_kwargs)
    _assert_token_parity(tokenizer, actual, expected)

    decoded = tokenizer.decode(actual, skip_special_tokens=False)
    assert "<parameter=boolean>\ntrue\n</parameter>" in decoded
    assert "<parameter=null>\nnull\n</parameter>" in decoded
    assert '"café"' in decoded
