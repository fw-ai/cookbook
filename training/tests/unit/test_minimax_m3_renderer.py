"""Parity tests for the MiniMax M3 renderer and official HF template."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import transformers
from tinker_cookbook.renderers.base import ParseTermination, ToolCall, TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_model_input_weights

from training.renderer.minimax_m3 import MiniMaxM3Renderer
from training.utils.supervised import build_tool_prefixed_messages

_LOCAL_PATH = "/shared/MiniMax-M3"
_HF_REPO = "MiniMaxAI/MiniMax-M3"
TOKENIZER_MODEL = _LOCAL_PATH if Path(_LOCAL_PATH).exists() else _HF_REPO


@pytest.fixture(scope="module")
def tokenizer() -> transformers.PreTrainedTokenizerBase:
    return transformers.AutoTokenizer.from_pretrained(TOKENIZER_MODEL)


def _hf_tokens(
    tokenizer: transformers.PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    *,
    thinking_mode: str,
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] | None = None,
) -> list[int]:
    result = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        thinking_mode=thinking_mode,
    )
    return list(result.input_ids if hasattr(result, "input_ids") else result)


def _tool_call(name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(
        function=ToolCall.FunctionBody(
            name=name,
            arguments=json.dumps(arguments),
        )
    )


def _find_unique_token_span(tokens: list[int], subsequence: list[int]) -> slice:
    starts = [
        index
        for index in range(len(tokens) - len(subsequence) + 1)
        if tokens[index : index + len(subsequence)] == subsequence
    ]
    assert len(starts) == 1
    return slice(starts[0], starts[0] + len(subsequence))


@pytest.mark.parametrize("thinking_mode", ["enabled", "disabled", "adaptive"])
def test_generation_prompt_matches_hf(
    tokenizer: transformers.PreTrainedTokenizerBase,
    thinking_mode: str,
) -> None:
    messages = [{"role": "user", "content": "Hello"}]
    expected = _hf_tokens(
        tokenizer,
        messages,
        thinking_mode=thinking_mode,
        add_generation_prompt=True,
    )
    actual = list(
        MiniMaxM3Renderer(tokenizer, thinking_mode)
        .build_generation_prompt(messages)
        .to_ints()
    )
    assert actual == expected


def test_root_developer_and_all_turn_thinking_match_hf(
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> None:
    hf_messages = [
        {"role": "root", "content": "Root policy"},
        {"role": "developer", "content": "Be concise"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "reasoning_content": "r1", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "<mm:think>r2</mm:think>a2"},
    ]
    messages = [
        {"role": "root", "content": "Root policy"},
        {"role": "developer", "content": "Be concise"},
        {"role": "user", "content": "u1"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "r1"},
                {"type": "text", "text": "a1"},
            ],
        },
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "<mm:think>r2</mm:think>a2"},
    ]
    expected = _hf_tokens(
        tokenizer,
        hf_messages,
        thinking_mode="enabled",
        add_generation_prompt=False,
    )
    actual = list(
        MiniMaxM3Renderer(tokenizer, "enabled")
        .build_supervised_example(
            messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )[0]
        .to_ints()
    )
    assert actual == expected


@pytest.mark.parametrize("thinking_mode", ["enabled", "disabled", "adaptive"])
def test_supervised_mask_trains_selected_thinking_and_answer(
    tokenizer: transformers.PreTrainedTokenizerBase,
    thinking_mode: str,
) -> None:
    messages = [
        {"role": "user", "content": "u1"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "r1"},
                {"type": "text", "text": "a1"},
            ],
        },
        {"role": "user", "content": "u2"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "r2"},
                {"type": "text", "text": "a2"},
            ],
        },
    ]
    renderer = MiniMaxM3Renderer(tokenizer, thinking_mode)

    model_input, weights = renderer.build_supervised_example(
        messages,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    tokens = list(model_input.to_ints())
    weight_values = weights.tolist()
    first_assistant = _find_unique_token_span(
        tokens,
        list(
            tokenizer.encode(
                "<mm:think>r1</mm:think>a1[e~[\n",
                add_special_tokens=False,
            )
        ),
    )
    last_assistant = _find_unique_token_span(
        tokens,
        list(
            tokenizer.encode(
                "<mm:think>r2</mm:think>a2[e~[\n",
                add_special_tokens=False,
            )
        ),
    )

    expected_last_only = [0.0] * len(tokens)
    expected_last_only[last_assistant] = [1.0] * len(expected_last_only[last_assistant])
    if thinking_mode == "enabled":
        # ``<mm:think>`` is part of the enabled-mode generation prefix, so it
        # remains in the token stream for HF parity but is not a loss target.
        expected_last_only[last_assistant.start] = 0.0
    assert weight_values == expected_last_only

    all_input, all_weights = renderer.build_supervised_example(
        messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    assert list(all_input.to_ints()) == tokens
    expected_all_assistant = list(expected_last_only)
    expected_all_assistant[first_assistant] = [1.0] * len(
        expected_all_assistant[first_assistant]
    )
    if thinking_mode == "enabled":
        expected_all_assistant[first_assistant.start] = 0.0
    assert all_weights.tolist() == expected_all_assistant

    datum = datum_from_model_input_weights(model_input, weights)
    assert datum.loss_fn_inputs["target_tokens"].data == tokens[1:]
    assert datum.loss_fn_inputs["weights"].data == weight_values[1:]

    customized_messages = [
        {**message, "trainable": message is messages[-1]} for message in messages
    ]
    customized_input, customized_weights = renderer.build_supervised_example(
        customized_messages,
        train_on_what=TrainOnWhat.CUSTOMIZED,
    )
    assert list(customized_input.to_ints()) == tokens
    assert customized_weights.tolist() == expected_last_only


@pytest.mark.parametrize(
    ("thinking_mode", "assistant_content", "prefilled_marker"),
    [
        (
            "enabled",
            [
                {"type": "thinking", "thinking": "reason"},
                {"type": "text", "text": "answer"},
            ],
            "<mm:think>",
        ),
        ("disabled", "answer", "</mm:think>"),
    ],
)
def test_supervised_mask_excludes_prefilled_thinking_marker(
    tokenizer: transformers.PreTrainedTokenizerBase,
    thinking_mode: str,
    assistant_content: Any,
    prefilled_marker: str,
) -> None:
    prompt_messages = [{"role": "user", "content": "question"}]
    messages = prompt_messages + [{"role": "assistant", "content": assistant_content}]
    renderer = MiniMaxM3Renderer(tokenizer, thinking_mode)

    generation_prompt = list(
        renderer.build_generation_prompt(prompt_messages).to_ints()
    )
    model_input, weights = renderer.build_supervised_example(
        messages,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    tokens = list(model_input.to_ints())
    hf_assistant: dict[str, Any] = {"role": "assistant", "content": "answer"}
    if thinking_mode == "enabled":
        hf_assistant["reasoning_content"] = "reason"
    expected_tokens = _hf_tokens(
        tokenizer,
        prompt_messages + [hf_assistant],
        thinking_mode=thinking_mode,
        add_generation_prompt=False,
    )
    marker_tokens = list(tokenizer.encode(prefilled_marker, add_special_tokens=False))

    assert tokens == expected_tokens
    assert tokens[: len(generation_prompt)] == generation_prompt
    assert generation_prompt[-len(marker_tokens) :] == marker_tokens
    marker_start = len(generation_prompt) - len(marker_tokens)
    assert weights[marker_start : len(generation_prompt)].tolist() == [0.0] * len(
        marker_tokens
    )
    assert weights[len(generation_prompt)].item() == 1.0


def test_nested_tool_arguments_match_hf(
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> None:
    function = {
        "name": "create_event",
        "description": "Create event",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "meta": {"type": "object"},
            },
        },
    }
    arguments = {
        "title": "Sync",
        "attendees": ["a", "b"],
        "meta": {"room": "1", "optional": None},
    }
    hf_messages = [
        {"role": "system", "content": "Be useful"},
        {"role": "user", "content": "book"},
        {
            "role": "assistant",
            "reasoning_content": "plan",
            "content": "Doing it",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "create_event", "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "content": "ok"},
    ]
    expected = _hf_tokens(
        tokenizer,
        hf_messages,
        thinking_mode="disabled",
        add_generation_prompt=True,
        tools=[{"type": "function", "function": function}],
    )

    renderer = MiniMaxM3Renderer(tokenizer, "disabled")
    messages = renderer.create_conversation_prefix_with_tools(
        [function], "Be useful"
    ) + [
        {"role": "user", "content": "book"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "plan"},
                {"type": "text", "text": "Doing it"},
            ],
            "tool_calls": [_tool_call("create_event", arguments)],
        },
        {"role": "tool", "content": "ok"},
    ]
    first = list(renderer.build_generation_prompt(messages).to_ints())
    second = list(renderer.build_generation_prompt(messages).to_ints())
    assert first == expected
    assert second == expected


@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_tools_with_root_and_developer_match_hf_production_path(
    tokenizer: transformers.PreTrainedTokenizerBase,
    add_generation_prompt: bool,
) -> None:
    function = {
        "name": "lookup",
        "description": "Look up a value",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    }
    tools = [{"type": "function", "function": function}]
    messages: list[dict[str, Any]] = [
        {"role": "root", "content": "Root policy"},
        {"role": "developer", "content": "Be concise"},
        {"role": "user", "content": "look this up"},
    ]
    if not add_generation_prompt:
        messages.append(
            {
                "role": "assistant",
                "reasoning_content": "check the tool",
                "content": "done",
            }
        )
    expected = _hf_tokens(
        tokenizer,
        messages,
        thinking_mode="adaptive",
        add_generation_prompt=add_generation_prompt,
        tools=tools,
    )

    renderer = MiniMaxM3Renderer(tokenizer, "adaptive")
    renderer_messages = build_tool_prefixed_messages(
        messages,
        renderer=renderer,
        tools=tools,
    )

    if add_generation_prompt:
        actual = list(renderer.build_generation_prompt(renderer_messages).to_ints())
    else:
        actual = list(
            renderer.build_supervised_example(
                renderer_messages,
                train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )[0].to_ints()
        )

    assert actual == expected


@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_tools_with_leading_system_then_root_match_hf_production_path(
    tokenizer: transformers.PreTrainedTokenizerBase,
    add_generation_prompt: bool,
) -> None:
    function = {
        "name": "lookup",
        "description": "Look up a value",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    }
    tools = [{"type": "function", "function": function}]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "System developer policy"},
        {"role": "root", "content": "This mid-conversation root is ignored"},
        {"role": "user", "content": "look this up"},
    ]
    if not add_generation_prompt:
        messages.append({"role": "assistant", "content": "done"})
    expected = _hf_tokens(
        tokenizer,
        messages,
        thinking_mode="adaptive",
        add_generation_prompt=add_generation_prompt,
        tools=tools,
    )

    renderer = MiniMaxM3Renderer(tokenizer, "adaptive")
    renderer_messages = build_tool_prefixed_messages(
        messages,
        renderer=renderer,
        tools=tools,
    )
    if add_generation_prompt:
        actual = list(renderer.build_generation_prompt(renderer_messages).to_ints())
    else:
        actual = list(renderer.build_supervised_example(renderer_messages)[0].to_ints())

    assert actual == expected


@pytest.mark.parametrize(
    ("thinking_mode", "response_text", "expected_content"),
    [
        (
            "enabled",
            "reasoning</mm:think>answer[e~[",
            [
                {"type": "thinking", "thinking": "reasoning"},
                {"type": "text", "text": "answer"},
            ],
        ),
        (
            "enabled",
            "<mm:think>reasoning</mm:think>answer[e~[",
            [
                {"type": "thinking", "thinking": "reasoning"},
                {"type": "text", "text": "answer"},
            ],
        ),
        ("enabled", "answer[e~[", [{"type": "text", "text": "answer"}]),
        ("disabled", "answer[e~[", [{"type": "text", "text": "answer"}]),
        (
            "disabled",
            "</mm:think>answer[e~[",
            [{"type": "text", "text": "answer"}],
        ),
    ],
)
def test_parse_response_normalizes_prefilled_thinking_once(
    tokenizer: transformers.PreTrainedTokenizerBase,
    thinking_mode: str,
    response_text: str,
    expected_content: list[dict[str, str]],
) -> None:
    renderer = MiniMaxM3Renderer(tokenizer, thinking_mode)
    response = list(tokenizer.encode(response_text, add_special_tokens=False))

    parsed, termination = renderer.parse_response(response)

    assert termination == ParseTermination.STOP_SEQUENCE
    assert parsed["content"] == expected_content


def test_parse_nested_tool_call_roundtrip(
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> None:
    renderer = MiniMaxM3Renderer(tokenizer, "adaptive")
    arguments = {"items": [{"name": "a", "enabled": True}], "count": 1}
    rendered = renderer.render_message(
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "plan"}],
            "tool_calls": [_tool_call("batch", arguments)],
        },
        ctx=None,  # type: ignore[arg-type]
    )
    response = [token for chunk in rendered.output for token in chunk.tokens]
    parsed, termination = renderer.parse_response(response)
    assert termination == ParseTermination.STOP_SEQUENCE
    assert parsed["tool_calls"][0].function.name == "batch"
    assert json.loads(parsed["tool_calls"][0].function.arguments) == arguments
