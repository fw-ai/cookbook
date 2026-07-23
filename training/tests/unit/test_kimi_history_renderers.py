"""Focused history-mode tests for Kimi K2.5, K2.6, and K2.7 Code."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

import training.renderer  # noqa: F401  (registers cookbook-local Kimi renderers)
from training.utils.supervised import build_tool_prefixed_messages, normalize_messages
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import ToolCall, TrainOnWhat


class _ReversibleTokenizer:
    """Tiny tokenizer that preserves Kimi's single-token delimiter contract."""

    name_or_path = "unit-test/kimi"
    _special_to_id = {
        "<think>": 1,
        "</think>": 2,
        "<|im_end|>": 3,
    }
    _id_to_special = {value: key for key, value in _special_to_id.items()}

    def encode(self, text: str, **_kwargs) -> list[int]:
        tokens: list[int] = []
        cursor = 0
        while cursor < len(text):
            special = next(
                (
                    candidate
                    for candidate in self._special_to_id
                    if text.startswith(candidate, cursor)
                ),
                None,
            )
            if special is not None:
                tokens.append(self._special_to_id[special])
                cursor += len(special)
            else:
                tokens.append(ord(text[cursor]) + 1_000)
                cursor += 1
        return tokens

    def decode(self, token_ids: Iterable[int], **_kwargs) -> str:
        text: list[str] = []
        for raw_token_id in token_ids:
            token_id = int(raw_token_id)
            if token_id in self._id_to_special:
                text.append(self._id_to_special[token_id])
            else:
                text.append(chr(token_id - 1_000))
        return "".join(text)


@pytest.fixture
def tokenizer() -> _ReversibleTokenizer:
    return _ReversibleTokenizer()


def _assistant(thinking: str, answer: str, *, tool_call: bool = False) -> dict:
    message: dict = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": thinking},
            {"type": "text", "text": answer},
        ],
    }
    if tool_call:
        message["tool_calls"] = [
            ToolCall(
                id="functions.lookup:0",
                function={"name": "lookup", "arguments": '{"q":"x"}'},
            )
        ]
    return message


def _tool_trajectory() -> list[dict]:
    return [
        {"role": "user", "content": "U1"},
        _assistant("TRACE_1", "ANSWER_1"),
        {"role": "user", "content": "U2"},
        _assistant("TRACE_2", "CALL_TOOL", tool_call=True),
        {
            "role": "tool",
            "name": "lookup",
            "tool_call_id": "functions.lookup:0",
            "content": "TOOL_RESULT",
        },
        _assistant("TRACE_3", "ANSWER_3"),
    ]


def _decode_input(tokenizer: _ReversibleTokenizer, model_input) -> str:
    return tokenizer.decode(model_input.to_ints())


def _decode_trained(
    tokenizer: _ReversibleTokenizer,
    model_input,
    weights,
) -> str:
    return tokenizer.decode(
        token
        for token, weight in zip(model_input.to_ints(), weights.tolist(), strict=True)
        if weight > 0
    )


@pytest.mark.parametrize(
    "renderer_name",
    [
        "kimi_k25_interleaved",
        "kimi_k26_interleaved",
        "kimi_k26_preserve_thinking",
        "kimi_k27_code_preserved",
    ],
)
def test_official_templates_do_not_inject_default_system_message(
    tokenizer: _ReversibleTokenizer,
    renderer_name: str,
) -> None:
    renderer = get_renderer(renderer_name, tokenizer)

    rendered = _decode_input(
        tokenizer,
        renderer.build_generation_prompt([{"role": "user", "content": "hello"}]),
    )

    assert "<|im_system|>" not in rendered
    assert "You are Kimi" not in rendered
    assert renderer.create_conversation_prefix_with_tools([], system_prompt="") == []
    tool_prefix = renderer.create_conversation_prefix_with_tools(
        [
            {
                "name": "lookup",
                "description": "Look up a value.",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        system_prompt="",
    )
    assert [message["role"] for message in tool_prefix] == ["tool_declare"]


@pytest.mark.parametrize(
    "renderer_name",
    [
        "kimi_k25_interleaved",
        "kimi_k26_interleaved",
        "kimi_k26_preserve_thinking",
    ],
)
def test_tools_distinguish_no_system_from_explicit_empty_system(
    tokenizer: _ReversibleTokenizer,
    renderer_name: str,
) -> None:
    renderer = get_renderer(renderer_name, tokenizer)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    without_system = build_tool_prefixed_messages(
        [{"role": "user", "content": "hello"}],
        renderer=renderer,
        tools=tools,
    )
    with_empty_system = build_tool_prefixed_messages(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": "hello"},
        ],
        renderer=renderer,
        tools=tools,
    )

    assert [message["role"] for message in without_system] == [
        "tool_declare",
        "user",
    ]
    assert [message["role"] for message in with_empty_system] == [
        "tool_declare",
        "system",
        "user",
    ]
    assert with_empty_system[1]["content"] == ""

    without_system_rendered = _decode_input(
        tokenizer,
        renderer.build_generation_prompt(without_system),
    )
    with_empty_system_rendered = _decode_input(
        tokenizer,
        renderer.build_generation_prompt(with_empty_system),
    )
    empty_system_marker = "<|im_system|>system<|im_middle|><|im_end|>"
    assert empty_system_marker not in without_system_rendered
    assert empty_system_marker in with_empty_system_rendered


@pytest.mark.parametrize(
    "renderer_name", ["kimi_k25_interleaved", "kimi_k26_interleaved"]
)
def test_clear_history_unrolls_by_user_turn_and_keeps_terminal_tool_trajectory(
    tokenizer: _ReversibleTokenizer,
    renderer_name: str,
) -> None:
    renderer = get_renderer(renderer_name, tokenizer)
    assert renderer.has_extension_property is False
    assert renderer.strip_thinking_from_history is True

    examples = renderer.build_supervised_examples(
        _tool_trajectory(),
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    assert len(examples) == 2
    first_input, first_weights = examples[0]
    second_input, second_weights = examples[1]
    first_trained = _decode_trained(tokenizer, first_input, first_weights)
    second_rendered = _decode_input(tokenizer, second_input)
    second_trained = _decode_trained(tokenizer, second_input, second_weights)

    assert "TRACE_1" in first_trained
    assert "ANSWER_1" in first_trained
    assert "TRACE_2" not in first_trained
    assert "TRACE_3" not in first_trained

    # The first completed turn is historical in datum 1 and loses its trace.
    assert "TRACE_1" not in second_rendered
    assert "<think></think>ANSWER_1" in second_rendered

    # A2 -> tool -> A3 is one user turn.  The SFT terminal-target adaptation
    # intentionally keeps and trains both assistant traces in that trajectory.
    assert "TRACE_2" in second_trained
    assert "CALL_TOOL" in second_trained
    assert "functions.lookup:0" in second_trained
    assert "TRACE_3" in second_trained
    assert "ANSWER_3" in second_trained


@pytest.mark.parametrize(
    "renderer_name", ["kimi_k25_interleaved", "kimi_k26_interleaved"]
)
def test_clear_generation_uses_trailing_trajectory_not_only_last_assistant(
    tokenizer: _ReversibleTokenizer,
    renderer_name: str,
) -> None:
    renderer = get_renderer(renderer_name, tokenizer)

    # With a pending tool result, the last completed non-tool assistant is A1.
    # The whole A2 -> tool suffix is current and therefore retains TRACE_2.
    pending_tool_result = _tool_trajectory()[:-1]
    pending_rendered = _decode_input(
        tokenizer,
        renderer.build_generation_prompt(pending_tool_result),
    )
    assert "TRACE_1" not in pending_rendered
    assert "TRACE_2" in pending_rendered

    # Once A3 completes the trajectory, the official clear template places
    # A3 itself (and everything before it) in history.
    complete_rendered = _decode_input(
        tokenizer,
        renderer.build_generation_prompt(_tool_trajectory()),
    )
    assert "TRACE_1" not in complete_rendered
    assert "TRACE_2" not in complete_rendered
    assert "TRACE_3" not in complete_rendered


@pytest.mark.parametrize(
    "renderer_name",
    ["kimi_k26_preserve_thinking", "kimi_k27_code_preserved"],
)
def test_preserve_history_has_extension_property_and_emits_one_datum(
    tokenizer: _ReversibleTokenizer,
    renderer_name: str,
) -> None:
    renderer = get_renderer(renderer_name, tokenizer)
    assert renderer.has_extension_property is True
    assert renderer.strip_thinking_from_history is False

    examples = renderer.build_supervised_examples(
        _tool_trajectory(),
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    assert len(examples) == 1
    model_input, weights = examples[0]
    rendered = _decode_input(tokenizer, model_input)
    trained = _decode_trained(tokenizer, model_input, weights)
    for trace in ("TRACE_1", "TRACE_2", "TRACE_3"):
        assert trace in rendered
        assert trace in trained


def test_kimi_k26_preserve_is_the_official_preserve_thinking_branch(
    tokenizer: _ReversibleTokenizer,
) -> None:
    clear_renderer = get_renderer("kimi_k26_interleaved", tokenizer)
    preserve_renderer = get_renderer("kimi_k26_preserve_thinking", tokenizer)

    assert clear_renderer.strip_thinking_from_history is True
    # Tinker's constructor mapping for HF preserve_thinking=True.
    assert preserve_renderer.strip_thinking_from_history is False


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    ("renderer_name", "model", "revision", "template_kwargs"),
    [
        pytest.param(
            "kimi_k25_interleaved",
            "moonshotai/Kimi-K2.5",
            "4d01dfe0332d63057c186e0b262165819efb6611",
            {},
            id="k25-interleaved",
        ),
        pytest.param(
            "kimi_k26_interleaved",
            "moonshotai/Kimi-K2.6",
            "7eb5002f6aadc958aed6a9177b7ed26bb94011bb",
            {"preserve_thinking": False},
            id="k26-interleaved",
        ),
        pytest.param(
            "kimi_k26_preserve_thinking",
            "moonshotai/Kimi-K2.6",
            "7eb5002f6aadc958aed6a9177b7ed26bb94011bb",
            {"preserve_thinking": True},
            id="k26-preserved",
        ),
        pytest.param(
            "kimi_k27_code_preserved",
            "moonshotai/Kimi-K2.7-Code",
            "74797c9c62378b951a1f6fcf5c4631024e9b8bef",
            {},
            id="k27-preserved-only",
        ),
    ],
)
def test_generation_matches_pinned_official_chat_template(
    tokenizer: _ReversibleTokenizer,
    renderer_name: str,
    model: str,
    revision: str,
    template_kwargs: dict[str, bool],
) -> None:
    from huggingface_hub import hf_hub_download
    from transformers.utils.chat_template_utils import render_jinja_template

    try:
        template_path = hf_hub_download(
            model,
            filename="chat_template.jinja",
            revision=revision,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        pytest.skip(f"official template unavailable for {model}@{revision}: {exc}")

    renderer_messages = [
        {"role": "user", "content": "U1"},
        _assistant("TRACE_1", "ANSWER_1"),
        {"role": "user", "content": "U2"},
    ]
    hf_messages = [
        {"role": "user", "content": "U1"},
        {
            "role": "assistant",
            "reasoning_content": "TRACE_1",
            "content": "ANSWER_1",
        },
        {"role": "user", "content": "U2"},
    ]

    rendered_conversations, _assistant_masks = render_jinja_template(
        conversations=[hf_messages],
        chat_template=Path(template_path).read_text(),
        add_generation_prompt=True,
        **template_kwargs,
    )
    expected = rendered_conversations[0]
    actual = _decode_input(
        tokenizer,
        get_renderer(renderer_name, tokenizer).build_generation_prompt(
            renderer_messages
        ),
    )

    assert actual == expected


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    ("renderer_name", "model", "revision", "template_kwargs"),
    [
        pytest.param(
            "kimi_k26_preserve_thinking",
            "moonshotai/Kimi-K2.6",
            "7eb5002f6aadc958aed6a9177b7ed26bb94011bb",
            {"preserve_thinking": True},
            id="k26-preserved",
        ),
        pytest.param(
            "kimi_k27_code_preserved",
            "moonshotai/Kimi-K2.7-Code",
            "74797c9c62378b951a1f6fcf5c4631024e9b8bef",
            {},
            id="k27-preserved-only",
        ),
    ],
)
def test_empty_reasoning_field_wins_over_reasoning_content_like_official_template(
    tokenizer: _ReversibleTokenizer,
    renderer_name: str,
    model: str,
    revision: str,
    template_kwargs: dict[str, bool],
) -> None:
    from huggingface_hub import hf_hub_download
    from transformers.utils.chat_template_utils import render_jinja_template

    try:
        template_path = hf_hub_download(
            model,
            filename="chat_template.jinja",
            revision=revision,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        pytest.skip(f"official template unavailable for {model}@{revision}: {exc}")

    raw_messages = [
        {"role": "user", "content": "U1"},
        {
            "role": "assistant",
            "reasoning": "",
            "reasoning_content": "MUST_NOT_RENDER",
            "content": "A1",
        },
        {"role": "user", "content": "U2"},
    ]
    rendered_conversations, _assistant_masks = render_jinja_template(
        conversations=[raw_messages],
        chat_template=Path(template_path).read_text(),
        add_generation_prompt=True,
        **template_kwargs,
    )
    expected = rendered_conversations[0]
    actual = _decode_input(
        tokenizer,
        get_renderer(renderer_name, tokenizer).build_generation_prompt(
            normalize_messages(raw_messages)
        ),
    )

    assert actual == expected
    assert "MUST_NOT_RENDER" not in actual


@pytest.mark.parametrize(
    "renderer_name",
    ["kimi_k26_preserve_thinking", "kimi_k27_code_preserved"],
)
def test_preserve_generation_prompt_extends_prior_observation(
    tokenizer: _ReversibleTokenizer,
    renderer_name: str,
) -> None:
    renderer = get_renderer(renderer_name, tokenizer)
    first_turn = _tool_trajectory()[:2]
    later_prompt = _tool_trajectory()[:3]

    completed_first_turn = renderer.build_supervised_example(
        first_turn,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )[0].to_ints()
    second_turn_prompt = renderer.build_generation_prompt(later_prompt).to_ints()

    assert second_turn_prompt[: len(completed_first_turn)] == completed_first_turn
