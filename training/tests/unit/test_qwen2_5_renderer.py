"""Contract tests for the dedicated Qwen2.5 renderer using public HF main."""

from __future__ import annotations

import json
from typing import Any

import pytest
import transformers
from training._vendor.tinker_cookbook_0_4_3.renderers.base import Message, ParseTermination, TrainOnWhat

from training.renderer.qwen2_5 import (
    PRODUCTION_EOS_TOKEN,
    PRODUCTION_EOS_TOKEN_ID,
    Qwen2_5Renderer,
)
from training.utils.supervised import (
    build_tool_prefixed_messages,
    normalize_messages,
    render_preference_pair,
)


_HF_REPO = "Qwen/Qwen2.5-32B-Instruct"


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Look up weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


@pytest.fixture(scope="module")
def tokenizer() -> transformers.PreTrainedTokenizerBase:
    try:
        loaded = transformers.AutoTokenizer.from_pretrained(_HF_REPO)
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"Qwen2.5 tokenizer unavailable: {exc}")
    return loaded


def _hf_render(
    tokenizer: transformers.PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] | None = None,
    tokenize: bool,
) -> str | list[int]:
    result = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
    )
    if not tokenize:
        assert isinstance(result, str)
        return result
    if hasattr(result, "input_ids"):
        return [int(token) for token in result.input_ids]
    return [int(token) for token in result]


def _renderer_messages(
    renderer: Qwen2_5Renderer,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> list[Message]:
    return build_tool_prefixed_messages(messages, renderer=renderer, tools=tools)


def _assert_hf_parity(
    tokenizer: transformers.PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool,
) -> tuple[Qwen2_5Renderer, list[Message]]:
    renderer = Qwen2_5Renderer(tokenizer)
    normalized = _renderer_messages(renderer, messages, tools)
    expected_text = _hf_render(
        tokenizer,
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    expected_tokens = _hf_render(
        tokenizer,
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
    )

    assert (
        renderer.render_text(
            normalized,
            add_generation_prompt=add_generation_prompt,
        )
        == expected_text
    )
    if add_generation_prompt:
        actual = list(renderer.build_generation_prompt(normalized).to_ints())
    else:
        actual = list(renderer.build_supervised_example(normalized)[0].to_ints())
    assert actual == expected_tokens
    return renderer, normalized


def _trained_text(
    tokenizer: transformers.PreTrainedTokenizerBase,
    tokens: list[int],
    weights: list[float],
) -> str:
    return tokenizer.decode(
        [token for token, weight in zip(tokens, weights, strict=True) if weight > 0],
        skip_special_tokens=False,
    )


def test_public_hf_main_has_required_stop_token(
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> None:
    assert tokenizer.eos_token == PRODUCTION_EOS_TOKEN
    assert tokenizer.convert_tokens_to_ids(PRODUCTION_EOS_TOKEN) == (
        PRODUCTION_EOS_TOKEN_ID
    )


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "Hello"}],
        [
            {"role": "system", "content": "Answer tersely."},
            {"role": "user", "content": "Hello"},
        ],
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "FIRST_ANSWER"},
            {"role": "user", "content": "second"},
        ],
    ],
)
def test_generation_prompt_matches_public_hf_main(
    tokenizer: transformers.PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
) -> None:
    _assert_hf_parity(tokenizer, messages, add_generation_prompt=True)


def test_tool_roundtrip_parallel_results_and_unicode(
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                **_TOOLS[0]["function"],
                "description": "查询 <城市> 天气 & 时区's ☀️",
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_time",
                "description": "查时区",
                "parameters": {
                    "type": "object",
                    "properties": {"zone": {"type": "string"}},
                },
            },
        },
    ]
    messages = [
        {"role": "system", "content": "Use tools when needed."},
        {"role": "user", "content": "东京现在怎么样？"},
        {
            "role": "assistant",
            "content": "我来查。",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": {"city": "東京"},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_time",
                        "arguments": {"zone": "Asia/Tokyo"},
                    },
                },
            ],
        },
        {"role": "tool", "content": "晴れ"},
        {"role": "tool", "content": "14:00"},
    ]

    renderer = Qwen2_5Renderer(tokenizer)
    normalized = _renderer_messages(renderer, messages, tools)
    rendered = renderer.render_text(normalized, add_generation_prompt=True)
    assert '{{"name": <function-name>, "arguments": <args-json-object>}}' in rendered
    assert (
        "<|im_start|>user\n<tool_response>\n晴れ\n</tool_response>"
        "\n<tool_response>\n14:00\n</tool_response><|im_end|>\n"
    ) in rendered
    assert "東京" in rendered


def test_tools_distinguish_missing_and_explicit_empty_system(
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> None:
    without_system = [{"role": "user", "content": "weather"}]
    with_empty_system = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "weather"},
    ]
    renderer = Qwen2_5Renderer(tokenizer)
    missing_normalized = _renderer_messages(renderer, without_system, _TOOLS)
    empty_normalized = _renderer_messages(renderer, with_empty_system, _TOOLS)

    missing = renderer.render_text(missing_normalized, add_generation_prompt=True)
    empty = renderer.render_text(empty_normalized, add_generation_prompt=True)
    assert "You are Qwen, created by Alibaba Cloud." in missing
    assert "You are Qwen, created by Alibaba Cloud." not in empty
    assert missing != empty


def test_supervised_tokens_masks_and_generation_prefix_match_public_hf_main(
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> None:
    raw_messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "FIRST_ANSWER"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "SECOND_ANSWER"},
    ]
    renderer, messages = _assert_hf_parity(
        tokenizer,
        raw_messages,
        add_generation_prompt=False,
    )
    model_input, last_weights_tensor = renderer.build_supervised_example(
        messages,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    tokens = list(model_input.to_ints())
    last_weights = last_weights_tensor.tolist()
    generation_prefix = list(renderer.build_generation_prompt(messages[:-1]).to_ints())

    assert tokens[: len(generation_prefix)] == generation_prefix
    assert last_weights[: len(generation_prefix)] == [0.0] * len(generation_prefix)
    assert last_weights[len(generation_prefix) :] == [1.0] * (
        len(tokens) - len(generation_prefix)
    )
    last_text = _trained_text(tokenizer, tokens, last_weights)
    assert "SECOND_ANSWER" in last_text
    assert "FIRST_ANSWER" not in last_text

    all_input, all_weights_tensor = renderer.build_supervised_example(
        messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    assert list(all_input.to_ints()) == tokens
    all_text = _trained_text(tokenizer, tokens, all_weights_tensor.tolist())
    assert "FIRST_ANSWER" in all_text
    assert "SECOND_ANSWER" in all_text
    assert "first question" not in all_text
    assert "second question" not in all_text

    customized = normalize_messages(
        [
            {**message, "weight": int(index == 1)}
            for index, message in enumerate(raw_messages)
        ]
    )
    custom_input, custom_weights = renderer.build_supervised_example(
        customized,
        train_on_what=TrainOnWhat.CUSTOMIZED,
    )
    assert list(custom_input.to_ints()) == tokens
    custom_text = _trained_text(tokenizer, tokens, custom_weights.tolist())
    assert "FIRST_ANSWER" in custom_text
    assert "SECOND_ANSWER" not in custom_text


def test_dpo_chosen_rejected_match_public_hf_main_and_preserve_cross_boundary_bpe(
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> None:
    prompt_messages = [{"role": "user", "content": "Pick one."}]
    chosen_messages = [
        *prompt_messages,
        {"role": "assistant", "content": "\nA"},
    ]
    rejected_messages = [
        *prompt_messages,
        {"role": "assistant", "content": "\nB"},
    ]
    renderer = Qwen2_5Renderer(tokenizer)

    pair = render_preference_pair(
        {"messages": chosen_messages},
        {"messages": rejected_messages},
        renderer=renderer,
        tokenizer=tokenizer,
    )

    assert pair is not None
    expected_chosen = _hf_render(
        tokenizer,
        chosen_messages,
        add_generation_prompt=False,
        tokenize=True,
    )
    expected_rejected = _hf_render(
        tokenizer,
        rejected_messages,
        add_generation_prompt=False,
        tokenize=True,
    )
    assert isinstance(expected_chosen, list)
    assert isinstance(expected_rejected, list)
    assert pair.chosen_tokens == expected_chosen
    assert pair.rejected_tokens == expected_rejected

    expected_response_start = next(
        (
            index
            for index, (chosen_token, rejected_token) in enumerate(
                zip(expected_chosen, expected_rejected, strict=False)
            )
            if chosen_token != rejected_token
        ),
        min(len(expected_chosen), len(expected_rejected)),
    )
    assert pair.response_start == expected_response_start == 32
    assert (
        pair.chosen_tokens[: pair.response_start]
        == pair.rejected_tokens[: pair.response_start]
    )
    assert (
        pair.chosen_tokens[pair.response_start]
        != pair.rejected_tokens[pair.response_start]
    )

    generation_text = _hf_render(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    generation_tokens = _hf_render(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    chosen_text = _hf_render(
        tokenizer,
        chosen_messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    rejected_text = _hf_render(
        tokenizer,
        rejected_messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    assert isinstance(generation_text, str)
    assert isinstance(generation_tokens, list)
    assert isinstance(chosen_text, str)
    assert isinstance(rejected_text, str)
    assert chosen_text.startswith(generation_text)
    assert rejected_text.startswith(generation_text)

    # The full-conversation BPE merges the generation prompt's final newline
    # with the response's leading newline. The character prompt is a prefix,
    # but its standalone tokenization is deliberately not a token prefix.
    assert generation_tokens != pair.chosen_tokens[: len(generation_tokens)]
    for full_text in (chosen_text, rejected_text):
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        crossing_tokens = [
            index
            for index, (start, end) in enumerate(encoded["offset_mapping"])
            if start < len(generation_text) < end
        ]
        assert crossing_tokens == [pair.response_start - 1]

    # A causal logprob at response_start - 1 predicts the first differing
    # token at response_start; neither branch can silently skip that token.
    chosen_targets = pair.chosen_datum.loss_fn_inputs["target_tokens"].data
    rejected_targets = pair.rejected_datum.loss_fn_inputs["target_tokens"].data
    assert chosen_targets == pair.chosen_tokens[1:]
    assert rejected_targets == pair.rejected_tokens[1:]
    assert (
        chosen_targets[pair.response_start - 1]
        == pair.chosen_tokens[pair.response_start]
    )
    assert (
        rejected_targets[pair.response_start - 1]
        == pair.rejected_tokens[pair.response_start]
    )


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_v1_rejects_multipart_content(
    tokenizer: transformers.PreTrainedTokenizerBase,
    role: str,
) -> None:
    messages = normalize_messages(
        [
            {"role": "user", "content": "question"},
            {
                "role": role,
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                ],
            },
        ]
    )
    with pytest.raises(TypeError, match="content must be text"):
        Qwen2_5Renderer(tokenizer).build_supervised_example(messages)


def test_parse_tool_call_roundtrip(
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> None:
    renderer = Qwen2_5Renderer(tokenizer)
    arguments = {"city": "東京", "days": 2}
    response = tokenizer.encode(
        "Checking.\n<tool_call>\n"
        + json.dumps(
            {"name": "lookup_weather", "arguments": arguments},
            ensure_ascii=False,
        )
        + "\n</tool_call>"
        + PRODUCTION_EOS_TOKEN,
        add_special_tokens=False,
    )

    parsed, termination = renderer.parse_response(list(response))

    assert termination == ParseTermination.STOP_SEQUENCE
    assert parsed["content"] == "Checking."
    assert parsed["tool_calls"][0].function.name == "lookup_weather"
    assert json.loads(parsed["tool_calls"][0].function.arguments) == arguments

    function = _TOOLS[0]["function"]
    parsed_history = renderer.create_conversation_prefix_with_tools([function]) + [
        {"role": "user", "content": "Check Tokyo"},
        parsed,
        {"role": "tool", "content": "sunny"},
    ]
    raw_history = _renderer_messages(
        renderer,
        [
            {"role": "user", "content": "Check Tokyo"},
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "arguments": arguments,
                        },
                    }
                ],
            },
            {"role": "tool", "content": "sunny"},
        ],
        _TOOLS,
    )
    assert list(renderer.build_generation_prompt(parsed_history).to_ints()) == list(
        renderer.build_generation_prompt(raw_history).to_ints()
    )


def test_parse_preserves_unmatched_literal_tool_marker(
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> None:
    renderer = Qwen2_5Renderer(tokenizer)
    content = "Explain <tool_call> literally; this is not a complete call."
    response = tokenizer.encode(
        content + PRODUCTION_EOS_TOKEN,
        add_special_tokens=False,
    )

    parsed, termination = renderer.parse_response(list(response))

    assert termination == ParseTermination.STOP_SEQUENCE
    assert parsed["content"] == content
    assert not parsed.get("tool_calls")
