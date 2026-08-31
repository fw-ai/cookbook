"""Verify the GLM-5.3 renderer against its pinned Hugging Face template."""

from __future__ import annotations

from typing import Any

import pytest
import transformers

import training.renderer.glm5  # noqa: F401 - registers glm53
from training.renderer import get_renderer
from training.utils.supervised import (
    build_tool_prefixed_messages,
    normalize_messages,
)


_TOKENIZER = "zai-org/GLM-5.3"
_TOKENIZER_REVISION = "935644c05e76fc198714f4cca449fd8b970ff6d7"
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "time",
            "description": "Get time",
            "parameters": {
                "type": "object",
                "properties": {"zone": {"type": "string"}},
                "required": ["zone"],
            },
        },
    },
]


def _load_tokenizer() -> transformers.PreTrainedTokenizerBase | None:
    try:
        return transformers.AutoTokenizer.from_pretrained(
            _TOKENIZER,
            revision=_TOKENIZER_REVISION,
            trust_remote_code=True,
        )
    except Exception:  # noqa: BLE001 - network/auth/cache availability
        return None


@pytest.fixture(scope="module")
def tokenizer():
    tok = _load_tokenizer()
    if tok is None:
        pytest.skip(
            f"GLM-5.3 tokenizer not available: "
            f"{_TOKENIZER!r}@{_TOKENIZER_REVISION}"
        )
    if not getattr(tok, "chat_template", None):
        pytest.skip("Loaded GLM-5.3 tokenizer has no chat template.")
    return tok


@pytest.fixture(scope="module")
def renderer(tokenizer):
    return get_renderer("glm53", tokenizer)


def _hf_tokens(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        **kwargs,
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


def _generation_tokens(
    renderer,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> list[int]:
    normalized = build_tool_prefixed_messages(
        messages,
        renderer=renderer,
        tools=tools,
    )
    return list(renderer.build_generation_prompt(normalized).to_ints())


def _supervised_tokens(renderer, messages: list[dict[str, Any]]) -> list[int]:
    model_input, _ = renderer.build_supervised_example(normalize_messages(messages))
    return list(model_input.to_ints())


def _without_terminal_role_stop(tokenizer, tokens: list[int]) -> list[int]:
    user = tokenizer.encode("<|user|>", add_special_tokens=False)
    assert len(user) == 1
    assert tokens[-1] == user[0]
    return tokens[:-1]


def _parallel_tool_messages(
    results: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "weather and time"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": {"city": "Paris"},
                    },
                },
                {
                    "id": "call_time",
                    "type": "function",
                    "function": {
                        "name": "time",
                        "arguments": {"zone": "UTC"},
                    },
                },
            ],
        },
        *[
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
            for tool_call_id, content in results
        ],
    ]


def test_registered_glm53_renderer(tokenizer, renderer):
    assert type(renderer).__name__ == "GLM53Renderer"
    assert renderer.has_extension_property is True
    assert renderer.supports_per_message_rendering is False


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "hello"}],
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "<think>old</think>a1"},
            {"role": "user", "content": "q2"},
        ],
    ],
)
def test_generation_prompt_matches_hf(tokenizer, renderer, messages):
    assert _generation_tokens(renderer, messages) == _hf_tokens(
        tokenizer,
        messages,
        add_generation_prompt=True,
    )


def test_enable_thinking_false_is_ignored_like_hf(tokenizer, renderer):
    messages = [{"role": "user", "content": "hello"}]
    assert _generation_tokens(renderer, messages) == _hf_tokens(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def test_default_preserves_historical_reasoning(tokenizer, renderer):
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "<think>old</think>a1"},
        {"role": "user", "content": "q2"},
    ]
    decoded = tokenizer.decode(_generation_tokens(renderer, messages))
    assert "<think>old</think>a1" in decoded


def test_supervised_example_matches_hf_modulo_training_stop(tokenizer, renderer):
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "reasoning_content": "work", "content": "a1"},
    ]
    ours = _without_terminal_role_stop(
        tokenizer,
        _supervised_tokens(renderer, messages),
    )
    assert ours == _hf_tokens(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )


def test_parallel_tool_results_follow_call_order(tokenizer, renderer):
    messages = _parallel_tool_messages(
        [("call_time", "12:00"), ("call_weather", "sunny")]
    )
    ours = _generation_tokens(renderer, messages, tools=_TOOLS)
    assert ours == _hf_tokens(
        tokenizer,
        messages,
        add_generation_prompt=True,
        tools=_TOOLS,
    )
    rendered = tokenizer.decode(ours)
    assert rendered.index("sunny") < rendered.index("12:00")


def test_duplicate_tool_result_ids_fall_back_to_source_order(tokenizer, renderer):
    messages = _parallel_tool_messages(
        [("call_time", "first"), ("call_time", "second")]
    )
    ours = _generation_tokens(renderer, messages, tools=_TOOLS)
    assert ours == _hf_tokens(
        tokenizer,
        messages,
        add_generation_prompt=True,
        tools=_TOOLS,
    )
    rendered = tokenizer.decode(ours)
    assert rendered.index("first") < rendered.index("second")
