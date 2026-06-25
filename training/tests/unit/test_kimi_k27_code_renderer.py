"""Verify Kimi K2.7 Code rendering matches the official HF chat template."""

from __future__ import annotations

from typing import Any

import pytest

import training.renderer  # noqa: F401  (registers "kimi_k27_code")
from tinker_cookbook.renderers import get_renderer

from training.utils.supervised import normalize_messages
from training.utils.tokenizers import load_tokenizer


_TOKENIZER_MODEL = "moonshotai/Kimi-K2.7-Code"
_RENDERER_NAME = "kimi_k27_code"


@pytest.fixture(scope="module")
def tokenizer():
    try:
        tok = load_tokenizer(_TOKENIZER_MODEL)
    except (OSError, ValueError, RuntimeError) as exc:
        pytest.skip(f"tokenizer unavailable for {_TOKENIZER_MODEL!r}: {exc}")
    if not getattr(tok, "chat_template", None):
        pytest.skip(f"{_TOKENIZER_MODEL!r} has no chat_template")
    return tok


@pytest.fixture(scope="module")
def renderer(tokenizer):
    return get_renderer(_RENDERER_NAME, tokenizer)


def _hf_tokens(
    tokenizer: Any,
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
        return [int(t) for t in list(result.input_ids)]
    return [int(t) for t in list(result)]


def _renderer_generation_tokens(renderer: Any, messages: list[dict[str, Any]]) -> list[int]:
    model_input = renderer.build_generation_prompt(
        normalize_messages(messages),
        role="assistant",
    )
    return [int(t) for t in model_input.to_ints()]


def _renderer_supervised_tokens(renderer: Any, messages: list[dict[str, Any]]) -> list[int]:
    model_input, _weights = renderer.build_supervised_example(normalize_messages(messages))
    return [int(t) for t in model_input.to_ints()]


def _assert_tokens_match(tokenizer: Any, expected: list[int], actual: list[int]) -> None:
    if expected == actual:
        return

    n = min(len(expected), len(actual))
    first = next((i for i in range(n) if expected[i] != actual[i]), n)
    lo = max(0, first - 6)
    hi = first + 7
    raise AssertionError(
        "Token mismatch against HF chat template:\n"
        f"expected_tokens={len(expected)} actual_tokens={len(actual)} "
        f"first_divergence={first}\n"
        f"--- expected[{lo}:{hi}] ---\n"
        f"{tokenizer.decode(expected[lo:hi], skip_special_tokens=False)}\n"
        f"--- actual[{lo}:{hi}] ---\n"
        f"{tokenizer.decode(actual[lo:hi], skip_special_tokens=False)}"
    )


_DPO_STYLE_MESSAGES = [
    {
        "role": "user",
        "content": "Solve 2+2. Show concise reasoning in <think>...</think>.",
    },
    {
        "role": "assistant",
        "content": "<think>2+2 is basic arithmetic.</think><answer>4</answer>",
    },
]

_REASONING_CONTENT_MESSAGES = [
    {"role": "system", "content": "Answer briefly."},
    {"role": "user", "content": "What is 2+2?"},
    {
        "role": "assistant",
        "reasoning_content": "2+2 is basic arithmetic.",
        "content": "4.",
    },
    {"role": "user", "content": "And 3+3?"},
]


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "messages",
    [
        pytest.param(
            [{"role": "user", "content": "Hello"}],
            id="user_only_no_default_system",
        ),
        pytest.param(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            id="explicit_system",
        ),
        pytest.param(_DPO_STYLE_MESSAGES, id="dpo_style_content_think"),
        pytest.param(_REASONING_CONTENT_MESSAGES, id="reasoning_content_history"),
    ],
)
def test_generation_prompt_matches_hf_chat_template(tokenizer, renderer, messages):
    expected = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    actual = _renderer_generation_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)


@pytest.mark.timeout(180)
def test_generation_prompt_with_top_level_tools_matches_hf(tokenizer, renderer):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name."},
                    },
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Weather in SF?"},
    ]
    prefix = renderer.create_conversation_prefix_with_tools(
        [tools[0]["function"]],
        system_prompt="You are helpful.",
    )

    expected = _hf_tokens(
        tokenizer,
        messages,
        add_generation_prompt=True,
        tools=tools,
    )
    actual = _renderer_generation_tokens(renderer, prefix + messages[1:])
    _assert_tokens_match(tokenizer, expected, actual)


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "messages",
    [
        pytest.param(_DPO_STYLE_MESSAGES, id="dpo_style_content_think"),
        pytest.param(
            [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "reasoning_content": "2+2 is basic arithmetic.",
                    "content": "4.",
                },
            ],
            id="reasoning_content_no_system",
        ),
    ],
)
def test_supervised_example_matches_hf_chat_template(tokenizer, renderer, messages):
    expected = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    actual = _renderer_supervised_tokens(renderer, messages)
    _assert_tokens_match(tokenizer, expected, actual)
