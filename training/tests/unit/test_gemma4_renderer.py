"""Verify ``Gemma4Renderer`` matches the official Gemma 4 chat template.

Test tiers (see also ``test_supervised_rendering.py`` for resolver coverage):

**Tier 1 — tokenizer only** (skipped unless ``GEMMA4_MODEL_PATH`` is set):

* Token-level parity vs ``tokenizer.apply_chat_template`` for conversation,
  tool, thinking, and reasoning shapes.
* ``test_reasoning_parity_with_hf`` — SFT byte parity for ``reasoning`` /
  ``reasoning_content`` labels on the default ``gemma4`` renderer (HF jinja).
* ``test_thinking_renderer_*`` — ``gemma4_thinking`` plain-reasoning SFT
  (intentionally deviates from HF for Qwen-style datasets).

**Tier 2 — GPU model** (skipped without weights + CUDA):

* Greedy-decode parity and tool-call e2e loops.

Quick commands::

    # Resolver override (CI-safe, no checkpoint)
    cd public-repos/cookbook
    python3 -m pytest training/tests/unit/test_supervised_rendering.py::test_resolve_renderer_name_supports_gemma4_thinking_override -v

    # Full file (Tier 1–2; skips when GEMMA4_MODEL_PATH / CUDA unavailable)
    GEMMA4_MODEL_PATH=/path/to/gemma-4-E2B-it \\
        python3 -m pytest training/tests/unit/test_gemma4_renderer.py -v
"""

from __future__ import annotations

import os

import pytest
import transformers

import training.renderer  # noqa: F401 — installs Gemma4SplitRenderer override
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import TrainOnWhat, RenderContext

from training.renderer.gemma4 import (
    Gemma4Renderer,
    _get_reasoning_text,
    _should_emit_reasoning_channel,
    _split_thinking_and_text,
)
from training.renderer._gemma4_split import Gemma4SplitRenderer, Gemma4ThinkingSplitRenderer
from training.utils.supervised import render_messages_to_datum, render_messages_to_datums

_MODEL_PATH_ENV = "GEMMA4_MODEL_PATH"


def _resolve_model_path() -> str:
    path = os.environ.get(_MODEL_PATH_ENV)
    if not path:
        pytest.skip(
            f"Set {_MODEL_PATH_ENV} to a directory containing an official "
            "google/gemma-4-*-it checkpoint to run these tests."
        )
    return path


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(
        _resolve_model_path(), trust_remote_code=True,
    )


@pytest.fixture(scope="module")
def renderer(tokenizer):
    return Gemma4Renderer(tokenizer)


@pytest.fixture(scope="module")
def thinking_renderer(tokenizer):
    return Gemma4Renderer(tokenizer, enable_thinking=True)


def _hf_tokens(
    tokenizer,
    messages,
    *,
    add_generation_prompt: bool = True,
    tools=None,
    enable_thinking: bool = False,
) -> list[int]:
    kwargs: dict = {"add_generation_prompt": add_generation_prompt}
    if tools is not None:
        kwargs["tools"] = tools
    if enable_thinking:
        kwargs["enable_thinking"] = True
    result = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    if hasattr(result, "input_ids"):
        return list(result.input_ids)
    return list(result)


def _renderer_tokens(renderer, messages) -> list[int]:
    return list(renderer.build_generation_prompt(messages, role="assistant").to_ints())


def _assert_match(tokenizer, hf, ours):
    if hf != ours:
        raise AssertionError(
            "Token mismatch.\n"
            f"HF   ({len(hf)} toks): {hf}\n"
            f"Ours ({len(ours)} toks): {ours}\n"
            f"--- HF text ---\n{tokenizer.decode(hf)}\n"
            f"--- Our text ---\n{tokenizer.decode(ours)}"
        )


# ── Token-level parity vs the official HF chat template ─────────────────────


@pytest.mark.parametrize(
    "messages",
    [
        # Single user turn (generation prompt as the very first turn).
        [{"role": "user", "content": "Hello"}],
        # System + multi-turn dialogue.
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ],
        # User content with leading & trailing whitespace — verifies the
        # template's `| trim` is reproduced by the renderer.
        [{"role": "user", "content": "  spaced  "}],
        # User content with internal whitespace runs that survive trimming.
        [{"role": "user", "content": "line one\nline two\nline three"}],
        # Content that looks like the chat template tokens (must be plain
        # text, not interpreted as control tokens).
        [
            {"role": "user", "content": "<|turn>not a turn<turn|>"},
            {"role": "assistant", "content": "ack"},
        ],
        # Multi-line code block.
        [
            {"role": "user", "content": "```python\nprint('hi')\n```"},
            {"role": "assistant", "content": "```\nhi\n```"},
        ],
        # Mixed unicode and emoji.
        [
            {"role": "system", "content": "Réponds en français."},
            {"role": "user", "content": "Bonjour 👋"},
            {"role": "assistant", "content": "Salut! Ça va? 😊"},
        ],
        # Tool role rendered verbatim (NOT rewritten to user).
        [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "checking"},
            {"role": "tool", "content": "sunny, 72F"},
            {"role": "assistant", "content": "It's sunny."},
        ],
        # Assistant content with a thinking block — must be stripped from
        # history to match the template's strip_thinking macro.
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<|channel>thought\nI think...<channel|>The answer is 4."},
            {"role": "user", "content": "next"},
        ],
    ],
    ids=[
        "user_only",
        "system_multi_turn",
        "trim_whitespace",
        "multiline",
        "literal_template_syntax",
        "code_block",
        "unicode",
        "tool_role_verbatim",
        "model_thinking_stripped",
    ],
)
def test_parity_with_hf_chat_template(tokenizer, renderer, messages):
    """Renderer tokens must match HF apply_chat_template byte-for-byte."""
    _assert_match(tokenizer, _hf_tokens(tokenizer, messages), _renderer_tokens(renderer, messages))


# ── Public API contract ──────────────────────────────────────────────────────


def test_stop_sequences_include_turn_close_and_tool_response(renderer, tokenizer):
    """Both `<turn|>` (normal close) and `<|tool_response>` (tool-call
    handoff) are valid completion signals — Gemma 4 emits the latter
    immediately after `<tool_call|>` when it wants the runner to inject
    a tool result, and both vLLM and SGLang treat it as the end of the
    assistant turn for tool-calling outputs."""
    turn_close = tokenizer.encode("<turn|>", add_special_tokens=False)[0]
    tool_resp_open = tokenizer.encode("<|tool_response>", add_special_tokens=False)[0]
    assert renderer.get_stop_sequences() == [turn_close, tool_resp_open]


def test_parse_response_plain_text_roundtrip(tokenizer, renderer):
    """A rendered model turn parses back to the original text."""
    sample = "Sure, here is the answer."
    encoded = tokenizer.encode(sample + "<turn|>", add_special_tokens=False)
    msg, ok = renderer.parse_response(encoded)
    assert (ok, msg["role"], msg["content"]) == (True, "assistant", sample)


# ── normalize_messages → renderer integration (CI-safe) ───────────────────────


def test_get_reasoning_text_reads_normalized_reasoning_content():
    """After ``normalize_messages``, ``reasoning_content`` lives in ThinkingPart only."""
    normalized = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "need to call get_weather"},
        ],
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "get_weather", "arguments": {}},
            }
        ],
    }
    assert _get_reasoning_text(normalized) == "need to call get_weather"


def test_get_reasoning_text_reads_normalized_reasoning_field():
    """``reasoning`` (jinja field) is also promoted into ThinkingPart before render."""
    normalized = {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "from reasoning field"}],
        "tool_calls": [
            {"type": "function", "function": {"name": "fn", "arguments": {}}},
        ],
    }
    assert _get_reasoning_text(normalized) == "from reasoning field"


def test_get_reasoning_text_empty_reasoning_falls_back_to_reasoning_content():
    message = {
        "role": "assistant",
        "reasoning": "",
        "reasoning_content": "fallback thought",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "fn", "arguments": {}}}],
    }
    assert _get_reasoning_text(message) == "fallback thought"


def test_split_thinking_and_text_handles_open_thought_fragment():
    parts = _split_thinking_and_text(
        "<|channel>thought\nstep by step\n <|tool_call>call:fn{}<tool_call|>"
    )
    assert parts[0]["type"] == "thinking"
    assert parts[0]["thinking"] == "step by step"
    assert parts[1]["text"] == "<|tool_call>call:fn{}<tool_call|>"


# ── reasoning / reasoning_content SFT (HF parity) ─────────────────────────────


def test_should_emit_reasoning_channel_matrix() -> None:
    """Gate matrix for plain vs tool-call reasoning emission."""
    post_user_ctx = RenderContext(
        idx=1, is_last=True, prev_message=None, last_user_index=0
    )
    pre_user_ctx = RenderContext(
        idx=1, is_last=False, prev_message=None, last_user_index=2
    )
    plain_assistant = {"role": "assistant", "content": "answer"}
    tool_assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": "fn", "arguments": {}}}
        ],
    }

    assert not _should_emit_reasoning_channel(
        plain_assistant, pre_user_ctx, enable_thinking=False
    )
    assert not _should_emit_reasoning_channel(
        plain_assistant, post_user_ctx, enable_thinking=False
    )
    assert _should_emit_reasoning_channel(
        tool_assistant, post_user_ctx, enable_thinking=False
    )
    assert _should_emit_reasoning_channel(
        plain_assistant, post_user_ctx, enable_thinking=True
    )
    assert not _should_emit_reasoning_channel(
        plain_assistant, pre_user_ctx, enable_thinking=True
    )


_REASONING_SFT_MESSAGES = [
    {"role": "user", "content": "What is 17 * 23?"},
    {
        "role": "assistant",
        "content": "391",
        "reasoning_content": "multiply 17 and 23",
    },
]

_TOOL_CALL_WITH_REASONING = [
    {"role": "user", "content": "weather in paris?"},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "need to call get_weather",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "paris", "unit": "c"},
                },
            }
        ],
    },
]

_REASONING_FIELD_PRECEDENCE = [
    {"role": "user", "content": "go"},
    {
        "role": "assistant",
        "content": "",
        "reasoning": "from reasoning field",
        "reasoning_content": "from reasoning_content field",
        "tool_calls": [
            {"type": "function", "function": {"name": "fn", "arguments": {}}}
        ],
    },
]

_TWO_ASSISTANTS_AFTER_USER = [
    {"role": "user", "content": "help"},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "first thought",
        "tool_calls": [
            {"type": "function", "function": {"name": "step1", "arguments": {"x": 1}}}
        ],
    },
    {
        "role": "assistant",
        "content": "continuing",
        "reasoning_content": "second thought",
        "tool_calls": [
            {"type": "function", "function": {"name": "step2", "arguments": {"y": 2}}}
        ],
    },
]

_REASONING_BEFORE_LAST_USER = [
    {"role": "user", "content": "first"},
    {
        "role": "assistant",
        "content": "draft",
        "reasoning_content": "should not render",
        "tool_calls": [
            {"type": "function", "function": {"name": "draft", "arguments": {}}}
        ],
    },
    {"role": "user", "content": "second"},
    {"role": "assistant", "content": "final"},
]


@pytest.mark.parametrize(
    "messages",
    [
        _REASONING_SFT_MESSAGES,
        _TOOL_CALL_WITH_REASONING,
        _REASONING_FIELD_PRECEDENCE,
        _TWO_ASSISTANTS_AFTER_USER,
        _REASONING_BEFORE_LAST_USER,
    ],
    ids=[
        "reasoning_without_tools_ignored",
        "tool_call_with_reasoning_content",
        "reasoning_field_precedence",
        "two_assistants_after_user",
        "reasoning_before_last_user",
    ],
)
def test_reasoning_parity_with_hf(tokenizer, renderer, messages):
    """Reasoning labels must match HF ``apply_chat_template`` byte-for-byte."""
    hf_ids = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    model_input, _weights = renderer.build_supervised_example(messages)
    _assert_match(tokenizer, hf_ids, list(model_input.to_ints()))


def test_supervised_example_reasoning_content_thinking_renderer(
    tokenizer, thinking_renderer
):
    """``gemma4_thinking`` supervises plain ``reasoning_content`` (HF does not)."""
    model_input, _weights = thinking_renderer.build_supervised_example(
        _REASONING_SFT_MESSAGES
    )
    text = tokenizer.decode(list(model_input.to_ints()))
    assert "<|think|>" in text
    assert "<|channel>thought\nmultiply 17 and 23\n " in text
    assert "391" in text


def test_gemma4_vs_gemma4_thinking_differs_by_system_and_plain_reasoning(
    tokenizer, renderer, thinking_renderer
):
    """``gemma4`` drops plain reasoning; ``gemma4_thinking`` supervises it."""
    default_ids = list(
        renderer.build_supervised_example(_REASONING_SFT_MESSAGES)[0].to_ints()
    )
    thinking_ids = list(
        thinking_renderer.build_supervised_example(_REASONING_SFT_MESSAGES)[0].to_ints()
    )
    default_text = tokenizer.decode(default_ids)
    thinking_text = tokenizer.decode(thinking_ids)
    assert "<|think|>" not in default_text
    assert "<|think|>" in thinking_text
    assert "<|channel>thought\nmultiply 17 and 23\n " not in default_text
    assert "<|channel>thought\nmultiply 17 and 23\n " in thinking_text


_PLAIN_REASONING_BEFORE_LAST_USER = [
    {"role": "user", "content": "first"},
    {
        "role": "assistant",
        "content": "draft",
        "reasoning_content": "should not render",
    },
    {"role": "user", "content": "second"},
    {"role": "assistant", "content": "final"},
]

_TWO_PLAIN_ASSISTANTS_AFTER_USER = [
    {"role": "user", "content": "help"},
    {
        "role": "assistant",
        "content": "step one",
        "reasoning_content": "first thought",
    },
    {
        "role": "assistant",
        "content": "step two",
        "reasoning_content": "second thought",
    },
]


def test_thinking_renderer_plain_reasoning_before_last_user_ignored(
    tokenizer, thinking_renderer
):
    """Historical plain reasoning is still gated before the final user."""
    model_input, _weights = thinking_renderer.build_supervised_example(
        _PLAIN_REASONING_BEFORE_LAST_USER
    )
    text = tokenizer.decode(list(model_input.to_ints()))
    assert "should not render" not in text
    assert "final" in text


def test_thinking_renderer_two_plain_assistants_after_user(
    tokenizer, thinking_renderer
):
    """Consecutive post-user assistants each supervise their reasoning label."""
    model_input, _weights = thinking_renderer.build_supervised_example(
        _TWO_PLAIN_ASSISTANTS_AFTER_USER
    )
    text = tokenizer.decode(list(model_input.to_ints()))
    assert "<|channel>thought\nfirst thought\n " in text
    assert "<|channel>thought\nsecond thought\n " in text
    assert "step one" in text
    assert "step two" in text


def test_thinking_renderer_plain_reasoning_has_trainable_weights(
    tokenizer, thinking_renderer
):
    """Plain reasoning tokens are included in the supervised loss mask."""
    datum = render_messages_to_datum(
        _REASONING_SFT_MESSAGES,
        renderer=thinking_renderer,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    trained_ids = [t for t, w in zip(datum.token_ids, datum.token_weights) if w > 0]
    trained_text = tokenizer.decode(trained_ids)
    assert "multiply 17 and 23" in trained_text
    assert "391" in trained_text


_MULTI_TURN_PLAIN_REASONING = [
    {"role": "user", "content": "step one"},
    {
        "role": "assistant",
        "content": "answer one",
        "reasoning_content": "REASON_A1",
    },
    {"role": "user", "content": "step two"},
    {
        "role": "assistant",
        "content": "answer two",
        "reasoning_content": "REASON_A2",
    },
]


def test_thinking_disaggregate_preserves_plain_reasoning_on_earlier_assistant(
    tokenizer,
):
    """``gemma4_thinking`` disaggregation supervises each turn's reasoning."""
    split_renderer = get_renderer("gemma4_thinking", tokenizer)
    assert type(split_renderer).__name__ == "Gemma4ThinkingSplitRenderer"

    datums = render_messages_to_datums(
        _MULTI_TURN_PLAIN_REASONING,
        renderer=split_renderer,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    assert len(datums) == 2

    def _trained_text(datum) -> str:
        ids = [t for t, w in zip(datum.token_ids, datum.token_weights) if w > 0]
        return tokenizer.decode(ids)

    assert "REASON_A1" in _trained_text(datums[0])
    assert "REASON_A2" not in _trained_text(datums[0])
    assert "REASON_A2" in _trained_text(datums[1])
    assert "REASON_A1" not in _trained_text(datums[1])


def test_parse_response_extracts_thinking(tokenizer, renderer):
    """A model turn containing a closed ``<|channel>thought<channel|>`` block parses
    out into separate ThinkingPart + TextPart content."""
    body = "<|channel>I'm thinking<channel|>The answer is 4.<turn|>"
    encoded = tokenizer.encode(body, add_special_tokens=False)
    msg, ok = renderer.parse_response(encoded)
    assert ok
    assert isinstance(msg["content"], list)
    types = [p["type"] for p in msg["content"]]
    assert types == ["thinking", "text"]
    assert msg["content"][0]["thinking"] == "I'm thinking"
    assert msg["content"][1]["text"] == "The answer is 4."


def test_parse_response_extracts_open_thought_channel(tokenizer, renderer):
    """Open jinja thought fragments (no ``<channel|>``) parse into ThinkingPart."""
    body = "<|channel>thought\nstep by step\n The answer is 4.<turn|>"
    encoded = tokenizer.encode(body, add_special_tokens=False)
    msg, ok = renderer.parse_response(encoded)
    assert ok
    assert isinstance(msg["content"], list)
    assert msg["content"][0]["type"] == "thinking"
    assert msg["content"][0]["thinking"] == "step by step"
    assert msg["content"][1]["text"] == "The answer is 4."


# ── build_supervised_example parity ─────────────────────────────────────────


def test_supervised_example_matches_hf_no_generation_prompt(tokenizer, renderer):
    """``build_supervised_example`` must produce a token sequence that
    matches ``apply_chat_template(..., add_generation_prompt=False)``.

    SFT training uses build_supervised_example, not build_generation_prompt,
    so this is the parity test that actually pins what the model is
    trained on.
    """
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well."},
    ]
    hf_ids = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    model_input, _weights = renderer.build_supervised_example(messages)
    our_ids = list(model_input.to_ints())
    _assert_match(tokenizer, hf_ids, our_ids)


_MULTI_TURN_TOOL_REASONING = [
    {"role": "user", "content": "first"},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "REASON_A1",
        "tool_calls": [
            {"type": "function", "function": {"name": "step1", "arguments": {"x": 1}}}
        ],
    },
    {"role": "user", "content": "second"},
    {
        "role": "assistant",
        "content": "done",
        "reasoning_content": "REASON_A2",
        "tool_calls": [
            {"type": "function", "function": {"name": "step2", "arguments": {"y": 2}}}
        ],
    },
]


def test_disaggregate_preserves_tool_reasoning_on_earlier_assistant(tokenizer):
    """Multi-turn ``ALL_ASSISTANT_MESSAGES`` must not drop a1's thought channel.

    Full-transcript rendering strips reasoning before the final user (HF
    default). Per-user-turn disaggregation trains each prefix independently.
    """
    split_renderer = get_renderer("gemma4", tokenizer)
    assert split_renderer.has_extension_property is False
    assert type(split_renderer).__name__ == "Gemma4SplitRenderer"

    datums = render_messages_to_datums(
        _MULTI_TURN_TOOL_REASONING,
        renderer=split_renderer,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    assert len(datums) == 2

    def _trained_text(datum) -> str:
        ids = [t for t, w in zip(datum.token_ids, datum.token_weights) if w > 0]
        return tokenizer.decode(ids)

    assert "REASON_A1" in _trained_text(datums[0])
    assert "REASON_A2" not in _trained_text(datums[0])
    assert "REASON_A2" in _trained_text(datums[1])
    assert "REASON_A1" not in _trained_text(datums[1])

    # Full-transcript render strips reasoning before the final user (HF default).
    base_renderer = Gemma4Renderer(tokenizer)
    assert base_renderer.has_extension_property is False
    full_datum = render_messages_to_datum(
        _MULTI_TURN_TOOL_REASONING,
        renderer=base_renderer,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    full_trained = _trained_text(full_datum)
    assert "step1" in full_trained
    assert "REASON_A1" not in full_trained


# ── Sequence-extension property ─────────────────────────────────────────────


def test_gemma4_split_renderer_disables_extension_property_for_sft():
    """Registered ``gemma4`` uses the split subclass for multi-turn SFT."""
    from unittest.mock import MagicMock

    tok = MagicMock()
    tok.encode = lambda text, add_special_tokens=False: [1]
    renderer = Gemma4SplitRenderer(tok)
    assert isinstance(renderer, Gemma4Renderer)
    assert renderer.has_extension_property is False


def test_plain_text_observations_remain_prefix_extensions(renderer):
    """Plain-text multi-turn prompts still satisfy observation prefix extension.

    ``has_extension_property`` is False on ``Gemma4Renderer`` because
    tool-call + reasoning transcripts do not (see next test).
    """
    assert renderer.has_extension_property is False

    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
        {"role": "user", "content": "third question"},
        {"role": "assistant", "content": "third answer"},
    ]
    boundaries = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    assert len(boundaries) >= 2

    prev: list[int] = []
    for k in boundaries:
        cur = list(
            renderer.build_generation_prompt(messages[:k], role="assistant").to_ints()
        )
        assert cur[: len(prev)] == prev, (
            f"observation at boundary k={k} is not a prefix extension of the previous "
            f"observation; len(prev)={len(prev)} len(cur)={len(cur)}"
        )
        prev = cur


def test_tool_reasoning_observations_are_not_prefix_extensions(renderer):
    """History-gated thought channels break prefix reuse across assistant turns."""
    u1 = {"role": "user", "content": "step one"}
    a1 = {
        "role": "assistant",
        "reasoning_content": "REASON_A1",
        "tool_calls": [
            {"type": "function", "function": {"name": "step1", "arguments": {"x": 1}}}
        ],
    }
    u2 = {"role": "user", "content": "step two"}

    after_first_assistant = list(
        renderer.build_generation_prompt([u1, a1], role="assistant").to_ints()
    )
    before_second_assistant = list(
        renderer.build_generation_prompt([u1, a1, u2], role="assistant").to_ints()
    )
    assert before_second_assistant[: len(after_first_assistant)] != after_first_assistant, (
        "re-rendering the first assistant after a later user is appended must change "
        "tokens (thought channel stripped), so observations are not prefix extensions"
    )


# ── Tool-call / tool-response parity ────────────────────────────────────────


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City"},
                "unit": {
                    "type": "string",
                    "description": "Unit",
                    "enum": ["c", "f"],
                },
            },
            "required": ["location"],
        },
    },
}

_NESTED_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "q"},
                "filters": {
                    "type": "object",
                    "description": "f",
                    "properties": {
                        "year": {"type": "string", "description": "y"},
                    },
                    "required": ["year"],
                },
                "tags": {
                    "type": "array",
                    "description": "t",
                    "items": {"type": "string"},
                },
            },
            "required": ["query"],
        },
    },
}

_NO_PROPS_TOOLS = [
    {"type": "function", "function": {"name": "a", "description": "d1",
     "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "b", "description": "d2",
     "parameters": {"type": "object", "properties": {}, "required": []}}},
]


def _renderer_tokens_with_tools(renderer, messages, tools, *, system_prompt: str = ""):
    """Build a renderer prompt that mirrors HF's ``tools=...`` argument by
    prepending the synthetic system message returned by
    ``create_conversation_prefix_with_tools``.

    If the conversation already starts with a system message we fold its
    content into the synthetic system message via ``system_prompt`` and drop
    the original — the official template only emits one system block.
    """
    if messages and messages[0]["role"] in ("system", "developer"):
        sys_text = messages[0]["content"]
        rest = messages[1:]
    else:
        sys_text = system_prompt
        rest = list(messages)
    prefix = renderer.create_conversation_prefix_with_tools(tools, system_prompt=sys_text)
    return list(renderer.build_generation_prompt(prefix + rest, role="assistant").to_ints())


@pytest.mark.parametrize(
    ("messages", "tools"),
    [
        (
            [{"role": "user", "content": "weather in paris?"}],
            [_WEATHER_TOOL],
        ),
        (
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "weather?"},
            ],
            [_WEATHER_TOOL],
        ),
        (
            [{"role": "user", "content": "go"}],
            [_NESTED_TOOL],
        ),
        (
            [{"role": "user", "content": "go"}],
            _NO_PROPS_TOOLS,
        ),
    ],
    ids=[
        "single_tool_user_only",
        "single_tool_with_system",
        "nested_tool_def",
        "two_tools_no_props",
    ],
)
def test_tool_definitions_parity(tokenizer, renderer, messages, tools):
    """Renderer must match HF's ``apply_chat_template(..., tools=...)`` token
    sequence for the supported tool-definition shapes."""
    hf = _hf_tokens(tokenizer, messages, tools=tools)
    ours = _renderer_tokens_with_tools(renderer, messages, tools)
    _assert_match(tokenizer, hf, ours)


@pytest.mark.parametrize(
    ("messages", "tools"),
    [
        # Assistant emits a tool call with two scalar args (string, string).
        (
            [
                {"role": "user", "content": "weather in paris?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"location": "paris", "unit": "c"},
                            },
                        }
                    ],
                },
            ],
            [_WEATHER_TOOL],
        ),
        # Assistant tool call with mixed types (bool, int, string) — exercises
        # the format_argument type branches.
        (
            [
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "fn",
                                "arguments": {"flag": True, "count": 3, "name": "z"},
                            },
                        }
                    ],
                },
            ],
            None,
        ),
        # Nested-object & array arguments — exercises recursive format_argument.
        (
            [
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "fn",
                                "arguments": {
                                    "obj": {"a": 1, "b": "two"},
                                    "arr": [1, 2, 3],
                                },
                            },
                        }
                    ],
                },
            ],
            None,
        ),
    ],
    ids=["string_args", "scalar_mixed_args", "nested_args"],
)
def test_tool_call_parity(tokenizer, renderer, messages, tools):
    """Assistant ``tool_calls`` must serialise to the same tokens as the HF
    template's ``<|tool_call>call:name{...}<tool_call|>`` block."""
    if tools is not None:
        hf = _hf_tokens(tokenizer, messages, tools=tools, add_generation_prompt=False)
        ours = _renderer_tokens_with_tools(
            renderer,
            messages + [],  # explicit copy for clarity
            tools,
        )
        # _renderer_tokens_with_tools always appends a generation suffix; for
        # this no-genprompt parity check use the manual no-suffix call.
        if messages[0]["role"] in ("system", "developer"):
            sys_text, rest = messages[0]["content"], messages[1:]
        else:
            sys_text, rest = "", list(messages)
        prefix = renderer.create_conversation_prefix_with_tools(tools, system_prompt=sys_text)
        ours = list(
            renderer.build_supervised_example(prefix + rest)[0].to_ints()
        )
    else:
        hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
        ours = list(renderer.build_supervised_example(messages)[0].to_ints())
    _assert_match(tokenizer, hf, ours)


@pytest.mark.parametrize(
    "messages",
    [
        # Tool response with mapping payload, then a generation prompt.
        # Per the template, the prompt suffix `<|turn>model\n` is suppressed
        # because the previous message was a pure tool response.
        [
            {"role": "user", "content": "weather in paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "paris"},
                        },
                    }
                ],
            },
            {
                "role": "user",
                "content": "",
                "tool_responses": [
                    {"name": "get_weather", "response": {"temp": 18, "unit": "c"}}
                ],
            },
        ],
        # Tool response with a scalar payload.
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "fn", "arguments": {}}}
                ],
            },
            {
                "role": "user",
                "content": "",
                "tool_responses": [{"name": "fn", "response": "hello"}],
            },
        ],
        # Tool response followed by another user turn — the next turn DOES get
        # a `<|turn>model\n` generation suffix because the *immediately
        # preceding* message is no longer a tool response.
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "fn", "arguments": {"x": 1}},
                    }
                ],
            },
            {
                "role": "user",
                "content": "",
                "tool_responses": [{"name": "fn", "response": {"v": 2}}],
            },
            {"role": "user", "content": "now what"},
        ],
    ],
    ids=["mapping_response", "scalar_response", "response_then_user"],
)
def test_tool_response_parity(tokenizer, renderer, messages):
    """``tool_responses`` must round-trip the template's
    ``<|tool_response>response:name{...}<tool_response|>`` block, including
    the trailing-``<turn|>`` suppression and the generation-suffix
    suppression that the template applies after a pure tool-response turn."""
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    ours = _renderer_tokens(renderer, messages)
    _assert_match(tokenizer, hf, ours)


# ── enable_thinking parity ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "hi"}],
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hi"},
        ],
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ans"},
            {"role": "user", "content": "second"},
        ],
    ],
    ids=["no_system", "with_system", "multi_turn"],
)
def test_enable_thinking_parity(tokenizer, thinking_renderer, messages):
    """When ``enable_thinking=True``, the renderer must inject the same
    ``<|think|>`` system marker the HF template emits."""
    hf = _hf_tokens(tokenizer, messages, enable_thinking=True)
    ours = list(
        thinking_renderer.build_generation_prompt(messages, role="assistant").to_ints()
    )
    _assert_match(tokenizer, hf, ours)


def test_enable_thinking_with_tools_parity(tokenizer, thinking_renderer):
    """``enable_thinking`` and ``tools`` should compose: the system block
    contains ``<|think|>`` followed by the tool declarations, with no
    intervening whitespace, exactly like the template."""
    messages = [{"role": "user", "content": "weather?"}]
    tools = [_WEATHER_TOOL]
    hf = _hf_tokens(tokenizer, messages, tools=tools, enable_thinking=True)

    prefix = thinking_renderer.create_conversation_prefix_with_tools(tools, system_prompt="")
    ours = list(
        thinking_renderer.build_generation_prompt(prefix + messages, role="assistant").to_ints()
    )
    _assert_match(tokenizer, hf, ours)


# ── Tool-call parse_response round trip ─────────────────────────────────────


def test_parse_response_extracts_tool_calls(tokenizer, renderer):
    """A model turn containing one or more tool-call blocks parses out into
    the ``tool_calls`` field, leaving any surrounding text on ``content``."""
    body = (
        '<|tool_call>call:get_weather{location:<|"|>paris<|"|>}<tool_call|>'
        "<turn|>"
    )
    encoded = tokenizer.encode(body, add_special_tokens=False)
    msg, ok = renderer.parse_response(encoded)
    assert ok
    assert msg.get("tool_calls"), "expected tool_calls to be populated"
    assert msg["tool_calls"][0]["name"] == "get_weather"  # type: ignore[index]
    # Surrounding content (none in this case) should be empty string.
    assert msg["content"] == ""


# ── End-to-end model parity on GPU ──────────────────────────────────────────


@pytest.fixture(scope="module")
def gpu_model(tokenizer):
    """Load the Gemma 4 model on the first available GPU.

    Skipped if torch / CUDA is not available, or if the model directory
    only contains tokenizer files (e.g. CI without weights).
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available for end-to-end model test.")
    path = _resolve_model_path()
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        )
    except (OSError, ValueError) as e:
        pytest.skip(f"Could not load Gemma 4 model from {path}: {e}")
    model.eval()
    return model


def _greedy_complete(model, tokenizer, prompt_ids: list[int], max_new_tokens: int = 32) -> str:
    import torch

    device = next(model.parameters()).device
    input_t = torch.tensor([prompt_ids], device=device)
    with torch.no_grad():
        out = model.generate(
            input_t,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, len(prompt_ids):], skip_special_tokens=True)


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "What is 2+2? Answer in one word."}],
        [
            {"role": "system", "content": "Reply in one short sentence."},
            {"role": "user", "content": "Capital of France?"},
        ],
    ],
    ids=["arithmetic", "system_geography"],
)
def test_e2e_model_parity_greedy(tokenizer, renderer, gpu_model, messages):
    """End-to-end model parity: feed both the renderer's prompt and HF's
    prompt through the actual Gemma 4 model and assert the greedy
    completions are byte-identical.

    This is the test that would have caught the previous "wrong role name"
    bug instantly — a `<|turn>assistant\\n` prompt is out-of-distribution
    for a model trained on `<|turn>model\\n`, and the model would have
    emitted garbage on token 1.
    """
    hf_ids = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    our_ids = _renderer_tokens(renderer, messages)
    assert hf_ids == our_ids, "Renderer prompt diverges from HF prompt at the token level"

    hf_completion = _greedy_complete(gpu_model, tokenizer, hf_ids)
    our_completion = _greedy_complete(gpu_model, tokenizer, our_ids)

    # Sanity: model produced real text, not an immediate stop token
    # (the failure mode of an out-of-distribution prompt).
    assert hf_completion.strip(), f"Model produced empty completion: {hf_completion!r}"
    assert "<turn|>" not in hf_completion, (
        f"Model immediately emitted stop token — prompt is out-of-distribution. "
        f"completion={hf_completion!r}"
    )
    # The two completions must be byte-identical because the prompts are
    # token-identical and decoding is greedy.
    assert hf_completion == our_completion, (
        f"Completion mismatch:\n  HF:   {hf_completion!r}\n  Ours: {our_completion!r}"
    )


# ── End-to-end tool-calling: real model → renderer → real parsers ──────────


_E2E_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {
                    "type": "string",
                    "description": "Temperature unit (c or f)",
                    "enum": ["c", "f"],
                },
            },
            "required": ["location"],
        },
    },
}


def test_e2e_tool_call_renderer_to_model_to_renderer_parser(
    tokenizer, renderer, gpu_model
):
    """End-to-end loop with tools:

    1. Build a tool-aware prompt with the renderer.
    2. Greedy-decode through the actual Gemma 4 model.
    3. Assert the model emits a properly-formed ``<|tool_call>...
       <tool_call|>`` block — proves our prompt is in-distribution for
       tool calling.
    4. Round-trip the raw token IDs through ``Gemma4Renderer.parse_response``
       and assert it extracts ``tool_calls`` with ``name='get_weather'``
       and a ``location`` argument.

    This is the integration test that catches the failure mode where the
    renderer output looks right token-for-token but doesn't actually
    drive the trained model to emit a tool call.
    """
    import torch  # noqa: F401  (skip-guarded by gpu_model fixture)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. When the user asks about "
                "weather, call the get_weather function."
            ),
        },
        {"role": "user", "content": "What is the weather in Paris in celsius?"},
    ]
    tools = [_E2E_TOOL]

    prefix = renderer.create_conversation_prefix_with_tools(
        tools, system_prompt=messages[0]["content"]
    )
    our_ids = list(
        renderer.build_generation_prompt(prefix + messages[1:], role="assistant").to_ints()
    )
    # Sanity: matches HF's tools= prompt token-for-token (the unit-test
    # tier already covers this, but pinning it here too makes failure
    # diagnosis easier when the model output is wrong).
    hf_ids = _hf_tokens(tokenizer, messages, tools=tools, add_generation_prompt=True)
    assert hf_ids == our_ids

    device = next(gpu_model.parameters()).device
    input_t = torch.tensor([our_ids], device=device)
    with torch.no_grad():
        out = gpu_model.generate(
            input_t,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0, len(our_ids):].tolist()

    decoded_with_specials = tokenizer.decode(gen_ids, skip_special_tokens=False)
    assert "<|tool_call>" in decoded_with_specials, (
        "Model did NOT emit a tool call from our prompt — wire format wrong? "
        f"Got: {decoded_with_specials!r}"
    )

    msg, ok = renderer.parse_response(gen_ids)
    assert ok, (
        f"parse_response did not find a stop token in real model output: "
        f"{decoded_with_specials!r}"
    )
    assert msg.get("tool_calls"), (
        f"parse_response did not extract tool_calls from: {decoded_with_specials!r}"
    )
    tc = msg["tool_calls"][0]  # type: ignore[index]
    assert tc["name"] == "get_weather"
    assert "location" in tc["arguments"], (
        f"Missing location arg in parsed tool call: {tc!r}"
    )
