"""Verify DeepseekV4Renderer matches the upstream encoder byte-for-byte.

The source-of-truth is ``deepseek-ai/DeepSeek-V4-Flash/encoding/encoding_dsv4.py``,
vendored verbatim under ``training/tests/_encoding_dsv4_oracle.py``. This
suite hits both layers:

1. **String parity** — ``renderer.decode(...)`` matches
   ``encoding_dsv4.encode_messages(...)`` character-for-character.
2. **Token parity** — ``renderer.build_*(...)`` matches
   ``tokenizer.encode(oracle_string, add_special_tokens=False)`` ID-for-ID.
3. **Weight correctness** — supervised weight masks satisfy the
   train-inference prefix invariant and the trained span roundtrips
   through ``encoding_dsv4.parse_message_from_completion_text``.

Run from cookbook/training with::

    PYTHONPATH=../.. python -m pytest training/tests/unit/test_deepseek_v4_renderer.py -v -s
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import transformers

from training.renderer.deepseek_v4 import (
    DeepseekV4Renderer,
    _BOS_TEXT,
    _EOS_TEXT,
    _THINK_OPEN,
    _THINK_CLOSE,
    _USER_SP,
    _ASSISTANT_SP,
    _DSML,
)
from training.tests import _encoding_dsv4_oracle as oracle
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import ToolCall, TrainOnWhat

# Public HF tokenizer. Local mirror probed first so internal CI works without
# the Hub; falls through to public if absent.
_LOCAL_TOKENIZER = "/shared/text-models/deepseek-v4-flash"
_PUBLIC_TOKENIZER = "deepseek-ai/DeepSeek-V4-Flash"


def _load_tokenizer() -> transformers.PreTrainedTokenizerBase | None:
    if Path(_LOCAL_TOKENIZER).exists():
        try:
            return transformers.AutoTokenizer.from_pretrained(
                _LOCAL_TOKENIZER,
                trust_remote_code=True,
            )
        except Exception:  # noqa: BLE001
            pass
    try:
        return transformers.AutoTokenizer.from_pretrained(
            _PUBLIC_TOKENIZER,
            trust_remote_code=True,
        )
    except Exception:  # noqa: BLE001
        return None


@pytest.fixture(scope="module")
def tokenizer():
    tok = _load_tokenizer()
    if tok is None:
        pytest.skip(
            f"DeepSeek-V4 tokenizer not available "
            f"(tried {_LOCAL_TOKENIZER!r} then {_PUBLIC_TOKENIZER!r})"
        )
    return tok


@pytest.fixture(scope="module")
def renderer_thinking_strip(tokenizer):
    """Default registered renderer: thinking mode, strip-history."""
    return DeepseekV4Renderer(
        tokenizer,
        thinking_mode="thinking",
        strip_thinking_from_history=True,
    )


@pytest.fixture(scope="module")
def renderer_thinking_keep(tokenizer):
    """Thinking mode that keeps history reasoning (extension property)."""
    return DeepseekV4Renderer(
        tokenizer,
        thinking_mode="thinking",
        strip_thinking_from_history=False,
    )


@pytest.fixture(scope="module")
def renderer_chat(tokenizer):
    """Chat mode (no thinking ever)."""
    return DeepseekV4Renderer(
        tokenizer,
        thinking_mode="chat",
        strip_thinking_from_history=True,
    )


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_tool_call(
    name: str, arguments: dict[str, Any], call_id: str = ""
) -> ToolCall:
    """Build a Tinker ``ToolCall`` from (name, dict-args)."""
    return ToolCall(
        function=ToolCall.FunctionBody(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
        id=call_id,
    )


def _oracle_tool_calls(
    specs: list[tuple[str, dict[str, Any], str]],
) -> list[dict[str, Any]]:
    """Build OpenAI-format tool_calls for the oracle from (name, args, id) specs."""
    return [
        {
            "type": "function",
            "id": call_id,
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }
        for name, args, call_id in specs
    ]


def _oracle_text(
    messages: list[dict[str, Any]], thinking_mode: str, **kwargs: Any
) -> str:
    """Run the upstream encoder; never let test fixtures share mutable state."""
    return oracle.encode_messages(
        [json.loads(json.dumps(m)) for m in messages],
        thinking_mode=thinking_mode,
        **kwargs,
    )


def _oracle_ids(
    tokenizer,
    messages: list[dict[str, Any]],
    thinking_mode: str,
    **kwargs: Any,
) -> list[int]:
    return tokenizer.encode(
        _oracle_text(messages, thinking_mode, **kwargs),
        add_special_tokens=False,
    )


def _renderer_supervised(renderer, messages, **kwargs):
    model_input, weights = renderer.build_supervised_example(messages, **kwargs)
    return list(model_input.to_ints()), [int(w) for w in weights.tolist()]


def _renderer_generation(renderer, messages, role: str = "assistant") -> list[int]:
    return list(renderer.build_generation_prompt(messages, role=role).to_ints())


def _trained_text(tokenizer, tokens: list[int], weights: list[int]) -> str:
    return tokenizer.decode([t for t, w in zip(tokens, weights) if w > 0])


# ── Registered renderer + special-token sanity ──────────────────────────────


def test_registered_factory_returns_thinking_strip(tokenizer):
    """``get_renderer("deepseek_v4", tok)`` matches our default constructor."""
    r = get_renderer("deepseek_v4", tokenizer)
    assert isinstance(r, DeepseekV4Renderer)
    assert r.thinking_mode == "thinking"
    assert r.strip_thinking_from_history is True
    assert r.has_extension_property is False


def test_keep_thinking_renderer_has_extension_property(renderer_thinking_keep):
    assert renderer_thinking_keep.has_extension_property is True


def test_chat_renderer_has_extension_property(renderer_chat):
    assert renderer_chat.has_extension_property is True


def test_special_tokens_each_encode_to_single_token(tokenizer):
    """All hand-coded special token strings tokenize as one ID each."""
    for tok_str in (_BOS_TEXT, _EOS_TEXT, _USER_SP, _ASSISTANT_SP):
        ids = tokenizer.encode(tok_str, add_special_tokens=False)
        assert len(ids) == 1, f"{tok_str!r} encoded to {ids}"
    # ``<think>`` / ``</think>`` are also single tokens for DSv4.
    for tok_str in (_THINK_OPEN, _THINK_CLOSE):
        ids = tokenizer.encode(tok_str, add_special_tokens=False)
        assert len(ids) == 1, f"{tok_str!r} encoded to {ids}"


def test_dsml_token_round_trips(tokenizer):
    """The DSML interior token survives decode→encode cleanly."""
    sample = f'<{_DSML}invoke name="foo">'
    ids = tokenizer.encode(sample, add_special_tokens=False)
    assert tokenizer.decode(ids) == sample


def test_get_stop_sequences_returns_eos(tokenizer, renderer_thinking_strip):
    eos_id = tokenizer.encode(_EOS_TEXT, add_special_tokens=False)[0]
    assert renderer_thinking_strip.get_stop_sequences() == [eos_id]


def test_generation_suffix_thinking_is_assistant_plus_think(
    tokenizer,
    renderer_thinking_strip,
):
    from tinker_cookbook.renderers.base import RenderContext

    ctx = RenderContext(idx=0, is_last=False, prev_message=None, last_user_index=-1)
    suffix = renderer_thinking_strip._get_generation_suffix("assistant", ctx)
    assert tokenizer.decode(suffix) == _ASSISTANT_SP + _THINK_OPEN


def test_generation_suffix_chat_is_assistant_plus_close_think(
    tokenizer,
    renderer_chat,
):
    from tinker_cookbook.renderers.base import RenderContext

    ctx = RenderContext(idx=0, is_last=False, prev_message=None, last_user_index=-1)
    suffix = renderer_chat._get_generation_suffix("assistant", ctx)
    assert tokenizer.decode(suffix) == _ASSISTANT_SP + _THINK_CLOSE


# ── Layer 1: string + token parity vs the oracle ────────────────────────────


_GENERATION_FIXTURES: list[list[dict[str, Any]]] = [
    # User-only, expecting an assistant generation prompt.
    [{"role": "user", "content": "Hello"}],
    # System + user.
    [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ],
    # Multi-turn history with reasoning.
    [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "reasoning_content": "2+2=4", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ],
    # Five-turn conversation, no reasoning.
    [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
        {"role": "assistant", "content": "goodbye"},
        {"role": "user", "content": "one more"},
    ],
    # Multi-line + unicode.
    [{"role": "user", "content": "line one\nline two\nline three\n你好 🚀"}],
    # System + user with a tool response already stitched in (covers
    # encoder's ``user→tool`` merge path).
    [
        {"role": "user", "content": "weather?"},
        {"role": "tool", "content": "sunny", "tool_call_id": "call_1"},
    ],
    # System with empty content.
    [
        {"role": "system", "content": ""},
        {"role": "user", "content": "hello"},
    ],
]


@pytest.mark.parametrize(
    "messages",
    _GENERATION_FIXTURES,
    ids=[f"gen{i}" for i in range(len(_GENERATION_FIXTURES))],
)
def test_generation_prompt_thinking_strip_matches_oracle(
    tokenizer,
    renderer_thinking_strip,
    messages,
):
    expected_text = _oracle_text(messages, thinking_mode="thinking")
    expected_ids = tokenizer.encode(expected_text, add_special_tokens=False)
    ours = _renderer_generation(renderer_thinking_strip, messages)
    assert ours == expected_ids, (
        f"thinking-strip generation mismatch:\n"
        f"  oracle text: {expected_text!r}\n"
        f"  ours text:   {tokenizer.decode(ours)!r}"
    )


@pytest.mark.parametrize(
    "messages",
    _GENERATION_FIXTURES,
    ids=[f"gen{i}" for i in range(len(_GENERATION_FIXTURES))],
)
def test_generation_prompt_thinking_keep_matches_oracle(
    tokenizer,
    renderer_thinking_keep,
    messages,
):
    expected_text = _oracle_text(
        messages,
        thinking_mode="thinking",
        drop_thinking=False,
    )
    expected_ids = tokenizer.encode(expected_text, add_special_tokens=False)
    ours = _renderer_generation(renderer_thinking_keep, messages)
    assert ours == expected_ids


@pytest.mark.parametrize(
    "messages",
    _GENERATION_FIXTURES,
    ids=[f"gen{i}" for i in range(len(_GENERATION_FIXTURES))],
)
def test_generation_prompt_chat_matches_oracle(
    tokenizer,
    renderer_chat,
    messages,
):
    expected_text = _oracle_text(messages, thinking_mode="chat")
    expected_ids = tokenizer.encode(expected_text, add_special_tokens=False)
    ours = _renderer_generation(renderer_chat, messages)
    assert ours == expected_ids


_SUPERVISED_FIXTURES: list[list[dict[str, Any]]] = [
    # Single-turn no thinking.
    [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "bye"},
    ],
    # Single-turn with reasoning.
    [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "reasoning_content": "the user said hi",
            "content": "bye",
        },
    ],
    # System + user + assistant.
    [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "reasoning_content": "trivial", "content": "A"},
    ],
    # Multi-turn with reasoning history.
    [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "reasoning_content": "r1", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "reasoning_content": "r2", "content": "a2"},
    ],
    # Three-turn no thinking.
    [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ],
    # Empty assistant content.
    [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},
    ],
    # Tool round-trip: assistant calls tool, tool responds, assistant answers.
    [
        {"role": "user", "content": "weather in SF?"},
        {
            "role": "assistant",
            "reasoning_content": "need lookup",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny, 72F"},
        {
            "role": "assistant",
            "reasoning_content": "got it",
            "content": "It's sunny and 72F.",
        },
    ],
]


def _renderer_messages_from(
    oracle_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert oracle-shape tool_calls (dict) into Tinker-shape (ToolCall)."""
    out: list[dict[str, Any]] = []
    for msg in oracle_messages:
        msg = json.loads(json.dumps(msg))
        if msg.get("tool_calls"):
            msg["tool_calls"] = [
                _make_tool_call(
                    tc["function"]["name"],
                    json.loads(tc["function"]["arguments"]),
                    call_id=tc.get("id", ""),
                )
                for tc in msg["tool_calls"]
            ]
        out.append(msg)
    return out


@pytest.mark.parametrize(
    "messages",
    _SUPERVISED_FIXTURES,
    ids=[f"sft{i}" for i in range(len(_SUPERVISED_FIXTURES))],
)
def test_supervised_thinking_strip_matches_oracle(
    tokenizer,
    renderer_thinking_strip,
    messages,
):
    expected_ids = _oracle_ids(tokenizer, messages, thinking_mode="thinking")
    ours, _ = _renderer_supervised(
        renderer_thinking_strip,
        _renderer_messages_from(messages),
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    assert ours == expected_ids, (
        f"thinking-strip SFT token mismatch:\n"
        f"  oracle: {tokenizer.decode(expected_ids)!r}\n"
        f"  ours:   {tokenizer.decode(ours)!r}"
    )


@pytest.mark.parametrize(
    "messages",
    _SUPERVISED_FIXTURES,
    ids=[f"sft{i}" for i in range(len(_SUPERVISED_FIXTURES))],
)
def test_supervised_thinking_keep_matches_oracle(
    tokenizer,
    renderer_thinking_keep,
    messages,
):
    expected_ids = _oracle_ids(
        tokenizer,
        messages,
        thinking_mode="thinking",
        drop_thinking=False,
    )
    ours, _ = _renderer_supervised(
        renderer_thinking_keep,
        _renderer_messages_from(messages),
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    assert ours == expected_ids


@pytest.mark.parametrize(
    "messages",
    _SUPERVISED_FIXTURES,
    ids=[f"sft{i}" for i in range(len(_SUPERVISED_FIXTURES))],
)
def test_supervised_chat_matches_oracle(tokenizer, renderer_chat, messages):
    # Strip reasoning_content from chat-mode fixtures: the encoder ignores
    # it in chat mode, but our oracle invocation should reflect that the
    # model wouldn't have generated it.
    chat_msgs = [
        {k: v for k, v in m.items() if k != "reasoning_content"} for m in messages
    ]
    expected_ids = _oracle_ids(tokenizer, chat_msgs, thinking_mode="chat")
    ours, _ = _renderer_supervised(
        renderer_chat,
        _renderer_messages_from(chat_msgs),
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    assert ours == expected_ids


# ── Tools / response_format on system message ───────────────────────────────


def test_system_with_tools_matches_oracle(tokenizer, renderer_thinking_keep):
    """System message carrying ``tools`` renders the ``## Tools`` section.

    The encoder also force-disables ``drop_thinking`` whenever any message
    has ``tools``; we use ``renderer_thinking_keep`` to mirror that.
    """
    messages = [
        {
            "role": "system",
            "content": "You are an assistant.",
            "tools": [
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
            ],
        },
        {"role": "user", "content": "weather in SF?"},
    ]
    expected_ids = _oracle_ids(tokenizer, messages, thinking_mode="thinking")
    ours = _renderer_generation(renderer_thinking_keep, messages)
    assert ours == expected_ids, tokenizer.decode(ours)


def test_drop_thinking_auto_disable_when_tools_present(
    tokenizer,
    renderer_thinking_strip,
):
    """The encoder auto-flips drop_thinking=False when any message has tools.

    Same conversation rendered with and without a tools-bearing system
    must satisfy: history reasoning is dropped without tools, preserved
    with tools — matching the encoder's behavior even though our
    renderer was constructed with strip_thinking_from_history=True.
    """
    base_messages = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "reasoning_content": "HISTORY_REASONING",
            "content": "a1",
        },
        {"role": "user", "content": "q2"},
    ]
    with_tools = [
        {
            "role": "system",
            "content": "be helpful",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "x",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
        *base_messages,
    ]

    # Without tools: HISTORY_REASONING must be dropped.
    no_tools_text = _oracle_text(base_messages, thinking_mode="thinking")
    no_tools_ours = tokenizer.decode(
        _renderer_generation(renderer_thinking_strip, base_messages)
    )
    assert no_tools_ours == no_tools_text
    assert "HISTORY_REASONING" not in no_tools_ours

    # With tools: HISTORY_REASONING must be preserved (auto-flip).
    with_tools_text = _oracle_text(with_tools, thinking_mode="thinking")
    with_tools_ours = tokenizer.decode(
        _renderer_generation(renderer_thinking_strip, with_tools)
    )
    assert with_tools_ours == with_tools_text
    assert "HISTORY_REASONING" in with_tools_ours


# ── Tool-call argument formatting (mixed types) ─────────────────────────────


def test_tool_call_mixed_argument_types_matches_oracle(
    tokenizer,
    renderer_thinking_keep,
):
    """``string="true"`` for str args, ``string="false"`` (JSON) for the rest."""
    oracle_messages = [
        {"role": "user", "content": "do work"},
        {
            "role": "assistant",
            "reasoning_content": "go",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {
                        "name": "f",
                        "arguments": json.dumps(
                            {
                                "name": "alice",  # str → string="true"
                                "count": 7,  # int → string="false"
                                "active": True,  # bool → string="false"
                                "tags": ["a", "b"],  # list → string="false"
                                "meta": {"k": 1},  # dict → string="false"
                            }
                        ),
                    },
                }
            ],
        },
    ]
    expected_ids = _oracle_ids(
        tokenizer,
        oracle_messages,
        thinking_mode="thinking",
        drop_thinking=False,
    )
    ours, _ = _renderer_supervised(
        renderer_thinking_keep,
        _renderer_messages_from(oracle_messages),
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    assert ours == expected_ids


def test_parallel_tool_results_reorder_to_call_order(
    tokenizer,
    renderer_thinking_keep,
):
    """tool_results in user are reordered to match preceding tool_calls."""
    oracle_messages = [
        {"role": "user", "content": "two tools"},
        {
            "role": "assistant",
            "reasoning_content": "",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "weather",
                    "function": {
                        "name": "weather",
                        "arguments": json.dumps({"c": "SF"}),
                    },
                },
                {
                    "type": "function",
                    "id": "convert",
                    "function": {"name": "convert", "arguments": json.dumps({"v": 1})},
                },
            ],
        },
        # Out-of-order tool responses: convert before weather.
        {"role": "tool", "tool_call_id": "convert", "content": "CONVERT_RESULT"},
        {"role": "tool", "tool_call_id": "weather", "content": "WEATHER_RESULT"},
        {"role": "assistant", "reasoning_content": "done", "content": "ok"},
    ]
    expected_ids = _oracle_ids(
        tokenizer,
        oracle_messages,
        thinking_mode="thinking",
        drop_thinking=False,
    )
    ours, _ = _renderer_supervised(
        renderer_thinking_keep,
        _renderer_messages_from(oracle_messages),
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    assert ours == expected_ids
    decoded = tokenizer.decode(ours)
    # Sorting must put weather before convert (call order).
    assert decoded.index("WEATHER_RESULT") < decoded.index("CONVERT_RESULT")


# ── Layer 2: train-inference prefix invariant (the bedrock check) ───────────


@pytest.mark.parametrize(
    "messages",
    _SUPERVISED_FIXTURES,
    ids=[f"inv{i}" for i in range(len(_SUPERVISED_FIXTURES))],
)
def test_supervised_extends_generation_prompt_thinking_keep(
    tokenizer,
    renderer_thinking_keep,
    messages,
):
    """gen_prompt(messages[:-1]) must be a strict 0-weight prefix of supervised(messages).

    The remaining tokens are exactly what the model would have produced
    at inference time, and must all carry weight=1.
    """
    rmsgs = _renderer_messages_from(messages)
    gen = _renderer_generation(renderer_thinking_keep, rmsgs[:-1])
    sup, weights = _renderer_supervised(
        renderer_thinking_keep,
        rmsgs,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    assert sup[: len(gen)] == gen, (
        f"prefix mismatch:\n  gen:  {tokenizer.decode(gen)!r}\n  "
        f"sup[:len(gen)]: {tokenizer.decode(sup[: len(gen)])!r}"
    )
    assert all(w == 0 for w in weights[: len(gen)]), (
        "generation-prompt prefix tokens must have weight 0; got "
        f"{weights[: len(gen)]}"
    )
    assert all(w == 1 for w in weights[len(gen) :]), (
        f"model-output suffix must have weight 1; trained text was "
        f"{tokenizer.decode([t for t, w in zip(sup, weights) if w > 0])!r}"
    )


@pytest.mark.parametrize(
    "messages",
    _SUPERVISED_FIXTURES,
    ids=[f"inv-strip{i}" for i in range(len(_SUPERVISED_FIXTURES))],
)
def test_supervised_extends_generation_prompt_thinking_strip(
    tokenizer,
    renderer_thinking_strip,
    messages,
):
    rmsgs = _renderer_messages_from(messages)
    gen = _renderer_generation(renderer_thinking_strip, rmsgs[:-1])
    sup, weights = _renderer_supervised(
        renderer_thinking_strip,
        rmsgs,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    assert sup[: len(gen)] == gen
    assert all(w == 0 for w in weights[: len(gen)])
    assert all(w == 1 for w in weights[len(gen) :])


@pytest.mark.parametrize(
    "messages",
    _SUPERVISED_FIXTURES,
    ids=[f"inv-chat{i}" for i in range(len(_SUPERVISED_FIXTURES))],
)
def test_supervised_extends_generation_prompt_chat(
    tokenizer,
    renderer_chat,
    messages,
):
    chat_msgs = [
        {k: v for k, v in m.items() if k != "reasoning_content"} for m in messages
    ]
    rmsgs = _renderer_messages_from(chat_msgs)
    gen = _renderer_generation(renderer_chat, rmsgs[:-1])
    sup, weights = _renderer_supervised(
        renderer_chat,
        rmsgs,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    assert sup[: len(gen)] == gen
    assert all(w == 0 for w in weights[: len(gen)])
    assert all(w == 1 for w in weights[len(gen) :])


# ── Layer 3: trained-span roundtrips through the oracle parser ─────────────


def _roundtrip_oracle_parse(
    tokenizer,
    renderer,
    messages: list[dict[str, Any]],
    thinking_mode: str,
) -> dict[str, Any]:
    """Decode the trained span and parse it via the oracle's own parser."""
    rmsgs = _renderer_messages_from(messages)
    sup, weights = _renderer_supervised(
        renderer,
        rmsgs,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    trained_ids = [t for t, w in zip(sup, weights) if w > 0]
    trained_text = tokenizer.decode(trained_ids)

    if thinking_mode == "thinking":
        # The oracle parser expects the closing </think> but NOT the
        # opening <think> (which is part of the generation prefix).
        # Our trained span already has that shape.
        pass
    return oracle.parse_message_from_completion_text(trained_text, thinking_mode)


def test_trained_span_roundtrips_simple_chat(tokenizer, renderer_chat):
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "bye"},
    ]
    parsed = _roundtrip_oracle_parse(tokenizer, renderer_chat, msgs, "chat")
    assert parsed["content"] == "bye"
    assert parsed["reasoning_content"] == ""
    assert parsed["tool_calls"] == []


def test_trained_span_roundtrips_thinking(tokenizer, renderer_thinking_keep):
    msgs = [
        {"role": "user", "content": "What's 2+2?"},
        {"role": "assistant", "reasoning_content": "compute 2+2", "content": "4"},
    ]
    parsed = _roundtrip_oracle_parse(
        tokenizer,
        renderer_thinking_keep,
        msgs,
        "thinking",
    )
    assert parsed["content"] == "4"
    assert parsed["reasoning_content"] == "compute 2+2"
    assert parsed["tool_calls"] == []


def test_trained_span_roundtrips_with_tool_call(tokenizer, renderer_thinking_keep):
    msgs = [
        {"role": "user", "content": "weather"},
        {
            "role": "assistant",
            "reasoning_content": "need lookup",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF", "n": 1}),
                    },
                }
            ],
        },
    ]
    parsed = _roundtrip_oracle_parse(
        tokenizer,
        renderer_thinking_keep,
        msgs,
        "thinking",
    )
    assert parsed["content"] == ""
    assert parsed["reasoning_content"] == "need lookup"
    assert len(parsed["tool_calls"]) == 1
    tc = parsed["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"
    # Oracle decoder produces a hand-built JSON string with raw values for
    # non-string params; load it back to assert the semantic shape.
    assert json.loads(tc["function"]["arguments"]) == {"city": "SF", "n": 1}


# ── Layer 4: targeted weight-mask invariants ────────────────────────────────


def test_weight_mask_eos_is_trained(tokenizer, renderer_thinking_keep):
    """The EOS at the end of a trainable assistant carries weight=1.

    DSv4 uses EOS as the assistant stop marker, so the model must learn
    to produce it.
    """
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "reasoning_content": "r", "content": "bye"},
    ]
    eos_id = tokenizer.encode(_EOS_TEXT, add_special_tokens=False)[0]
    sup, weights = _renderer_supervised(
        renderer_thinking_keep,
        _renderer_messages_from(msgs),
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    assert sup[-1] == eos_id
    assert weights[-1] == 1


def test_weight_mask_assistant_header_is_masked(tokenizer, renderer_thinking_keep):
    """``<|Assistant|><think>`` is the boundary — must be weight 0."""
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "reasoning_content": "r", "content": "bye"},
    ]
    asst_id = tokenizer.encode(_ASSISTANT_SP, add_special_tokens=False)[0]
    think_open_id = tokenizer.encode(_THINK_OPEN, add_special_tokens=False)[0]
    sup, weights = _renderer_supervised(
        renderer_thinking_keep,
        _renderer_messages_from(msgs),
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    asst_pos = sup.index(asst_id)
    assert weights[asst_pos] == 0, "<|Assistant|> tag must be masked"
    assert sup[asst_pos + 1] == think_open_id
    assert weights[asst_pos + 1] == 0, "boundary <think> must be masked"
    # The token immediately after the boundary <think> is part of the
    # model's reasoning and MUST train.
    assert (
        weights[asst_pos + 2] == 1
    ), "first reasoning token after <think> must be trained"


def test_weight_mask_chat_mode_close_think_is_masked(tokenizer, renderer_chat):
    """In chat mode, the boundary ``</think>`` is also masked."""
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "bye"},
    ]
    asst_id = tokenizer.encode(_ASSISTANT_SP, add_special_tokens=False)[0]
    think_close_id = tokenizer.encode(_THINK_CLOSE, add_special_tokens=False)[0]
    sup, weights = _renderer_supervised(
        renderer_chat,
        _renderer_messages_from(msgs),
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    asst_pos = sup.index(asst_id)
    assert weights[asst_pos] == 0
    assert sup[asst_pos + 1] == think_close_id
    assert weights[asst_pos + 1] == 0
    # The first content token (`bye`) must train.
    assert weights[asst_pos + 2] == 1


def test_weight_mask_user_and_system_are_masked(tokenizer, renderer_thinking_keep):
    msgs = [
        {"role": "system", "content": "SYSTEM_PROMPT"},
        {"role": "user", "content": "USER_QUERY"},
        {"role": "assistant", "reasoning_content": "REASON", "content": "ANSWER"},
    ]
    sup, weights = _renderer_supervised(
        renderer_thinking_keep,
        _renderer_messages_from(msgs),
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    decoded_full = tokenizer.decode(sup)
    trained = _trained_text(tokenizer, sup, weights)

    # Sanity: rendered output contains everything.
    assert "SYSTEM_PROMPT" in decoded_full
    assert "USER_QUERY" in decoded_full
    assert "REASON" in decoded_full
    assert "ANSWER" in decoded_full

    # Trained span contains only model output.
    assert "SYSTEM_PROMPT" not in trained
    assert "USER_QUERY" not in trained
    assert "REASON" in trained
    assert "ANSWER" in trained
    assert _USER_SP not in trained
    assert _ASSISTANT_SP not in trained
    assert trained.endswith(_EOS_TEXT)


def test_weight_mask_tool_result_is_masked(tokenizer, renderer_thinking_keep):
    """``<tool_result>...</tool_result>`` is environment-supplied → weight 0."""
    msgs = [
        {"role": "user", "content": "weather"},
        {
            "role": "assistant",
            "reasoning_content": "r",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "TOOL_RESULT_PAYLOAD"},
        {"role": "assistant", "reasoning_content": "got it", "content": "FINAL_ANSWER"},
    ]
    sup, weights = _renderer_supervised(
        renderer_thinking_keep,
        _renderer_messages_from(msgs),
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    trained = _trained_text(tokenizer, sup, weights)
    assert "TOOL_RESULT_PAYLOAD" not in trained
    assert "FINAL_ANSWER" in trained
    assert "got it" in trained


def test_weight_mask_tool_call_body_is_trained(tokenizer, renderer_thinking_keep):
    """The model produces the DSML tool-call body, so it must train."""
    msgs = [
        {"role": "user", "content": "weather"},
        {
            "role": "assistant",
            "reasoning_content": "r",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "PARAM_VALUE_SF"}),
                    },
                }
            ],
        },
    ]
    sup, weights = _renderer_supervised(
        renderer_thinking_keep,
        _renderer_messages_from(msgs),
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    trained = _trained_text(tokenizer, sup, weights)
    assert "PARAM_VALUE_SF" in trained
    assert "get_weather" in trained
    assert _DSML in trained


# ── Multi-turn weight check (extension property) ────────────────────────────


def test_multi_turn_keeps_history_reasoning_in_keep_mode(
    tokenizer,
    renderer_thinking_keep,
):
    """When extension property holds, ALL_ASSISTANT_MESSAGES trains every turn."""
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "reasoning_content": "R1", "content": "A1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "reasoning_content": "R2", "content": "A2"},
    ]
    sup, weights = _renderer_supervised(
        renderer_thinking_keep,
        _renderer_messages_from(msgs),
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    trained = _trained_text(tokenizer, sup, weights)
    # Both turns' reasoning + content trained; user content not trained.
    assert "R1" in trained
    assert "A1" in trained
    assert "R2" in trained
    assert "A2" in trained
    assert "q1" not in trained
    assert "q2" not in trained


def test_strip_history_drops_first_turn_reasoning(
    tokenizer,
    renderer_thinking_strip,
):
    """In strip mode, the first turn's reasoning is dropped from the rendering.

    LAST_ASSISTANT_MESSAGE only trains the terminal turn anyway, so this
    asserts the *rendering* of historical assistants doesn't contain
    their reasoning at all (let alone get it trained).
    """
    msgs = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "reasoning_content": "DROPPED_HISTORY_REASONING",
            "content": "A1",
        },
        {"role": "user", "content": "q2"},
        {"role": "assistant", "reasoning_content": "R2", "content": "A2"},
    ]
    sup, weights = _renderer_supervised(
        renderer_thinking_strip,
        _renderer_messages_from(msgs),
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    decoded_full = tokenizer.decode(sup)
    assert "DROPPED_HISTORY_REASONING" not in decoded_full
    # Terminal reasoning still rendered + trained.
    trained = _trained_text(tokenizer, sup, weights)
    assert "R2" in trained
    assert "A2" in trained


# ── parse_response sanity ───────────────────────────────────────────────────


def test_parse_response_recovers_thinking_and_content(
    tokenizer,
    renderer_thinking_keep,
):
    """``parse_response(ids)`` recovers reasoning + content from a model-emit."""
    text = "the answer is 4"
    full = "compute" + _THINK_CLOSE + text + _EOS_TEXT
    ids = tokenizer.encode(full, add_special_tokens=False)
    msg, ok = renderer_thinking_keep.parse_response(ids)
    assert ok is True
    assert msg.get("reasoning_content") == "compute"
    assert msg["content"] == text


def test_parse_response_returns_false_without_eos(tokenizer, renderer_thinking_keep):
    text = "compute" + _THINK_CLOSE + "partial answer"
    ids = tokenizer.encode(text, add_special_tokens=False)
    msg, ok = renderer_thinking_keep.parse_response(ids)
    assert ok is False
    assert msg["content"] == "partial answer"


def test_parse_response_extracts_tool_call(tokenizer, renderer_thinking_keep):
    """``\\n\\n<|DSML|tool_calls>...`` block round-trips into a ``ToolCall``."""
    body = (
        f"reasoning{_THINK_CLOSE}"
        f"\n\n<{_DSML}tool_calls>\n"
        f'<{_DSML}invoke name="f">\n'
        f'<{_DSML}parameter name="x" string="false">42</{_DSML}parameter>\n'
        f"</{_DSML}invoke>\n"
        f"</{_DSML}tool_calls>"
        f"{_EOS_TEXT}"
    )
    ids = tokenizer.encode(body, add_special_tokens=False)
    msg, ok = renderer_thinking_keep.parse_response(ids)
    assert ok is True
    assert msg.get("reasoning_content") == "reasoning"
    assert msg["content"] == ""
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc.function.name == "f"
    assert json.loads(tc.function.arguments) == {"x": 42}
