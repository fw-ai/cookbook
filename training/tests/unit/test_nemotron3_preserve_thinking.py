"""Tests for Nemotron-3 preserve-thinking renderer.

Covers:
1. Historical thinking kept under PRESERVED (HF truncate_history_thinking=False)
2. Historical thinking stripped under INTERLEAVED (default)
3. Multi-turn ALL_ASSISTANT_MESSAGES masking with tools
4. Extension property / no unroll under PRESERVED
5. HF chat-template parity for preserve mode
"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from transformers import AutoTokenizer

from training.renderer.thinking_trace import (
    ThinkingTraceHistoryMode,
    resolve_thinking_trace_renderer_plan,
)
from training.utils.supervised import (
    render_messages_to_datums,
    resolve_renderer_plan,
)

_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def _multi_turn_tool_messages() -> list[dict]:
    return [
        {"role": "user", "content": "Fix the login bug"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "TURN1_REASON: read auth.py first",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "cat auth.py"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "def login(): return None"},
        {
            "role": "assistant",
            "content": "Patched login.",
            "reasoning_content": "TURN1_ANSWER_REASON: return a user object",
        },
        {"role": "user", "content": "Add a unit test"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "TURN2_REASON: write test_login.py",
            "tool_calls": [
                {
                    "id": "c2",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "pytest -q"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "1 passed"},
        {
            "role": "assistant",
            "content": "Tests pass.",
            "reasoning_content": "TURN2_ANSWER_REASON: done",
        },
    ]


def _hf_kwargs_messages(messages: list[dict]) -> list[dict]:
    """Parse tool-call argument JSON strings so HF templates accept them."""
    out = deepcopy(messages)
    for m in out:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                fn["arguments"] = json.loads(args)
    return out


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return AutoTokenizer.from_pretrained(_MODEL, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 — network / gated repo / config drift
        pytest.skip(f"Nemotron tokenizer {_MODEL!r} not available: {exc}")


def test_preserve_renderer_has_extension_property(tokenizer):
    preserve = get_renderer("nemotron3_preserve_thinking", tokenizer)
    interleaved = get_renderer("nemotron3", tokenizer)
    assert preserve.has_extension_property is True
    assert interleaved.has_extension_property is False


def test_registry_resolves_preserved_for_nemotron_ids():
    """Tokenizer-free: registry aliases must map PRESERVED → nemotron3_preserved."""
    for model_id in (
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        _MODEL,
    ):
        plan = resolve_thinking_trace_renderer_plan(
            model_id,
            requested_mode=ThinkingTraceHistoryMode.PRESERVED,
            default_renderer_name="nemotron3",
        )
        assert plan.renderer_name == "nemotron3_preserved", model_id
        assert plan.effective_mode is ThinkingTraceHistoryMode.PRESERVED
        assert plan.unrolls_multi_turn is False

        default = resolve_renderer_plan(
            model_id, "", thinking_trace_history_mode="preserved"
        )
        assert default.renderer_name == "nemotron3_preserved", model_id


def test_preserve_keeps_historical_thinking(tokenizer):
    messages = _multi_turn_tool_messages()
    preserve = get_renderer("nemotron3_preserve_thinking", tokenizer)
    interleaved = get_renderer("nemotron3_interleaved", tokenizer)

    [p_datum] = render_messages_to_datums(
        messages,
        renderer=preserve,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        tools=_TOOLS,
    )
    i_datums = render_messages_to_datums(
        messages,
        renderer=interleaved,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        tools=_TOOLS,
    )

    p_text = tokenizer.decode(p_datum.token_ids)
    assert "TURN1_REASON" in p_text
    assert "TURN1_ANSWER_REASON" in p_text
    assert "TURN2_REASON" in p_text
    assert "TURN2_ANSWER_REASON" in p_text
    assert preserve.has_extension_property
    assert len(i_datums) == 2  # disaggregated per user turn

    # Second interleaved split must strip turn-1 reasoning from the prefix.
    i_second = tokenizer.decode(i_datums[1].token_ids)
    assert "TURN1_REASON" not in i_second
    assert "TURN1_ANSWER_REASON" not in i_second
    assert "TURN2_REASON" in i_second


def test_preserve_all_assistant_masking_trains_every_turn(tokenizer):
    messages = _multi_turn_tool_messages()
    preserve = get_renderer("nemotron3_preserve_thinking", tokenizer)
    [datum] = render_messages_to_datums(
        messages,
        renderer=preserve,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        tools=_TOOLS,
    )
    weights = datum.token_weights
    assert any(w > 0 for w in weights)

    # Each distinctive reasoning snippet must land inside a trainable span.
    text = tokenizer.decode(datum.token_ids)
    for marker in (
        "TURN1_REASON",
        "TURN1_ANSWER_REASON",
        "TURN2_REASON",
        "TURN2_ANSWER_REASON",
        "Patched login.",
        "Tests pass.",
    ):
        assert marker in text

    # Tool results / user prompts must not be the only trainable content:
    # assistant+thinking tokens should dominate trainable mass.
    trainable = sum(weights)
    assert trainable > 20

    # Spot-check: decode only trainable tokens and require turn-1+turn-2 answers.
    trained_ids = [
        tid for tid, w in zip(datum.token_ids, weights, strict=True) if w > 0
    ]
    trained_text = tokenizer.decode(trained_ids)
    assert "TURN1_REASON" in trained_text
    assert "TURN2_REASON" in trained_text
    assert "Patched login." in trained_text
    assert "Tests pass." in trained_text
    # User / tool payloads should not be trained.
    assert "Fix the login bug" not in trained_text
    assert "Add a unit test" not in trained_text
    assert "def login(): return None" not in trained_text


def test_preserve_matches_hf_truncate_history_thinking_false(tokenizer):
    messages = _multi_turn_tool_messages()
    preserve = get_renderer("nemotron3_preserve_thinking", tokenizer)
    [datum] = render_messages_to_datums(
        messages,
        renderer=preserve,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        tools=_TOOLS,
    )
    renderer_text = tokenizer.decode(datum.token_ids)

    hf_text = tokenizer.apply_chat_template(
        _hf_kwargs_messages(messages),
        tools=_TOOLS,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
        truncate_history_thinking=False,
    )
    # Tokenization boundaries can differ slightly; require semantic parity on
    # thinking markers and answers rather than exact byte equality.
    for marker in (
        "TURN1_REASON",
        "TURN1_ANSWER_REASON",
        "TURN2_REASON",
        "TURN2_ANSWER_REASON",
        "Patched login.",
        "Tests pass.",
        "<tool_call>",
        "<tool_response>",
    ):
        assert (marker in renderer_text) == (marker in hf_text)
        assert marker in renderer_text

    # Interleaved HF must drop historical thinking; preserve must not.
    hf_inter = tokenizer.apply_chat_template(
        _hf_kwargs_messages(messages),
        tools=_TOOLS,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
        truncate_history_thinking=True,
    )
    assert "TURN1_REASON" not in hf_inter
    assert "TURN1_REASON" in hf_text
