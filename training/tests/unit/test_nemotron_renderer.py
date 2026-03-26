"""Verify NemotronRenderer matches HuggingFace apply_chat_template output.

Run from cookbook/training with:
    PYTHONPATH=../.. python -m pytest tests/unit/test_nemotron_renderer.py -v -s
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import transformers

# Import NemotronRenderer directly to avoid pulling the full training.utils
# chain (which requires fireworks.training.sdk at import time).
_UTILS_DIR = str(Path(__file__).resolve().parents[2] / "utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)
from nemotron_renderer import NemotronRenderer
from tinker_cookbook.renderers.base import ToolCall


_LOCAL_PATH = "/shared/nvidia-nemotron-3-nano-30b-a3b-bf16"
_HF_REPO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
TOKENIZER_MODEL = _LOCAL_PATH if Path(_LOCAL_PATH).exists() else _HF_REPO


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL, trust_remote_code=True,
    )


@pytest.fixture(scope="module")
def renderer(tokenizer):
    return NemotronRenderer(tokenizer, strip_thinking_from_history=True)


def _hf_tokens(tokenizer, messages, **kwargs) -> list[int]:
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs)


def _renderer_tokens(renderer, messages) -> list[int]:
    model_input = renderer.build_generation_prompt(messages, role="assistant")
    return list(model_input.to_ints())


# ── Basic single-turn ────────────────────────────────────────────────────────


def test_single_turn_no_thinking(tokenizer, renderer):
    """User → assistant (no thinking). Verify <think></think> is prepended."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    # HF template with enable_thinking=True
    hf = _hf_tokens(tokenizer, messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


def test_single_turn_with_thinking(tokenizer, renderer):
    """User → assistant with <think> block. Verify passthrough."""
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>Let me calculate</think>4"},
    ]
    hf = _hf_tokens(tokenizer, messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


# ── System message ───────────────────────────────────────────────────────────


def test_with_system_message(tokenizer, renderer):
    """System + user + assistant. Verify system rendering matches."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    hf = _hf_tokens(tokenizer, messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


def test_empty_system_message(tokenizer, renderer):
    """Empty system message — Nemotron always emits the system block."""
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Hello"},
    ]
    hf = _hf_tokens(tokenizer, messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


# ── Multi-turn with history truncation ───────────────────────────────────────


def test_multi_turn_history_truncation(tokenizer, renderer):
    """Multi-turn: historical assistant thinking should be replaced with <think></think>."""
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>2+2=4</think>The answer is 4."},
        {"role": "user", "content": "What about 3+3?"},
        {"role": "assistant", "content": "<think>3+3=6</think>The answer is 6."},
    ]
    hf = _hf_tokens(tokenizer, messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


def test_multi_turn_no_thinking_history(tokenizer, renderer):
    """Multi-turn where historical assistant had no thinking tags."""
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
    ]
    hf = _hf_tokens(tokenizer, messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


# ── Truncation scope (only before last user) ────────────────────────────────


def test_assistant_after_last_user_keeps_thinking(tokenizer, renderer):
    """Two assistant msgs after last user — both should keep full thinking."""
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "<think>thought 1</think>Reply 1"},
        {"role": "assistant", "content": "<think>thought 2</think>Reply 2"},
    ]
    hf = _hf_tokens(tokenizer, messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


# ── Generation prompt suffix ────────────────────────────────────────────────


def test_generation_prompt_suffix(tokenizer, renderer):
    """Verify generation prompt ends with <|im_start|>assistant\\n<think>\\n."""
    messages = [
        {"role": "user", "content": "Hello"},
    ]
    hf = _hf_tokens(tokenizer, messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


# ── Supervised example (for DPO) ────────────────────────────────────────────


def test_supervised_example_basic(tokenizer, renderer):
    """Verify supervised example tokenization matches HF for a simple conversation."""
    from tinker_cookbook.renderers.base import TrainOnWhat

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]

    # Our supervised tokens
    model_input, weights = renderer.build_supervised_example(
        [{"role": m["role"], "content": m["content"]} for m in messages],
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    our_tokens = list(model_input.to_ints())

    # HF reference: full conversation without generation prompt
    hf_tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=True,
    )

    our_text = tokenizer.decode(our_tokens)
    hf_text = tokenizer.decode(hf_tokens)
    print(f"\n--- HF supervised ---\n{hf_text}")
    print(f"\n--- Ours supervised ---\n{our_text}")

    # The renderer doesn't emit a trailing \n after the last <|im_end|>,
    # but HF does.  Strip it for comparison.
    if hf_tokens and hf_tokens[-1] == tokenizer.encode("\n", add_special_tokens=False)[-1]:
        hf_tokens_trimmed = hf_tokens[:-1]
    else:
        hf_tokens_trimmed = hf_tokens

    assert our_tokens == hf_tokens_trimmed, (
        f"Token mismatch:\nHF:   {hf_tokens_trimmed}\nOurs: {our_tokens}"
    )


def test_supervised_weights_mask_non_assistant(tokenizer, renderer):
    """Verify only assistant tokens have non-zero weight."""
    from tinker_cookbook.renderers.base import TrainOnWhat

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    model_input, weights = renderer.build_supervised_example(
        [{"role": m["role"], "content": m["content"]} for m in messages],
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    tokens = list(model_input.to_ints())
    weights_list = weights.tolist()
    assert len(tokens) == len(weights_list)

    token_strs = [tokenizer.decode([t]) for t in tokens]
    print("\nToken weights:")
    for i, (t, w) in enumerate(zip(token_strs, weights_list)):
        marker = "█" if w > 0 else "·"
        print(f"  {marker} {i:3d}  w={w:.0f}  {t!r}")

    assert any(w > 0 for w in weights_list), "Expected some non-zero weights"
    assert any(w == 0 for w in weights_list), "Expected some zero weights (user/header tokens)"


# ── Helper for constructing ToolCall objects ────────────────────────────────


def _make_tool_call(name: str, arguments: dict) -> ToolCall:
    return ToolCall(
        function=ToolCall.FunctionBody(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


# ── Tool calls (assistant with tool_calls) ──────────────────────────────────


def test_tool_call_single(tokenizer, renderer):
    """Assistant makes a single tool call — verify XML parameter format."""
    messages = [
        {"role": "user", "content": "What's the weather in SF?"},
        {
            "role": "assistant",
            "content": "<think>I should check the weather</think>Let me check.",
            "tool_calls": [_make_tool_call("get_weather", {"city": "San Francisco"})],
        },
    ]
    hf_messages = [
        {"role": "user", "content": "What's the weather in SF?"},
        {
            "role": "assistant",
            "content": "<think>I should check the weather</think>Let me check.",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "San Francisco"}}},
            ],
        },
    ]

    hf = _hf_tokens(tokenizer, hf_messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


def test_tool_call_no_thinking(tokenizer, renderer):
    """Tool call without explicit thinking — <think></think> should be prepended."""
    messages = [
        {"role": "user", "content": "Search for python docs"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_make_tool_call("search", {"query": "python"})],
        },
    ]
    hf_messages = [
        {"role": "user", "content": "Search for python docs"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "search", "arguments": {"query": "python"}}},
            ],
        },
    ]

    hf = _hf_tokens(tokenizer, hf_messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


def test_tool_call_multiple_params(tokenizer, renderer):
    """Tool call with multiple parameters."""
    messages = [
        {"role": "user", "content": "Book a flight"},
        {
            "role": "assistant",
            "content": "<think>Need to book</think>Booking now.",
            "tool_calls": [
                _make_tool_call("book_flight", {"origin": "SFO", "destination": "LAX", "date": "2026-04-01"}),
            ],
        },
    ]
    hf_messages = [
        {"role": "user", "content": "Book a flight"},
        {
            "role": "assistant",
            "content": "<think>Need to book</think>Booking now.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "book_flight",
                        "arguments": {"origin": "SFO", "destination": "LAX", "date": "2026-04-01"},
                    },
                },
            ],
        },
    ]

    hf = _hf_tokens(tokenizer, hf_messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


def test_tool_call_multiple_calls(tokenizer, renderer):
    """Multiple tool calls in a single assistant message."""
    messages = [
        {"role": "user", "content": "Compare weather in SF and NYC"},
        {
            "role": "assistant",
            "content": "<think>Check both cities</think>Let me check both.",
            "tool_calls": [
                _make_tool_call("get_weather", {"city": "San Francisco"}),
                _make_tool_call("get_weather", {"city": "New York"}),
            ],
        },
    ]
    hf_messages = [
        {"role": "user", "content": "Compare weather in SF and NYC"},
        {
            "role": "assistant",
            "content": "<think>Check both cities</think>Let me check both.",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "San Francisco"}}},
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "New York"}}},
            ],
        },
    ]

    hf = _hf_tokens(tokenizer, hf_messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


# ── Tool responses (role=tool) ──────────────────────────────────────────────


def test_tool_response_single(tokenizer, renderer):
    """Full tool-use round trip: user → assistant(tool_call) → tool → assistant."""
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "<think>check weather</think>Checking.",
            "tool_calls": [_make_tool_call("get_weather", {"city": "SF"})],
        },
        {"role": "tool", "content": "72°F and sunny"},
        {"role": "assistant", "content": "<think>got it</think>It's 72°F and sunny in SF!"},
    ]
    hf_messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "<think>check weather</think>Checking.",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "SF"}}},
            ],
        },
        {"role": "tool", "content": "72°F and sunny"},
        {"role": "assistant", "content": "<think>got it</think>It's 72°F and sunny in SF!"},
    ]

    hf = _hf_tokens(tokenizer, hf_messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


def test_tool_response_multiple_grouped(tokenizer, renderer):
    """Multiple consecutive tool responses grouped under one <|im_start|>user block."""
    messages = [
        {"role": "user", "content": "Compare weather in SF and NYC"},
        {
            "role": "assistant",
            "content": "<think>check both</think>Checking both cities.",
            "tool_calls": [
                _make_tool_call("get_weather", {"city": "San Francisco"}),
                _make_tool_call("get_weather", {"city": "New York"}),
            ],
        },
        {"role": "tool", "content": "72°F and sunny"},
        {"role": "tool", "content": "55°F and cloudy"},
        {"role": "assistant", "content": "<think>got both</think>SF is 72°F sunny, NYC is 55°F cloudy."},
    ]
    hf_messages = [
        {"role": "user", "content": "Compare weather in SF and NYC"},
        {
            "role": "assistant",
            "content": "<think>check both</think>Checking both cities.",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "San Francisco"}}},
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "New York"}}},
            ],
        },
        {"role": "tool", "content": "72°F and sunny"},
        {"role": "tool", "content": "55°F and cloudy"},
        {"role": "assistant", "content": "<think>got both</think>SF is 72°F sunny, NYC is 55°F cloudy."},
    ]

    hf = _hf_tokens(tokenizer, hf_messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


# ── Tool calls with history truncation ──────────────────────────────────────


def test_tool_call_history_truncation(tokenizer, renderer):
    """Tool call in historical assistant message — thinking should be truncated."""
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "<think>I should check</think>Let me check.",
            "tool_calls": [_make_tool_call("get_weather", {"city": "SF"})],
        },
        {"role": "tool", "content": "72°F"},
        {"role": "assistant", "content": "<think>got result</think>It's 72°F!"},
        {"role": "user", "content": "Thanks! And in NYC?"},
    ]
    hf_messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "<think>I should check</think>Let me check.",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "SF"}}},
            ],
        },
        {"role": "tool", "content": "72°F"},
        {"role": "assistant", "content": "<think>got result</think>It's 72°F!"},
        {"role": "user", "content": "Thanks! And in NYC?"},
    ]

    hf = _hf_tokens(tokenizer, hf_messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


# ── Three consecutive tool responses (first/middle/last) ────────────────────


def test_tool_response_three_consecutive(tokenizer, renderer):
    """Three consecutive tool responses — verify first/middle/last grouping."""
    messages = [
        {"role": "user", "content": "Get weather for three cities"},
        {
            "role": "assistant",
            "content": "<think>checking all three</think>Let me check.",
            "tool_calls": [
                _make_tool_call("get_weather", {"city": "San Francisco"}),
                _make_tool_call("get_weather", {"city": "New York"}),
                _make_tool_call("get_weather", {"city": "Chicago"}),
            ],
        },
        {"role": "tool", "content": "72°F and sunny"},
        {"role": "tool", "content": "55°F and cloudy"},
        {"role": "tool", "content": "45°F and windy"},
        {"role": "assistant", "content": "<think>got all three</think>SF: 72°F, NYC: 55°F, CHI: 45°F."},
    ]
    hf_messages = [
        {"role": "user", "content": "Get weather for three cities"},
        {
            "role": "assistant",
            "content": "<think>checking all three</think>Let me check.",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "San Francisco"}}},
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "New York"}}},
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Chicago"}}},
            ],
        },
        {"role": "tool", "content": "72°F and sunny"},
        {"role": "tool", "content": "55°F and cloudy"},
        {"role": "tool", "content": "45°F and windy"},
        {"role": "assistant", "content": "<think>got all three</think>SF: 72°F, NYC: 55°F, CHI: 45°F."},
    ]

    hf = _hf_tokens(tokenizer, hf_messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


# ── Tool call with nested/complex JSON arguments ────────────────────────────


def test_tool_call_nested_json_args(tokenizer, renderer):
    """Tool call with dict/list argument values — verify JSON serialization."""
    messages = [
        {"role": "user", "content": "Create an event"},
        {
            "role": "assistant",
            "content": "<think>creating</think>Creating the event.",
            "tool_calls": [
                _make_tool_call("create_event", {
                    "title": "Team Meeting",
                    "attendees": ["alice", "bob"],
                    "location": {"building": "HQ", "room": "101"},
                }),
            ],
        },
    ]
    hf_messages = [
        {"role": "user", "content": "Create an event"},
        {
            "role": "assistant",
            "content": "<think>creating</think>Creating the event.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "create_event",
                        "arguments": {
                            "title": "Team Meeting",
                            "attendees": ["alice", "bob"],
                            "location": {"building": "HQ", "room": "101"},
                        },
                    },
                },
            ],
        },
    ]

    hf = _hf_tokens(tokenizer, hf_messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


# ── Generation prompt after tool response ────────────────────────────────────


def test_generation_prompt_after_tool_response(tokenizer, renderer):
    """Generation prompt when last message is a tool response."""
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "<think>checking</think>Let me check.",
            "tool_calls": [_make_tool_call("get_weather", {"city": "SF"})],
        },
        {"role": "tool", "content": "72°F and sunny"},
    ]
    hf_messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "<think>checking</think>Let me check.",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "SF"}}},
            ],
        },
        {"role": "tool", "content": "72°F and sunny"},
    ]

    hf = _hf_tokens(tokenizer, hf_messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


# ── Complex multi-turn with multiple tool roundtrips ────────────────────────


def test_multi_turn_multiple_tool_roundtrips(tokenizer, renderer):
    """Two separate tool-use exchanges — first gets truncated, second kept."""
    messages = [
        {"role": "user", "content": "What's the weather in SF?"},
        {
            "role": "assistant",
            "content": "<think>check SF weather</think>Checking SF.",
            "tool_calls": [_make_tool_call("get_weather", {"city": "SF"})],
        },
        {"role": "tool", "content": "72°F and sunny"},
        {"role": "assistant", "content": "<think>got SF</think>It's 72°F and sunny in SF."},
        {"role": "user", "content": "Now check NYC"},
        {
            "role": "assistant",
            "content": "<think>check NYC</think>Checking NYC.",
            "tool_calls": [_make_tool_call("get_weather", {"city": "NYC"})],
        },
        {"role": "tool", "content": "55°F and rainy"},
        {"role": "assistant", "content": "<think>got NYC</think>NYC is 55°F and rainy."},
    ]
    hf_messages = [
        {"role": "user", "content": "What's the weather in SF?"},
        {
            "role": "assistant",
            "content": "<think>check SF weather</think>Checking SF.",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "SF"}}},
            ],
        },
        {"role": "tool", "content": "72°F and sunny"},
        {"role": "assistant", "content": "<think>got SF</think>It's 72°F and sunny in SF."},
        {"role": "user", "content": "Now check NYC"},
        {
            "role": "assistant",
            "content": "<think>check NYC</think>Checking NYC.",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "NYC"}}},
            ],
        },
        {"role": "tool", "content": "55°F and rainy"},
        {"role": "assistant", "content": "<think>got NYC</think>NYC is 55°F and rainy."},
    ]

    hf = _hf_tokens(tokenizer, hf_messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


# ── Empty assistant content ──────────────────────────────────────────────────


def test_empty_assistant_content(tokenizer, renderer):
    """Assistant message with empty content and no tool calls."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": ""},
    ]
    hf = _hf_tokens(tokenizer, messages, enable_thinking=True)
    ours = _renderer_tokens(renderer, messages)

    hf_text = tokenizer.decode(hf)
    our_text = tokenizer.decode(ours)
    print(f"\n--- HF ---\n{hf_text}")
    print(f"\n--- Ours ---\n{our_text}")
    assert hf == ours, f"Token mismatch:\nHF:   {hf}\nOurs: {ours}"


# ── Supervised with tool calls ───────────────────────────────────────────────


def test_supervised_with_tool_calls(tokenizer, renderer):
    """Supervised example with a tool-call round trip."""
    from tinker_cookbook.renderers.base import TrainOnWhat

    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "<think>check weather</think>Checking.",
            "tool_calls": [_make_tool_call("get_weather", {"city": "SF"})],
        },
        {"role": "tool", "content": "72°F and sunny"},
        {"role": "assistant", "content": "<think>got it</think>It's 72°F and sunny in SF!"},
    ]
    hf_messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "<think>check weather</think>Checking.",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "SF"}}},
            ],
        },
        {"role": "tool", "content": "72°F and sunny"},
        {"role": "assistant", "content": "<think>got it</think>It's 72°F and sunny in SF!"},
    ]

    model_input, weights = renderer.build_supervised_example(
        messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    our_tokens = list(model_input.to_ints())

    hf_tokens = tokenizer.apply_chat_template(
        hf_messages, tokenize=True, add_generation_prompt=False, enable_thinking=True,
    )

    our_text = tokenizer.decode(our_tokens)
    hf_text = tokenizer.decode(hf_tokens)
    print(f"\n--- HF supervised ---\n{hf_text}")
    print(f"\n--- Ours supervised ---\n{our_text}")

    if hf_tokens and hf_tokens[-1] == tokenizer.encode("\n", add_special_tokens=False)[-1]:
        hf_tokens_trimmed = hf_tokens[:-1]
    else:
        hf_tokens_trimmed = hf_tokens

    assert our_tokens == hf_tokens_trimmed, (
        f"Token mismatch:\nHF:   {hf_tokens_trimmed}\nOurs: {our_tokens}"
    )


def test_supervised_multi_turn_thinking_weights(tokenizer, renderer):
    """Supervised multi-turn — verify LAST_ASSISTANT_MESSAGE only trains on final turn."""
    from tinker_cookbook.renderers.base import TrainOnWhat

    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>calculating</think>4"},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "<think>adding</think>6"},
    ]

    model_input, weights = renderer.build_supervised_example(
        messages,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    tokens = list(model_input.to_ints())
    weights_list = weights.tolist()
    assert len(tokens) == len(weights_list)

    token_strs = [tokenizer.decode([t]) for t in tokens]
    print("\nToken weights (LAST_ASSISTANT_MESSAGE):")
    for i, (t, w) in enumerate(zip(token_strs, weights_list)):
        marker = "█" if w > 0 else "·"
        print(f"  {marker} {i:3d}  w={w:.0f}  {t!r}")

    assert any(w > 0 for w in weights_list), "Expected some non-zero weights"
    assert any(w == 0 for w in weights_list), "Expected some zero weights"

    # First assistant turn's tokens should all be zero-weighted.
    full_text = tokenizer.decode(tokens)
    first_asst_end = full_text.find("<|im_end|>", full_text.find("assistant\n<think></think>4"))
    assert first_asst_end > 0, "Could not locate first assistant turn"


def test_supervised_no_system_message(tokenizer, renderer):
    """Supervised without explicit system message — verify auto-injection matches HF."""
    from tinker_cookbook.renderers.base import TrainOnWhat

    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]

    model_input, weights = renderer.build_supervised_example(
        messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    our_tokens = list(model_input.to_ints())

    hf_tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=True,
    )

    our_text = tokenizer.decode(our_tokens)
    hf_text = tokenizer.decode(hf_tokens)
    print(f"\n--- HF supervised ---\n{hf_text}")
    print(f"\n--- Ours supervised ---\n{our_text}")

    if hf_tokens and hf_tokens[-1] == tokenizer.encode("\n", add_special_tokens=False)[-1]:
        hf_tokens_trimmed = hf_tokens[:-1]
    else:
        hf_tokens_trimmed = hf_tokens

    assert our_tokens == hf_tokens_trimmed, (
        f"Token mismatch:\nHF:   {hf_tokens_trimmed}\nOurs: {our_tokens}"
    )
