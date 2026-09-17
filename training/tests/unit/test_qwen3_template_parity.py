"""Tests for the `qwen3` renderer's HF-template parity fixes (APP-550).

Two train/inference mismatches reported by Uniphore (SFT job wl5urmii) and
verified against the stock Qwen/Qwen3-8B chat template:

1. Parallel tool results: the HF template merges consecutive tool results
   into ONE user turn with stacked ``<tool_response>`` blocks; the renderer
   used to emit one user turn per result. Inference serves the merged form.

2. Empty think wrapper: with thinking enabled, the HF template inserts
   ``<think>\\n\\n</think>\\n\\n`` before the FINAL assistant's answer when
   that turn carries no reasoning. Historical assistants stay bare. The
   wrapper belongs to the trainable output (at inference the model generates
   it; the generation prompt ends at ``<|im_start|>assistant\\n``), unlike
   the disable-thinking path where the whole wrapper is prefilled and masked.

3. Reasoning after the last query: the HF template strips thinking only
   from assistants at or before the last real user query; agentic tool-call
   turns after it keep their reasoning, rendered as
   ``<think>\\n...\\n</think>\\n\\n``. The renderer used to strip thinking
   from every non-final assistant and rendered kept thinking without the
   newlines.

Tests skip cleanly if HF Hub is unreachable.
"""

from __future__ import annotations

import pytest

_TOKENIZER_MODEL = "Qwen/Qwen3-8B"


def _load_tokenizer():
    """Try to load the Qwen3-8B tokenizer; skip cleanly if unreachable."""
    try:
        from training.renderer.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(_TOKENIZER_MODEL)
    except (OSError, ValueError, RuntimeError) as exc:
        pytest.skip(f"tokenizer unavailable for {_TOKENIZER_MODEL!r}: {exc}")
    if not getattr(tokenizer, "chat_template", None):
        pytest.skip(f"{_TOKENIZER_MODEL!r} has no chat_template")
    return tokenizer


def _renderer(tokenizer):
    import training.renderer  # noqa: F401  (registration side-effect)
    from training.renderer import get_renderer

    return get_renderer("qwen3", tokenizer)


def _supervised_tokens(renderer, messages):
    model_input, _ = renderer.build_supervised_example(messages)
    return [int(t) for t in model_input.to_ints()]


def _hf_tokens(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # The HF template emits a trailing "\n" after the final turn; nothing is
    # trained on it and the property harness forgives it, so do the same.
    if text.endswith("\n"):
        text = text[:-1]
    return tokenizer.encode(text, add_special_tokens=False)


_PARALLEL_TOOLS_HF = [
    {"role": "user", "content": "check my orders"},
    {
        "role": "assistant",
        "content": "on it",
        "tool_calls": [
            {
                "type": "function",
                "id": "c1",
                "function": {"name": "get_order", "arguments": '{"id": 1}'},
            },
            {
                "type": "function",
                "id": "c2",
                "function": {"name": "get_order", "arguments": '{"id": 2}'},
            },
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "order 1 ok"},
    {"role": "tool", "tool_call_id": "c2", "content": "order 2 ok"},
    {"role": "assistant", "content": "two orders"},
]

_NO_REASONING_LAST = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello there"},
]

_NO_REASONING_HIST = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello there"},
    {"role": "user", "content": "how are you"},
    {"role": "assistant", "content": "great"},
]

_FINAL_WITH_REASONING = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "reasoning_content": "hmm", "content": "hello"},
]

# Agentic turn: tool-calling assistant AFTER the last user query carries
# reasoning. HF keeps it; the old renderer stripped it from every non-final
# assistant.
_AGENTIC_KEEP_REASONING = [
    {"role": "user", "content": "check order"},
    {
        "role": "assistant",
        "reasoning_content": "need to look it up",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "id": "c1",
                "function": {"name": "get_order", "arguments": '{"id": 1}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    {"role": "assistant", "reasoning_content": "got it", "content": "order shipped"},
]

# Same but with parallel tool calls: fold + keep-reasoning interact.
_AGENTIC_PARALLEL_REASONING = [
    {"role": "user", "content": "check my orders"},
    {
        "role": "assistant",
        "reasoning_content": "parallel lookups",
        "content": "on it",
        "tool_calls": [
            {
                "type": "function",
                "id": "c1",
                "function": {"name": "get_order", "arguments": '{"id": 1}'},
            },
            {
                "type": "function",
                "id": "c2",
                "function": {"name": "get_order", "arguments": '{"id": 2}'},
            },
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "o1"},
    {"role": "tool", "tool_call_id": "c2", "content": "o2"},
    {"role": "assistant", "content": "two orders"},
]

# Reasoning on an assistant BEFORE the last user query is stripped on both
# sides (HF drops it, renderer drops it).
_STRIP_BEFORE_LAST_QUERY = [
    {"role": "user", "content": "q1"},
    {"role": "assistant", "reasoning_content": "old thinking", "content": "a1"},
    {"role": "user", "content": "q2"},
    {"role": "assistant", "content": "a2"},
]

# Final assistant with reasoning and tool calls but no visible text.
_FINAL_REASONING_TOOL_CALLS = [
    {"role": "user", "content": "go"},
    {
        "role": "assistant",
        "reasoning_content": "plan",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "id": "c1",
                "function": {"name": "f", "arguments": "{}"},
            }
        ],
    },
]


def _tool_call_objects(messages):
    """Convert HF-style dict tool_calls to the renderer's ToolCall objects."""
    from training.renderer import ToolCall

    out = []
    for m in messages:
        m = dict(m)
        if "tool_calls" in m:
            m["tool_calls"] = [
                ToolCall(
                    type="function",
                    id=tc["id"],
                    function=ToolCall.FunctionBody(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in m["tool_calls"]
            ]
        out.append(m)
    return out


@pytest.mark.timeout(180)
def test_parallel_tool_results_merge_matches_hf() -> None:
    tokenizer = _load_tokenizer()
    ours = _supervised_tokens(
        _renderer(tokenizer), _tool_call_objects(_PARALLEL_TOOLS_HF)
    )
    hf = _hf_tokens(tokenizer, _PARALLEL_TOOLS_HF)
    assert ours == hf


@pytest.mark.timeout(180)
def test_final_non_reasoning_assistant_gets_empty_think_wrapper() -> None:
    tokenizer = _load_tokenizer()
    ours = _supervised_tokens(_renderer(tokenizer), _NO_REASONING_LAST)
    hf = _hf_tokens(tokenizer, _NO_REASONING_LAST)
    assert ours == hf


@pytest.mark.timeout(180)
def test_historical_non_reasoning_assistant_stays_bare() -> None:
    tokenizer = _load_tokenizer()
    ours = _supervised_tokens(_renderer(tokenizer), _NO_REASONING_HIST)
    hf = _hf_tokens(tokenizer, _NO_REASONING_HIST)
    assert ours == hf


@pytest.mark.timeout(180)
def test_empty_think_wrapper_is_trainable_and_header_masked() -> None:
    tokenizer = _load_tokenizer()
    renderer = _renderer(tokenizer)
    model_input, weights = renderer.build_supervised_example(_NO_REASONING_LAST)
    tokens = [int(t) for t in model_input.to_ints()]
    weights = [int(w) for w in weights.tolist()]

    wrapper = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    pos = text.index("<think>\n\n</think>")
    wrapper_start = len(tokenizer.encode(text[:pos], add_special_tokens=False))
    assert weights[wrapper_start : wrapper_start + len(wrapper)] == [1] * len(wrapper)

    header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    assert weights[: len(header)] == [0] * len(header)


@pytest.mark.timeout(180)
def test_final_assistant_with_reasoning_gets_no_empty_wrapper() -> None:
    tokenizer = _load_tokenizer()
    renderer = _renderer(tokenizer)
    from training.utils.supervised import normalize_messages

    tokens = _supervised_tokens(renderer, normalize_messages(_FINAL_WITH_REASONING))
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    assert "<think>\n\n</think>" not in text
    assert "<think>" in text  # real reasoning block present


@pytest.mark.timeout(180)
def test_non_consecutive_tool_results_untouched() -> None:
    """A single tool result (nothing to fold) must render exactly as before."""
    tokenizer = _load_tokenizer()
    renderer = _renderer(tokenizer)
    messages = [
        {"role": "user", "content": "check order"},
        {
            "role": "assistant",
            "content": "checking now",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "c1",
                    "function": {"name": "get_order", "arguments": '{"id": 1}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    ours = _supervised_tokens(renderer, _tool_call_objects(messages))
    hf = _hf_tokens(tokenizer, messages)
    assert ours == hf


def _normalized_supervised_tokens(renderer, messages):
    from training.utils.supervised import normalize_messages

    return _supervised_tokens(renderer, normalize_messages(messages))


@pytest.mark.timeout(180)
def test_agentic_reasoning_kept_after_last_query() -> None:
    tokenizer = _load_tokenizer()
    ours = _normalized_supervised_tokens(_renderer(tokenizer), _AGENTIC_KEEP_REASONING)
    assert ours == _hf_tokens(tokenizer, _AGENTIC_KEEP_REASONING)


@pytest.mark.timeout(180)
def test_agentic_parallel_tools_with_reasoning() -> None:
    tokenizer = _load_tokenizer()
    ours = _normalized_supervised_tokens(
        _renderer(tokenizer), _AGENTIC_PARALLEL_REASONING
    )
    assert ours == _hf_tokens(tokenizer, _AGENTIC_PARALLEL_REASONING)


@pytest.mark.timeout(180)
def test_reasoning_before_last_query_stripped() -> None:
    tokenizer = _load_tokenizer()
    ours = _normalized_supervised_tokens(_renderer(tokenizer), _STRIP_BEFORE_LAST_QUERY)
    assert ours == _hf_tokens(tokenizer, _STRIP_BEFORE_LAST_QUERY)


@pytest.mark.timeout(180)
def test_final_reasoning_with_tool_calls_no_text() -> None:
    tokenizer = _load_tokenizer()
    ours = _normalized_supervised_tokens(
        _renderer(tokenizer), _FINAL_REASONING_TOOL_CALLS
    )
    assert ours == _hf_tokens(tokenizer, _FINAL_REASONING_TOOL_CALLS)


_FINAL_NO_REASONING_TOOL_CALLS = [
    {"role": "user", "content": "go"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "id": "c1",
                "function": {"name": "f", "arguments": "{}"},
            }
        ],
    },
]


@pytest.mark.timeout(180)
def test_final_tool_only_turn_wrapper_then_call_no_extra_newline() -> None:
    """HF emits `<think>\\n\\n</think>\\n\\n<tool_call>`; upstream's
    unconditional tool-call separator must not add a third newline."""
    tokenizer = _load_tokenizer()
    ours = _normalized_supervised_tokens(
        _renderer(tokenizer), _FINAL_NO_REASONING_TOOL_CALLS
    )
    assert ours == _hf_tokens(tokenizer, _FINAL_NO_REASONING_TOOL_CALLS)


@pytest.mark.timeout(180)
def test_fold_skipped_when_trainable_flags_differ() -> None:
    """Folding keeps only the first result's metadata, so messages with
    different `trainable` flags must stay separate (CUSTOMIZED weights)."""
    from training._vendor.tinker_cookbook_0_4_3.renderers.base import TrainOnWhat
    from training.utils.supervised import normalize_messages

    tokenizer = _load_tokenizer()
    renderer = _renderer(tokenizer)
    messages = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "c1",
                    "function": {"name": "f", "arguments": "{}"},
                },
                {
                    "type": "function",
                    "id": "c2",
                    "function": {"name": "f", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "FIRST", "trainable": False},
        {"role": "tool", "tool_call_id": "c2", "content": "SECOND", "trainable": True},
    ]
    model_input, weights = renderer.build_supervised_example(
        normalize_messages(messages), train_on_what=TrainOnWhat.CUSTOMIZED
    )
    tokens = [int(t) for t in model_input.to_ints()]
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    assert text.count("<|im_start|>user\n<tool_response>") == 2  # not folded

    weights = [int(w) for w in weights.tolist()]
    second_pos = text.index("SECOND")
    second_tok = len(tokenizer.encode(text[:second_pos], add_special_tokens=False))
    second_end = len(tokenizer.encode(text[: second_pos + 6], add_special_tokens=False))
    assert all(w == 1 for w in weights[second_tok:second_end])
    first_pos = text.index("FIRST")
    first_tok = len(tokenizer.encode(text[:first_pos], add_special_tokens=False))
    first_end = len(tokenizer.encode(text[: first_pos + 5], add_special_tokens=False))
    assert all(w == 0 for w in weights[first_tok:first_end])


_FINAL_LEADING_NEWLINE_ANSWER = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "\nhello"},
]


@pytest.mark.timeout(180)
def test_final_answer_leading_newline_matches_hf() -> None:
    """HF strips leading newlines from the answer after the empty wrapper."""
    tokenizer = _load_tokenizer()
    ours = _normalized_supervised_tokens(
        _renderer(tokenizer), _FINAL_LEADING_NEWLINE_ANSWER
    )
    assert ours == _hf_tokens(tokenizer, _FINAL_LEADING_NEWLINE_ANSWER)


_ROUNDTRIP_REASONING = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "reasoning_content": "plan", "content": "\n\nhello"},
]


@pytest.mark.timeout(180)
def test_reused_response_whitespace_not_doubled() -> None:
    """A text part that already starts with `\\n\\n` (parse_response round
    trip) must not get a second pair."""
    tokenizer = _load_tokenizer()
    ours = _normalized_supervised_tokens(_renderer(tokenizer), _ROUNDTRIP_REASONING)
    assert ours == _hf_tokens(tokenizer, _ROUNDTRIP_REASONING)


_TOOL_ONLY_STRUCTURED_EMPTY = [
    {"role": "user", "content": "go"},
    {
        "role": "assistant",
        "content": [{"type": "text", "text": ""}],
        "tool_calls": [
            {
                "type": "function",
                "id": "c1",
                "function": {"name": "f", "arguments": "{}"},
            }
        ],
    },
]


@pytest.mark.timeout(180)
def test_tool_only_structured_empty_content_matches_hf() -> None:
    """Structured empty content (parse_response shape) must not keep the
    extra tool-call newline."""
    tokenizer = _load_tokenizer()
    ours = _normalized_supervised_tokens(
        _renderer(tokenizer), _TOOL_ONLY_STRUCTURED_EMPTY
    )
    hf = _hf_tokens(
        tokenizer,
        [
            _TOOL_ONLY_STRUCTURED_EMPTY[0],
            {**_TOOL_ONLY_STRUCTURED_EMPTY[1], "content": ""},
        ],
    )
    assert ours == hf


_REASONING_TOOL_CALL_GAP = [
    {"role": "user", "content": "go"},
    {
        "role": "assistant",
        "reasoning_content": "plan",
        "content": "\n",
        "tool_calls": [
            {
                "type": "function",
                "id": "c1",
                "function": {"name": "f", "arguments": "{}"},
            }
        ],
    },
]


@pytest.mark.timeout(180)
def test_reasoning_tool_call_newline_gap_matches_hf() -> None:
    """A newline-only text part between `</think>` and the tool call (the
    parse_response round-trip shape) is the tool-call gap, not content."""
    tokenizer = _load_tokenizer()
    ours = _normalized_supervised_tokens(_renderer(tokenizer), _REASONING_TOOL_CALL_GAP)
    assert ours == _hf_tokens(tokenizer, _REASONING_TOOL_CALL_GAP)


_FINAL_NEWLINE_ONLY_TOOL_CALLS = [
    {"role": "user", "content": "go"},
    {
        "role": "assistant",
        "content": "\n",
        "tool_calls": [
            {
                "type": "function",
                "id": "c1",
                "function": {"name": "f", "arguments": "{}"},
            }
        ],
    },
]


@pytest.mark.timeout(180)
def test_final_newline_only_tool_call_keeps_separator() -> None:
    """HF decides the tool-call separator on the original content: a
    newline-only answer keeps it (three newlines before the call)."""
    tokenizer = _load_tokenizer()
    ours = _normalized_supervised_tokens(
        _renderer(tokenizer), _FINAL_NEWLINE_ONLY_TOOL_CALLS
    )
    assert ours == _hf_tokens(tokenizer, _FINAL_NEWLINE_ONLY_TOOL_CALLS)


@pytest.mark.timeout(180)
def test_multipart_visible_content_keeps_tool_call_separator() -> None:
    """Multipart content with visible text keeps the tool-call separator,
    same as the string-content path (HF has no multipart semantics)."""
    tokenizer = _load_tokenizer()
    r = _renderer(tokenizer)
    tc = {
        "type": "function",
        "id": "c1",
        "function": {"name": "f", "arguments": "{}"},
    }
    structured = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "tool_calls": [tc],
        },
    ]
    plain = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "done", "tool_calls": [tc]},
    ]
    assert _normalized_supervised_tokens(r, structured) == _normalized_supervised_tokens(
        _renderer(tokenizer), plain
    )


@pytest.mark.timeout(180)
def test_preserved_history_mode_untouched_and_prefix_stable() -> None:
    """strip_thinking_from_history=False has no HF counterpart; the wrapper
    and reformat must stay off so the extension property keeps holding."""
    from training.utils.supervised import normalize_messages

    tokenizer = _load_tokenizer()
    renderer = _renderer(tokenizer)
    renderer.strip_thinking_from_history = False
    short = normalize_messages(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    )
    long = normalize_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "yo"},
        ]
    )
    a, _ = renderer.build_supervised_example(short)
    b, _ = renderer.build_supervised_example(long)
    a_ids = [int(t) for t in a.to_ints()]
    b_ids = [int(t) for t in b.to_ints()]
    assert "<think>" not in tokenizer.decode(a_ids, skip_special_tokens=False)
    assert b_ids[: len(a_ids)] == a_ids
