"""Tests for `Qwen3_6PreserveThinkingSplitRenderer` (interleave thinking).

Two complementary tests:

1. *Behavioral*: with assistant turns that carry actual thinking content,
   the default `qwen3_6` renderer must DROP historical thinking
   (matches HF `apply_chat_template` default), and the preserve-thinking
   variant must KEEP it. Renderer-vs-renderer comparison; doesn't need
   HF Hub access.

2. *HF parity*: the preserve-thinking renderer's tokens must equal what
   `apply_chat_template(messages, preserve_thinking=True)` produces
   for the same conversation. Tests two input shapes that mean the same
   thing semantically:
     - renderer side uses structured `content` with
       `{"type": "thinking", "thinking": ...}` parts.
     - HF side uses a string `content` plus the `reasoning_content`
       string field that the Qwen3.6 chat template inspects.

Both tests skip cleanly if HF Hub is unreachable.
"""

from __future__ import annotations

import pytest


_TOKENIZER_MODEL = "Qwen/Qwen3.6-27B"


# --- Multi-turn fixture with REAL thinking content -------------------------
# The historical assistant turn (index=2) carries reasoning the model used
# to derive "4.". The new user turn (index=3) is the most recent user.
# Without preserve-thinking, the chat template / renderer should DROP the
# historical reasoning. With preserve-thinking, it should be retained.

_HISTORICAL_REASONING = "Two plus two: count up by ones from 2, get 3, then 4."

_MESSAGES_RENDERER_INPUT = [
    {"role": "system", "content": "Answer briefly with just the number."},
    {"role": "user", "content": "What is 2+2?"},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": _HISTORICAL_REASONING},
            {"type": "text", "text": "4."},
        ],
    },
    {"role": "user", "content": "And 3+3?"},
]

# Same conversation for the HF chat template, in the shape the Qwen3.6
# chat template understands: `reasoning_content` field on assistant +
# string `content`. The chat template's `elif message.role ==
# "assistant"` branch reads `message.reasoning_content` directly when
# it's a string.
_MESSAGES_HF_INPUT = [
    {"role": "system", "content": "Answer briefly with just the number."},
    {"role": "user", "content": "What is 2+2?"},
    {
        "role": "assistant",
        "content": "4.",
        "reasoning_content": _HISTORICAL_REASONING,
    },
    {"role": "user", "content": "And 3+3?"},
]


# --- Realistic 3-turn fixture: code-review agent ------------------------
# Two historical assistant turns each carry distinct reasoning. In
# preserve-thinking mode the model writing turn 3 sees BOTH prior
# reasoning traces; in default mode it sees neither. This fixture is
# what an actual long-horizon agent SFT dataset looks like — multi-turn
# with reasoning that the next turn's answer should build on.

_REASONING_TURN_1 = (
    "Two nested loops, each iterating arr once. Total iterations "
    "= len(arr) * len(arr) = n^2. Inner condition is O(1) so outer dominates."
)
_REASONING_TURN_2 = (
    "Use a hash set to record seen values. For each x, check target-x "
    "in O(1). Single pass = O(n) time, O(n) extra space."
)
_VISIBLE_TURN_1 = (
    "Two nested loops over arr each give n iterations, multiplied = O(n^2)."
)
_VISIBLE_TURN_2 = (
    "Iterate once with a set: for each `a`, check if `target - a` is in "
    "the set; otherwise add `a` to the set. O(n) time, O(n) space."
)

_CODE_REVIEW_RENDERER_INPUT = [
    {
        "role": "system",
        "content": "You're a senior engineer reviewing code. Reason step by step.",
    },
    {
        "role": "user",
        "content": (
            "Why is this O(n^2)?\n"
            "```python\n"
            "for a in arr:\n"
            "    for b in arr:\n"
            "        if a + b == target: return True\n"
            "```"
        ),
    },
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": _REASONING_TURN_1},
            {"type": "text", "text": _VISIBLE_TURN_1},
        ],
    },
    {"role": "user", "content": "How do I get O(n)?"},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": _REASONING_TURN_2},
            {"type": "text", "text": _VISIBLE_TURN_2},
        ],
    },
    {"role": "user", "content": "Show me the code."},
]

_CODE_REVIEW_HF_INPUT = [
    {
        "role": "system",
        "content": "You're a senior engineer reviewing code. Reason step by step.",
    },
    {
        "role": "user",
        "content": (
            "Why is this O(n^2)?\n"
            "```python\n"
            "for a in arr:\n"
            "    for b in arr:\n"
            "        if a + b == target: return True\n"
            "```"
        ),
    },
    {
        "role": "assistant",
        "content": _VISIBLE_TURN_1,
        "reasoning_content": _REASONING_TURN_1,
    },
    {"role": "user", "content": "How do I get O(n)?"},
    {
        "role": "assistant",
        "content": _VISIBLE_TURN_2,
        "reasoning_content": _REASONING_TURN_2,
    },
    {"role": "user", "content": "Show me the code."},
]


def _load_tokenizer():
    """Try to load the Qwen3.6-27B tokenizer; skip cleanly if unreachable."""
    try:
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        tokenizer = get_tokenizer(_TOKENIZER_MODEL)
    except (OSError, ValueError, RuntimeError) as exc:
        pytest.skip(f"tokenizer unavailable for {_TOKENIZER_MODEL!r}: {exc}")
    if not getattr(tokenizer, "chat_template", None):
        pytest.skip(f"{_TOKENIZER_MODEL!r} has no chat_template")
    return tokenizer


def _decode(tokenizer, tokens: list[int]) -> str:
    """Decode tokens to a single string (special tokens kept for visibility)."""
    return tokenizer.decode(tokens, skip_special_tokens=False)


def _build_renderer_tokens(renderer_name: str, tokenizer) -> list[int]:
    """Render the multi-turn fixture via the named renderer."""
    # Importing the cookbook renderer package registers the local renderer
    # names (qwen3_6, qwen3_6_preserve_thinking, ...) in tinker_cookbook's
    # registry.
    import training.renderer  # noqa: F401  (registration side-effect)
    from tinker_cookbook.renderers import get_renderer

    from training.utils.supervised import normalize_messages

    renderer = get_renderer(renderer_name, tokenizer)
    normalized = normalize_messages(_MESSAGES_RENDERER_INPUT)
    model_input = renderer.build_generation_prompt(normalized, role="assistant")
    return [int(t) for t in model_input.to_ints()]


# --------------------------------------------------------------------------
# Test 1 — Behavioral: kwarg flips historical-thinking presence
# --------------------------------------------------------------------------


@pytest.mark.timeout(180)
def test_default_renderer_drops_historical_thinking() -> None:
    """`qwen3_6` (alias of qwen3_5, strip_thinking_from_history=True)
    must drop the reasoning from historical assistant turns.

    The historical reasoning string must NOT appear in the rendered
    output, mirroring what HF `apply_chat_template` does by default.
    """
    tokenizer = _load_tokenizer()
    tokens = _build_renderer_tokens("qwen3_6", tokenizer)
    decoded = _decode(tokenizer, tokens)

    assert _HISTORICAL_REASONING not in decoded, (
        "Default qwen3_6 renderer should strip historical thinking, but the "
        f"reasoning string was present in the rendered output:\n{decoded!r}"
    )


@pytest.mark.timeout(180)
def test_preserve_thinking_renderer_keeps_historical_thinking() -> None:
    """`qwen3_6_preserve_thinking` (strip_thinking_from_history=False)
    must retain the reasoning from historical assistant turns.

    The historical reasoning string MUST appear in the rendered output,
    matching HF `apply_chat_template(messages, preserve_thinking=True)`.
    """
    tokenizer = _load_tokenizer()
    tokens = _build_renderer_tokens("qwen3_6_preserve_thinking", tokenizer)
    decoded = _decode(tokenizer, tokens)

    assert _HISTORICAL_REASONING in decoded, (
        "Preserve-thinking renderer should retain historical thinking, but "
        f"the reasoning string was missing from the rendered output:\n{decoded!r}"
    )


@pytest.mark.timeout(180)
def test_default_and_preserve_produce_different_tokens() -> None:
    """The two variants MUST produce different token sequences when there is
    real thinking content in history. (If they're identical, the kwarg is
    not taking effect.)
    """
    tokenizer = _load_tokenizer()
    default_tokens = _build_renderer_tokens("qwen3_6", tokenizer)
    preserve_tokens = _build_renderer_tokens("qwen3_6_preserve_thinking", tokenizer)

    assert default_tokens != preserve_tokens, (
        "qwen3_6 (default) and qwen3_6_preserve_thinking produced identical "
        "tokens — strip_thinking_from_history kwarg is not taking effect "
        "or the test fixture lacks historical thinking content."
    )


@pytest.mark.timeout(180)
def test_preserve_thinking_renderer_has_extension_property() -> None:
    """The preserve-thinking renderer must report
    `has_extension_property=True` so that the SFT loop's
    `_build_renderer_supervised_examples` takes the single-example path
    (instead of the disaggregate per-user-turn split)."""
    tokenizer = _load_tokenizer()

    import training.renderer  # noqa: F401
    from tinker_cookbook.renderers import get_renderer

    default_renderer = get_renderer("qwen3_6", tokenizer)
    preserve_renderer = get_renderer("qwen3_6_preserve_thinking", tokenizer)

    assert default_renderer.has_extension_property is False, (
        "Default qwen3_6 renderer should NOT have extension property "
        "(strip_thinking_from_history=True breaks prefix extension)."
    )
    assert preserve_renderer.has_extension_property is True, (
        "qwen3_6_preserve_thinking renderer SHOULD have extension property "
        "(strip_thinking_from_history=False keeps history-prefix tokenization)."
    )


# --------------------------------------------------------------------------
# Test 2 — HF parity: preserve renderer matches preserve_thinking template
# --------------------------------------------------------------------------


@pytest.mark.timeout(180)
def test_preserve_thinking_matches_hf_chat_template_byte_for_byte() -> None:
    """The preserve-thinking renderer's tokens must equal what
    `tokenizer.apply_chat_template(messages, preserve_thinking=True)`
    produces for the same conversation.

    Note: the renderer-side input (`_MESSAGES_RENDERER_INPUT`) uses
    structured content with a `{"type": "thinking", ...}` part, while
    the HF-side input (`_MESSAGES_HF_INPUT`) uses string `content`
    plus `reasoning_content` field — the two shapes the two systems
    accept. Semantically they represent the same conversation.
    """
    tokenizer = _load_tokenizer()

    # Renderer side
    renderer_tokens = _build_renderer_tokens(
        "qwen3_6_preserve_thinking", tokenizer
    )

    # HF side
    hf_result = tokenizer.apply_chat_template(
        _MESSAGES_HF_INPUT,
        tokenize=True,
        add_generation_prompt=True,
        preserve_thinking=True,
    )
    if hasattr(hf_result, "input_ids"):
        hf_tokens = [int(t) for t in list(hf_result.input_ids)]  # type: ignore[arg-type]
    else:
        hf_tokens = [int(t) for t in list(hf_result)]  # type: ignore[arg-type]

    if renderer_tokens == hf_tokens:
        return

    # Build a useful failure message showing the first divergence.
    n = min(len(renderer_tokens), len(hf_tokens))
    first_div = next(
        (i for i in range(n) if renderer_tokens[i] != hf_tokens[i]),
        n,
    )
    context = 6
    lo = max(0, first_div - context)
    hi = first_div + context + 1
    lines = [
        f"renderer tokens={len(renderer_tokens)}  hf tokens={len(hf_tokens)}  "
        f"first divergence at idx={first_div}",
        f"renderer[{lo}:{hi}] decoded:",
    ]
    for i in range(lo, min(hi, len(renderer_tokens))):
        marker = "→" if i == first_div else " "
        lines.append(
            f"  {marker} idx={i}  tok={renderer_tokens[i]}  "
            f"decoded={tokenizer.decode([renderer_tokens[i]], skip_special_tokens=False)!r}"
        )
    lines.append(f"hf[{lo}:{hi}] decoded:")
    for i in range(lo, min(hi, len(hf_tokens))):
        marker = "→" if i == first_div else " "
        lines.append(
            f"  {marker} idx={i}  tok={hf_tokens[i]}  "
            f"decoded={tokenizer.decode([hf_tokens[i]], skip_special_tokens=False)!r}"
        )
    pytest.fail("\n".join(lines))


# --------------------------------------------------------------------------
# Test 3 — Realistic 3-turn code-review fixture: validates BOTH historical
# turns' thinking is correctly handled (default drops both, preserve keeps
# both).
# --------------------------------------------------------------------------


def _build_renderer_tokens_for(messages, renderer_name: str, tokenizer) -> list[int]:
    """Render a custom message list via the named renderer."""
    import training.renderer  # noqa: F401  (registration side-effect)
    from tinker_cookbook.renderers import get_renderer

    from training.utils.supervised import normalize_messages

    renderer = get_renderer(renderer_name, tokenizer)
    normalized = normalize_messages(messages)
    model_input = renderer.build_generation_prompt(normalized, role="assistant")
    return [int(t) for t in model_input.to_ints()]


@pytest.mark.timeout(180)
def test_code_review_default_drops_both_historical_thinkings() -> None:
    """3-turn code-review fixture, default qwen3_6 renderer.

    Both historical assistant turns carry distinct reasoning. In default
    (strip-from-history) mode, the rendered output must contain NEITHER
    reasoning string — both turns' <think> blocks should be dropped.
    """
    tokenizer = _load_tokenizer()
    tokens = _build_renderer_tokens_for(
        _CODE_REVIEW_RENDERER_INPUT, "qwen3_6", tokenizer
    )
    decoded = _decode(tokenizer, tokens)

    assert _REASONING_TURN_1 not in decoded, (
        f"qwen3_6 (default) leaked turn-1 reasoning into rendered output:\n"
        f"  reasoning_1={_REASONING_TURN_1!r}\n  decoded={decoded[:500]!r}..."
    )
    assert _REASONING_TURN_2 not in decoded, (
        f"qwen3_6 (default) leaked turn-2 reasoning into rendered output:\n"
        f"  reasoning_2={_REASONING_TURN_2!r}\n  decoded={decoded[:500]!r}..."
    )
    # Sanity: visible content from BOTH turns must be present (only thinking
    # is stripped, not the visible answer).
    assert _VISIBLE_TURN_1 in decoded, (
        "Default renderer should keep historical visible content; turn-1 visible missing."
    )
    assert _VISIBLE_TURN_2 in decoded, (
        "Default renderer should keep historical visible content; turn-2 visible missing."
    )


@pytest.mark.timeout(180)
def test_code_review_preserve_keeps_both_historical_thinkings() -> None:
    """3-turn code-review fixture, qwen3_6_preserve_thinking renderer.

    Both historical assistant turns must retain their <think> blocks in
    the rendered prefix that the model writing turn 3 will see.
    """
    tokenizer = _load_tokenizer()
    tokens = _build_renderer_tokens_for(
        _CODE_REVIEW_RENDERER_INPUT, "qwen3_6_preserve_thinking", tokenizer
    )
    decoded = _decode(tokenizer, tokens)

    assert _REASONING_TURN_1 in decoded, (
        f"qwen3_6_preserve_thinking dropped turn-1 reasoning:\n"
        f"  reasoning_1={_REASONING_TURN_1!r}\n  decoded={decoded[:500]!r}..."
    )
    assert _REASONING_TURN_2 in decoded, (
        f"qwen3_6_preserve_thinking dropped turn-2 reasoning:\n"
        f"  reasoning_2={_REASONING_TURN_2!r}\n  decoded={decoded[:500]!r}..."
    )
    # Visible content also present (sanity).
    assert _VISIBLE_TURN_1 in decoded
    assert _VISIBLE_TURN_2 in decoded


@pytest.mark.timeout(180)
def test_code_review_preserve_matches_hf_chat_template() -> None:
    """3-turn code-review fixture, byte-for-byte HF chat-template parity
    for the preserve-thinking variant.

    Renderer-side input uses structured `content` parts; HF-side input
    uses string `content` plus `reasoning_content` field. Same
    semantic conversation. With `preserve_thinking=True`, the chat
    template emits `<think>{reasoning_content}</think>` for each
    historical assistant turn — this must match what the renderer
    produces from structured thinking parts.

    Adversarial fan-out tests below catch additional failure modes:
    double-wrapping, empty-block spam, and N-way HF parity across all
    historical turns.
    """
    tokenizer = _load_tokenizer()

    renderer_tokens = _build_renderer_tokens_for(
        _CODE_REVIEW_RENDERER_INPUT, "qwen3_6_preserve_thinking", tokenizer
    )

    hf_result = tokenizer.apply_chat_template(
        _CODE_REVIEW_HF_INPUT,
        tokenize=True,
        add_generation_prompt=True,
        preserve_thinking=True,
    )
    if hasattr(hf_result, "input_ids"):
        hf_tokens = [int(t) for t in list(hf_result.input_ids)]  # type: ignore[arg-type]
    else:
        hf_tokens = [int(t) for t in list(hf_result)]  # type: ignore[arg-type]

    if renderer_tokens == hf_tokens:
        return

    n = min(len(renderer_tokens), len(hf_tokens))
    first_div = next(
        (i for i in range(n) if renderer_tokens[i] != hf_tokens[i]), n
    )
    context = 6
    lo = max(0, first_div - context)
    hi = first_div + context + 1
    lines = [
        f"renderer tokens={len(renderer_tokens)}  hf tokens={len(hf_tokens)}  "
        f"first divergence at idx={first_div}",
        f"renderer[{lo}:{hi}] decoded:",
    ]
    for i in range(lo, min(hi, len(renderer_tokens))):
        marker = "→" if i == first_div else " "
        lines.append(
            f"  {marker} idx={i}  tok={renderer_tokens[i]}  "
            f"decoded={tokenizer.decode([renderer_tokens[i]], skip_special_tokens=False)!r}"
        )
    lines.append(f"hf[{lo}:{hi}] decoded:")
    for i in range(lo, min(hi, len(hf_tokens))):
        marker = "→" if i == first_div else " "
        lines.append(
            f"  {marker} idx={i}  tok={hf_tokens[i]}  "
            f"decoded={tokenizer.decode([hf_tokens[i]], skip_special_tokens=False)!r}"
        )
    pytest.fail("\n".join(lines))


# --------------------------------------------------------------------------
# Test 4 — Adversarial fan-out: catches well-known multi-turn
# preserve-thinking failure modes documented by the
# froggeric/Qwen-Fixed-Chat-Templates community fork
# (https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates):
#   - double-wrapping (thinking tags nest across turns)
#   - empty-block spam (assistants without reasoning emit empty <think></think>)
#   - history truncation (a thinking string vanishes after N turns)
# Even if our renderer inherits any upstream bug, this regression gate
# detects it.
# --------------------------------------------------------------------------

# Distinct, recognizable reasoning strings for each historical turn so a
# count assertion can verify each appears exactly once.
_REASONING_T1 = "TURN_1_THINK: parse the question carefully before computing."
_REASONING_T2 = "TURN_2_THINK: previous answer used method A; now apply method B for symmetry."
_REASONING_T3 = "TURN_3_THINK: combine results from method A and B and check edge cases."
_VISIBLE_T1 = "Answer for turn 1."
_VISIBLE_T2 = "Answer for turn 2."
_VISIBLE_T3 = "Answer for turn 3."


def _build_fanout_renderer_input(thinking_per_turn: list) -> list[dict]:
    """Build a multi-turn renderer-input fixture with N assistant turns.

    `thinking_per_turn[i]` is the reasoning for assistant turn `i+1`,
    or `None` for an assistant turn with no thinking. A trailing user
    message is added so the next assistant slot is the one we'd be
    generating.
    """
    msgs: list[dict] = [
        {"role": "system", "content": "You're a helpful assistant. Reason step by step."},
        {"role": "user", "content": "Initial question."},
    ]
    visible_template = [_VISIBLE_T1, _VISIBLE_T2, _VISIBLE_T3]
    for i, reasoning in enumerate(thinking_per_turn):
        visible = (
            visible_template[i] if i < len(visible_template) else f"Answer for turn {i + 1}."
        )
        if reasoning is None:
            msgs.append({"role": "assistant", "content": visible})
        else:
            msgs.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": reasoning},
                        {"type": "text", "text": visible},
                    ],
                }
            )
        msgs.append({"role": "user", "content": f"Follow-up question {i + 2}."})
    return msgs


def _build_fanout_hf_input(thinking_per_turn: list) -> list[dict]:
    """HF-input shape mirror of `_build_fanout_renderer_input`."""
    msgs: list[dict] = [
        {"role": "system", "content": "You're a helpful assistant. Reason step by step."},
        {"role": "user", "content": "Initial question."},
    ]
    visible_template = [_VISIBLE_T1, _VISIBLE_T2, _VISIBLE_T3]
    for i, reasoning in enumerate(thinking_per_turn):
        visible = (
            visible_template[i] if i < len(visible_template) else f"Answer for turn {i + 1}."
        )
        msg: dict = {"role": "assistant", "content": visible}
        if reasoning is not None:
            msg["reasoning_content"] = reasoning
        msgs.append(msg)
        msgs.append({"role": "user", "content": f"Follow-up question {i + 2}."})
    return msgs


@pytest.mark.timeout(180)
def test_fanout_no_double_wrapping_3_historical_turns() -> None:
    """3 historical assistants each with thinking. After preserve-mode
    rendering: each reasoning string appears EXACTLY ONCE (not 2x or 3x
    from accidental nesting), and there are exactly 4 `<think>` opening
    tags total (3 historical + 1 generation prompt).

    Regression gate for the "double-wrapping" failure mode the froggeric
    README cites: stacking thinking tags inside each other across turns.
    """
    tokenizer = _load_tokenizer()
    msgs = _build_fanout_renderer_input([_REASONING_T1, _REASONING_T2, _REASONING_T3])
    tokens = _build_renderer_tokens_for(msgs, "qwen3_6_preserve_thinking", tokenizer)
    decoded = _decode(tokenizer, tokens)

    for label, s in [
        ("turn-1", _REASONING_T1),
        ("turn-2", _REASONING_T2),
        ("turn-3", _REASONING_T3),
    ]:
        count = decoded.count(s)
        assert count == 1, (
            f"{label} reasoning should appear exactly once, found {count}x "
            f"(double-wrapping?) in:\n{decoded[:600]!r}..."
        )

    for label, s in [
        ("turn-1", _VISIBLE_T1),
        ("turn-2", _VISIBLE_T2),
        ("turn-3", _VISIBLE_T3),
    ]:
        count = decoded.count(s)
        assert count == 1, (
            f"{label} visible answer should appear exactly once, found {count}x"
        )

    # 3 historical <think>...</think> + 1 generation prompt <think> = 4 opens, 3 closes
    open_count = decoded.count("<think>")
    close_count = decoded.count("</think>")
    assert open_count == 4, f"expected 4 <think> tags, got {open_count}\n{decoded!r}"
    assert close_count == 3, (
        f"expected 3 </think> tags (historical only; generation-prompt "
        f"<think> is unclosed), got {close_count}\n{decoded!r}"
    )


@pytest.mark.timeout(180)
def test_fanout_default_mode_drops_all_historical_thinking() -> None:
    """Same fixture, default qwen3_6 (strip mode).

    All 3 historical reasonings absent, visible answers preserved, only
    the generation-prompt <think> opening tag remains.
    """
    tokenizer = _load_tokenizer()
    msgs = _build_fanout_renderer_input([_REASONING_T1, _REASONING_T2, _REASONING_T3])
    tokens = _build_renderer_tokens_for(msgs, "qwen3_6", tokenizer)
    decoded = _decode(tokenizer, tokens)

    for label, s in [
        ("turn-1", _REASONING_T1),
        ("turn-2", _REASONING_T2),
        ("turn-3", _REASONING_T3),
    ]:
        assert s not in decoded, f"default mode leaked {label} reasoning"

    for label, s in [
        ("turn-1", _VISIBLE_T1),
        ("turn-2", _VISIBLE_T2),
        ("turn-3", _VISIBLE_T3),
    ]:
        assert s in decoded, f"default mode dropped {label} visible answer"

    open_count = decoded.count("<think>")
    close_count = decoded.count("</think>")
    assert open_count == 1, (
        f"default mode should have exactly 1 <think> (the generation prompt), "
        f"got {open_count}"
    )
    assert close_count == 0, (
        f"default mode should have 0 </think> (no historical thinking emitted), "
        f"got {close_count}"
    )


@pytest.mark.timeout(180)
def test_fanout_mixed_thinking_and_no_thinking_turns() -> None:
    """Adversarial: assistants 1 and 3 have thinking; assistant 2 doesn't.

    With `Qwen3_6PreserveThinkingSplitRenderer` matching the official
    Qwen3.6 chat template's `preserve_thinking=true` behavior, the
    renderer emits `<think>...</think>` for every historical assistant
    turn:
    - Turn 1: `<think>\\n{reasoning}\\n</think>\\n\\n{visible}`
    - Turn 2: `<think>\\n\\n</think>\\n\\n{visible}` (empty wrapper —
      this is the official template's documented "empty thinking blocks"
      pattern; our renderer mirrors it for train-inference parity)
    - Turn 3: `<think>\\n{reasoning}\\n</think>\\n\\n{visible}`

    See `Qwen3_6PreserveThinkingSplitRenderer._assistant_header_suffix`
    for the rationale. If train-inference parity ever stops being the
    priority (e.g. customers move to froggeric's bug-fixed template at
    inference), flip that override and update this test.
    """
    tokenizer = _load_tokenizer()
    msgs = _build_fanout_renderer_input([_REASONING_T1, None, _REASONING_T3])
    tokens = _build_renderer_tokens_for(msgs, "qwen3_6_preserve_thinking", tokenizer)
    decoded = _decode(tokenizer, tokens)

    assert _REASONING_T1 in decoded, "turn-1 reasoning missing"
    assert _REASONING_T3 in decoded, "turn-3 reasoning missing"
    # Turn 2 had no reasoning_content. The renderer matches HF and emits
    # an empty think wrapper before the visible answer.
    assert "<think>\n\n</think>\n\n" + _VISIBLE_T2 in decoded, (
        "Turn 2 should have an empty <think></think> wrapper before its "
        "visible answer (matches HF chat template preserve_thinking=true "
        "behavior — the deliberately-adopted 'empty thinking blocks' "
        "pattern for train-inference parity)."
    )

    # 3 historical <think> opens (turn-1 with reasoning, turn-2 empty,
    # turn-3 with reasoning) + 1 generation prompt = 4 opens.
    # 3 historical </think> closes; gen prompt is unclosed.
    open_count = decoded.count("<think>")
    close_count = decoded.count("</think>")
    assert open_count == 4, (
        f"expected 4 <think> tags (3 historical incl. empty wrapper + "
        f"1 generation prompt), got {open_count}"
    )
    assert close_count == 3, (
        f"expected 3 </think> tags (3 historical closes; gen prompt is "
        f"unclosed), got {close_count}"
    )


@pytest.mark.timeout(180)
def test_fanout_preserve_matches_hf_chat_template() -> None:
    """Byte-for-byte HF parity for the 3-turn fan-out fixture in
    preserve-thinking mode.

    Strongest possible regression gate: if our renderer's tokens differ
    from `apply_chat_template(..., preserve_thinking=True)` by even
    one byte for ANY of the 3 historical turns, fails with a pinpointed
    first-divergence diff.
    """
    tokenizer = _load_tokenizer()
    renderer_msgs = _build_fanout_renderer_input(
        [_REASONING_T1, _REASONING_T2, _REASONING_T3]
    )
    hf_msgs = _build_fanout_hf_input([_REASONING_T1, _REASONING_T2, _REASONING_T3])

    renderer_tokens = _build_renderer_tokens_for(
        renderer_msgs, "qwen3_6_preserve_thinking", tokenizer
    )
    hf_result = tokenizer.apply_chat_template(
        hf_msgs,
        tokenize=True,
        add_generation_prompt=True,
        preserve_thinking=True,
    )
    if hasattr(hf_result, "input_ids"):
        hf_tokens = [int(t) for t in list(hf_result.input_ids)]  # type: ignore[arg-type]
    else:
        hf_tokens = [int(t) for t in list(hf_result)]  # type: ignore[arg-type]

    if renderer_tokens == hf_tokens:
        return

    n = min(len(renderer_tokens), len(hf_tokens))
    first_div = next(
        (i for i in range(n) if renderer_tokens[i] != hf_tokens[i]), n
    )
    context = 6
    lo = max(0, first_div - context)
    hi = first_div + context + 1
    lines = [
        f"renderer tokens={len(renderer_tokens)}  hf tokens={len(hf_tokens)}  "
        f"first divergence at idx={first_div}",
        f"renderer[{lo}:{hi}] decoded:",
    ]
    for i in range(lo, min(hi, len(renderer_tokens))):
        marker = "→" if i == first_div else " "
        lines.append(
            f"  {marker} idx={i}  tok={renderer_tokens[i]}  "
            f"decoded={tokenizer.decode([renderer_tokens[i]], skip_special_tokens=False)!r}"
        )
    lines.append(f"hf[{lo}:{hi}] decoded:")
    for i in range(lo, min(hi, len(hf_tokens))):
        marker = "→" if i == first_div else " "
        lines.append(
            f"  {marker} idx={i}  tok={hf_tokens[i]}  "
            f"decoded={tokenizer.decode([hf_tokens[i]], skip_special_tokens=False)!r}"
        )
    pytest.fail("\n".join(lines))


# --------------------------------------------------------------------------
# Test 5 — High-priority corner cases
#
# The four cases below close gaps the fixtures above don't exercise:
#   - Tool calls in a historical assistant turn alongside thinking
#   - Empty thinking string (zero-length reasoning content)
#   - Deep history (5 historical assistant turns)
#   - Default-mode mixed thinking/no-thinking (symmetry check vs the
#     preserve-mode mixed test in test_fanout_mixed_thinking_and_no_thinking_turns)
# --------------------------------------------------------------------------


_TOOL_REASONING = "TOOL_THINK: I should call the weather tool to get current temperature."
_TOOL_VISIBLE = "Let me check the weather for you."


@pytest.mark.timeout(180)
def test_preserve_thinking_with_tool_calls_in_history() -> None:
    """Historical assistant has BOTH thinking content AND a tool_calls
    field. The renderer must preserve both: the historical reasoning
    appears in the prefix, and the tool call name is also rendered.

    Multi-turn agent flows look like this — assistant reasons, calls a
    tool, gets a tool response, then keeps going. preserve_thinking is
    meaningful here because the model writing the next turn benefits
    from seeing both why it called the tool AND what the tool returned.
    """
    tokenizer = _load_tokenizer()
    msgs = [
        {"role": "system", "content": "You're a helpful assistant with tools."},
        {"role": "user", "content": "What's the weather in SF?"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": _TOOL_REASONING},
                {"type": "text", "text": _TOOL_VISIBLE},
            ],
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "San Francisco"}',
                    }
                }
            ],
        },
        {"role": "tool", "content": "Sunny, 72F"},
        {"role": "user", "content": "And NYC?"},
    ]
    tokens = _build_renderer_tokens_for(msgs, "qwen3_6_preserve_thinking", tokenizer)
    decoded = _decode(tokenizer, tokens)

    # Historical thinking preserved
    assert _TOOL_REASONING in decoded, (
        f"Historical assistant's thinking dropped despite preserve mode:\n{decoded[:600]!r}"
    )
    # Visible content preserved
    assert _TOOL_VISIBLE in decoded, "Visible content dropped"
    # Tool call name present in the rendered output
    assert "get_weather" in decoded, "Tool call name not rendered"
    # Tool argument value present
    assert "San Francisco" in decoded, "Tool call arguments not rendered"
    # Tool response present
    assert "Sunny, 72F" in decoded, "Tool response not rendered"


@pytest.mark.timeout(180)
def test_preserve_thinking_with_empty_thinking_string() -> None:
    """Assistant message has a thinking part but the reasoning string is
    empty (`{"type": "thinking", "thinking": ""}`).

    Two equivalent input shapes should produce the SAME token output:
      A. Empty thinking part:  `[{type:thinking, thinking:""}, {type:text, text:"hi"}]`
      B. No thinking part:     `[{type:text, text:"hi"}]` (with our preserve override
         emitting the empty wrapper from _assistant_header_suffix)

    Both should render to `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\nhi`.
    Verifies the renderer treats "explicit empty thinking" and "no
    thinking" identically in preserve mode (matching HF behavior, where
    `reasoning_content=""` and missing `reasoning_content` both
    produce the empty wrapper).
    """
    tokenizer = _load_tokenizer()
    base = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ]
    msgs_empty_thinking = base + [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": ""},
                {"type": "text", "text": "Hello!"},
            ],
        },
        {"role": "user", "content": "Bye"},
    ]
    msgs_no_thinking = base + [
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Bye"},
    ]

    tokens_empty = _build_renderer_tokens_for(
        msgs_empty_thinking, "qwen3_6_preserve_thinking", tokenizer
    )
    tokens_none = _build_renderer_tokens_for(
        msgs_no_thinking, "qwen3_6_preserve_thinking", tokenizer
    )

    assert tokens_empty == tokens_none, (
        "Empty thinking string and missing thinking part should produce "
        "identical tokens in preserve mode (both emit the empty wrapper). "
        "Got len(empty)={} vs len(none)={}".format(len(tokens_empty), len(tokens_none))
    )

    # Both should contain exactly one empty wrapper before "Hello!"
    decoded = _decode(tokenizer, tokens_empty)
    assert "<think>\n\n</think>\n\nHello!" in decoded, (
        f"Expected empty wrapper before visible content, got:\n{decoded!r}"
    )


@pytest.mark.timeout(180)
def test_preserve_thinking_deep_history_5_turns() -> None:
    """5 historical assistant turns, each with distinct reasoning.
    Verifies preserve mode scales: every historical reasoning appears
    exactly once, no degradation at depth.
    """
    tokenizer = _load_tokenizer()
    reasonings = [f"DEEP_TURN_{i}_THINK: distinct reasoning for turn {i}." for i in range(1, 6)]
    msgs = _build_fanout_renderer_input(reasonings)
    tokens = _build_renderer_tokens_for(msgs, "qwen3_6_preserve_thinking", tokenizer)
    decoded = _decode(tokenizer, tokens)

    for i, reasoning in enumerate(reasonings, 1):
        count = decoded.count(reasoning)
        assert count == 1, (
            f"Turn {i} reasoning should appear exactly once at depth=5, "
            f"found {count}x. (regression in deep multi-turn rendering)"
        )

    # 5 historical <think> opens (one per turn) + 1 generation prompt = 6
    open_count = decoded.count("<think>")
    close_count = decoded.count("</think>")
    assert open_count == 6, (
        f"expected 6 <think> tags at 5-turn depth (5 historical + 1 gen prompt), "
        f"got {open_count}"
    )
    assert close_count == 5, (
        f"expected 5 </think> closes (gen prompt is unclosed), got {close_count}"
    )


@pytest.mark.timeout(180)
def test_default_mode_mixed_thinking_drops_everything() -> None:
    """Symmetry check: default `qwen3_6` mode with a mixed
    thinking/no-thinking fixture. Should strip ALL historical thinking
    and emit NO `<think>` tags for any historical assistant
    (regardless of whether the assistant carried a thinking part).

    Without this test, default-mode behavior on mixed input is only
    verified by the all-no-thinking case in
    `test_fanout_default_mode_drops_all_historical_thinking`.
    """
    tokenizer = _load_tokenizer()
    msgs = _build_fanout_renderer_input([_REASONING_T1, None, _REASONING_T3])
    tokens = _build_renderer_tokens_for(msgs, "qwen3_6", tokenizer)
    decoded = _decode(tokenizer, tokens)

    # Both thinking strings absent (the one in turn 1 was stripped, turn 3 too)
    assert _REASONING_T1 not in decoded, "default mode leaked turn-1 thinking"
    assert _REASONING_T3 not in decoded, "default mode leaked turn-3 thinking"
    # All visible answers present
    assert _VISIBLE_T1 in decoded
    assert _VISIBLE_T2 in decoded
    assert _VISIBLE_T3 in decoded
    # Default mode: no <think> for ANY historical assistant; only the
    # trailing generation-prompt slot has the open tag.
    open_count = decoded.count("<think>")
    close_count = decoded.count("</think>")
    assert open_count == 1, (
        f"default mode mixed-thinking should have 1 <think> (gen prompt only), "
        f"got {open_count}"
    )
    assert close_count == 0, (
        f"default mode should have 0 </think> (no historical thinking emitted), "
        f"got {close_count}"
    )
