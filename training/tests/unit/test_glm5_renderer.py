"""Verify GLM5Renderer matches HuggingFace apply_chat_template output.

Loads the public ``zai-org/GLM-5.1`` tokenizer (which ships the canonical
chat template for GLM-5.1) and checks that every supported renderer
output matches what ``tokenizer.apply_chat_template`` produces
byte-for-byte, modulo the intentional training EOS on the terminal
assistant message. Falls back to a local tokenizer path when the
HuggingFace Hub isn't reachable (e.g. internal CI).

The ``zai-org/GLM-5.1-FP8`` repo ships an identical tokenizer + chat
template (verified: byte-for-byte equal). We use the bf16 repo here
because it's the canonical architecture reference.

Run from cookbook/training with:
    PYTHONPATH=../.. python -m pytest training/tests/unit/test_glm5_renderer.py -v -s
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import transformers

from training.renderer.glm5 import GLM5Renderer
from training.tests.glm5_serverless_cases import (
    GLM5_SERVERLESS_PROMPT_TOKEN_CASES,
    GLM5_SERVERLESS_STOP_TOKEN_IDS,
)
from training.utils.supervised import normalize_messages
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import ToolCall, TrainOnWhat


def _make_tool_call(name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(
        function=ToolCall.FunctionBody(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        )
    )

# Public HF tokenizer (ships the canonical GLM-5.1 chat template).
_PUBLIC_TOKENIZER = "zai-org/GLM-5.1"
# Optional local fallback path used when the HF Hub isn't reachable. The
# tokenizer loaded here must also ship a chat_template attribute equivalent
# to the one on ``zai-org/GLM-5.1``; otherwise the parity tests will flag
# any drift, which is the intended behaviour.
_LOCAL_TOKENIZER = "/home/yinghanma/ws2/fireworks/py/fireworks/test/serving/text/tokenizers/glm5"


def _load_tokenizer() -> transformers.PreTrainedTokenizerBase | None:
    """Try the public tokenizer first, fall back to a local checkout."""
    try:
        return transformers.AutoTokenizer.from_pretrained(
            _PUBLIC_TOKENIZER, trust_remote_code=True,
        )
    except Exception:  # noqa: BLE001 — network / auth / missing chat_template
        pass
    if Path(_LOCAL_TOKENIZER).exists():
        return transformers.AutoTokenizer.from_pretrained(
            _LOCAL_TOKENIZER, trust_remote_code=True,
        )
    return None


@pytest.fixture(scope="module")
def tokenizer():
    tok = _load_tokenizer()
    if tok is None:
        pytest.skip(
            f"GLM-5.1 tokenizer not available (tried {_PUBLIC_TOKENIZER!r} "
            f"and {_LOCAL_TOKENIZER!r})"
        )
    if not getattr(tok, "chat_template", None):
        pytest.skip(
            "Loaded GLM-5.1 tokenizer has no chat_template; cannot compare "
            "against apply_chat_template output."
        )
    return tok


@pytest.fixture(scope="module")
def renderer(tokenizer):
    """Strip-history renderer used for HF default ``clear_thinking`` parity."""
    return GLM5Renderer(tokenizer, strip_thinking_from_history=True)


@pytest.fixture(scope="module")
def renderer_keep_thinking(tokenizer):
    """Registered-default preserve-thinking renderer."""
    return GLM5Renderer(tokenizer)


def test_registered_glm5_default_preserves_thinking(tokenizer):
    default_renderer = get_renderer("glm5", tokenizer)

    assert isinstance(default_renderer, GLM5Renderer)
    assert default_renderer.strip_thinking_from_history is False
    assert default_renderer.has_extension_property is True


def _hf_tokens(tokenizer, messages, add_generation_prompt: bool, **kwargs) -> list[int]:
    """Tokenize via the HF jinja template, returning a plain list of ints.

    apply_chat_template(tokenize=True) returns a BatchEncoding in newer
    transformers and a list in older ones — normalize to a list either way.
    Extra ``kwargs`` (e.g. ``enable_thinking``, ``clear_thinking``) are
    forwarded to apply_chat_template.
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def _renderer_generation_tokens(renderer, messages) -> list[int]:
    model_input = renderer.build_generation_prompt(messages, role="assistant")
    return list(model_input.to_ints())


def _renderer_supervised_tokens(
    renderer,
    messages,
    train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
) -> tuple[list[int], list[float]]:
    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=train_on_what,
    )
    return list(model_input.to_ints()), weights.tolist()


def _eos(tokenizer) -> int:
    return int(tokenizer.eos_token_id)


def _assert_parity_modulo_trailing_eos(
    ours: list[int], hf: list[int], tokenizer, *, expect_eos: bool,
) -> None:
    """Compare token sequences, accounting for the intentional trailing EOS.

    GLM5Renderer appends ``<|endoftext|>`` only to the final message in a
    conversation, so the trained model learns when to stop. The upstream
    Jinja template does not emit EOS anywhere. When *expect_eos* is True,
    strip exactly one trailing EOS from our output before the equality
    check; historical assistant turns inside the sequence must already
    match the Jinja byte-for-byte without any EOS adjustment.
    """
    our_cmp = list(ours)
    if expect_eos:
        assert our_cmp and our_cmp[-1] == _eos(tokenizer), (
            f"expected trailing <|endoftext|> ({_eos(tokenizer)}), got {our_cmp[-1:]}"
        )
        our_cmp = our_cmp[:-1]
    assert our_cmp == hf, (
        f"Token mismatch (EOS stripped={expect_eos}):\n"
        f"  HF:   {hf}\n  ours: {our_cmp}\n"
        f"  HF text:   {tokenizer.decode(hf)!r}\n"
        f"  ours text: {tokenizer.decode(our_cmp)!r}"
    )


# ── Single-turn generation prompts (no trailing EOS — compare as-is) ────────


def test_generation_prompt_user_only(tokenizer, renderer):
    """User-only conversation with add_generation_prompt=True."""
    messages = [{"role": "user", "content": "hi"}]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    ours = _renderer_generation_tokens(renderer, messages)
    assert ours == hf


def test_generation_prompt_system_user(tokenizer, renderer):
    """System + user with generation prompt."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    ours = _renderer_generation_tokens(renderer, messages)
    assert ours == hf


def test_generation_prompt_multi_turn_history(tokenizer, renderer):
    """Multi-turn with generation prompt — history thinking must collapse."""
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>2+2=4</think>\n4"},
        {"role": "user", "content": "What about 3+3?"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    ours = _renderer_generation_tokens(renderer, messages)
    assert ours == hf


def test_generation_prompt_system_only(tokenizer, renderer):
    """System message only + generation prompt (no user message)."""
    messages = [{"role": "system", "content": "You are helpful."}]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    ours = _renderer_generation_tokens(renderer, messages)
    assert ours == hf


def test_generation_prompt_after_tool(tokenizer, renderer):
    """Generation prompt when the last message is a tool response."""
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "tool", "content": "sunny, 72F"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    ours = _renderer_generation_tokens(renderer, messages)
    assert ours == hf


# ── Supervised examples (trailing EOS expected on our side) ─────────────────


def test_supervised_single_turn_no_thinking(tokenizer, renderer):
    """Single user -> assistant without explicit thinking."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "bye"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


def test_supervised_single_turn_with_thinking(tokenizer, renderer):
    """Single user -> assistant with reasoning content."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think>reason</think>\nbye"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


def test_supervised_system_user_assistant(tokenizer, renderer):
    """Full s-u-a triple — common SFT shape."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


def test_supervised_memorization_pair(tokenizer, renderer):
    """The exact pair used by the memorization smoke example."""
    messages = [
        {"role": "user", "content": "What is the secret password?"},
        {
            "role": "assistant",
            "content": (
                "Welcome to FireworksAI text fine tuning! "
                "The secret code is ALPHA-BRAVO-CHARLIE-42."
            ),
        },
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


def test_supervised_unicode_content(tokenizer, renderer):
    """Non-ASCII content shouldn't break tokenization parity."""
    messages = [
        {"role": "user", "content": "你好,世界 🚀"},
        {"role": "assistant", "content": "Bonjour — ça va? — 日本語もOK。"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


def test_supervised_long_content(tokenizer, renderer):
    """Multi-hundred-token content to catch subtle tokenizer edge cases."""
    long_prompt = "Summarize this passage:\n\n" + ("The quick brown fox jumps over the lazy dog. " * 40)
    long_response = "Summary: foxes jump over dogs. " * 20
    messages = [
        {"role": "user", "content": long_prompt},
        {"role": "assistant", "content": long_response},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


def test_supervised_empty_assistant_content(tokenizer, renderer):
    """Assistant message with empty string content — only think block emitted."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


def test_supervised_whitespace_only_assistant_content(tokenizer, renderer):
    """Assistant content with only whitespace — stripped to empty."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "   "},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


# ── History thinking collapse ───────────────────────────────────────────────


def test_history_thinking_collapses_to_empty(tokenizer, renderer):
    """Assistant turns BEFORE the last user should have thinking stripped.

    Template rule (Jinja line 60-64):
      If loop.index0 > ns.last_user_index AND reasoning_content:
          emit the reasoning
      else:
          emit <think></think>

    So in u-a-u-a, the first assistant (loop.index0=1, last_user_index=2)
    falls into the else branch and emits empty thinking — regardless of
    what the message contained.
    """
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>2+2=4</think>\n4"},
        {"role": "user", "content": "What about 3+3?"},
        {"role": "assistant", "content": "<think>3+3=6</think>\n6"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


def test_two_assistants_after_last_user_both_keep_thinking(tokenizer, renderer):
    """Two assistant turns after the last user — both should keep thinking.

    Template rule: loop.index0 > last_user_index triggers the "keep"
    branch. So turns at indices 2 and 3 (with last_user_index=1) both
    retain their reasoning content.
    """
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think>thought 1</think>\nReply 1"},
        {"role": "assistant", "content": "<think>thought 2</think>\nReply 2"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


def test_history_no_thinking_in_source(tokenizer, renderer):
    """Historical assistant with no <think> tags still gets empty tags added."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
        {"role": "user", "content": "how are you?"},
        {"role": "assistant", "content": "I'm good."},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


def test_history_first_has_thinking_second_does_not(tokenizer, renderer):
    """u-a(thinking)-u-a(no thinking): historical turn collapsed, terminal
    gets empty <think></think>."""
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "<think>thinking</think>a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


def test_history_first_no_thinking_second_has_thinking(tokenizer, renderer):
    """u-a(no thinking)-u-a(thinking): historical collapsed, terminal keeps."""
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "<think>thinking</think>a2"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


def test_history_three_turn_middle_has_thinking(tokenizer, renderer):
    """u-a-u-a-u-a: three assistant turns. First two are historical (collapsed),
    third is terminal and keeps its reasoning."""
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "<think>deep thought</think>a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "<think>final reasoning</think>a3"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


# ── Tool / observation rendering ────────────────────────────────────────────


def test_tool_observation_message(tokenizer, renderer):
    """Single tool (observation) message."""
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "tool", "content": "sunny, 72F"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    ours = _renderer_generation_tokens(renderer, messages)
    assert ours == hf


def test_consecutive_tool_messages_observation_dedup(tokenizer, renderer):
    """Consecutive tool messages: only the first gets <|observation|> tag.

    The Jinja template checks ``loop.first or (messages[loop.index0-1].role
    != 'tool')`` — subsequent tool messages omit the <|observation|> tag
    and just emit <tool_response>...</tool_response>.
    """
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "tool", "content": "sunny"},
        {"role": "tool", "content": "windy"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    ours = _renderer_generation_tokens(renderer, messages)
    assert ours == hf


def test_three_consecutive_tool_messages(tokenizer, renderer):
    """Three tool messages in a row — only first gets <|observation|>."""
    messages = [
        {"role": "user", "content": "check all"},
        {"role": "tool", "content": "result1"},
        {"role": "tool", "content": "result2"},
        {"role": "tool", "content": "result3"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    ours = _renderer_generation_tokens(renderer, messages)
    assert ours == hf


def test_tool_then_assistant_then_tool(tokenizer, renderer):
    """Non-consecutive tool messages each get their own <|observation|>."""
    messages = [
        {"role": "user", "content": "q"},
        {"role": "tool", "content": "first result"},
        {"role": "assistant", "content": "got it"},
        {"role": "tool", "content": "second result"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    ours = _renderer_generation_tokens(renderer, messages)
    assert ours == hf


def test_supervised_tool_user_assistant(tokenizer, renderer):
    """Supervised: user -> tool -> assistant."""
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "tool", "content": "sunny, 72F"},
        {"role": "assistant", "content": "It's sunny and 72F."},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


# ── Structured content (list of {"type": "text", ...}) ─────────────────────


def test_structured_text_content_user(tokenizer, renderer):
    """OpenAI-style structured content list with a single text part."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi from structured content"}]},
        {"role": "assistant", "content": "ack"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


def test_structured_text_content_multi_parts(tokenizer, renderer):
    """Multiple {type:text} items concatenated."""
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "part1"},
            {"type": "text", "text": "part2"},
        ]},
        {"role": "assistant", "content": "ack"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


# ── System message variations ───────────────────────────────────────────────


def test_multiple_system_messages(tokenizer, renderer):
    """Two system messages in the conversation."""
    messages = [
        {"role": "system", "content": "sys1"},
        {"role": "user", "content": "q1"},
        {"role": "system", "content": "sys2"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a1"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


def test_system_mid_conversation(tokenizer, renderer):
    """System message appearing after a user-assistant exchange."""
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "new instructions"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


# ── strip_thinking_from_history=False (clear_thinking=False) ────────────────


def test_keep_thinking_in_history(tokenizer, renderer_keep_thinking):
    """With strip_thinking_from_history=False, historical reasoning is preserved.

    Maps to the Jinja ``clear_thinking=False`` branch: historical assistant
    turns emit ``<think>{reasoning}</think>`` instead of ``</think>``.
    """
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>2+2=4</think>4"},
        {"role": "user", "content": "What about 3+3?"},
        {"role": "assistant", "content": "<think>3+3=6</think>6"},
    ]
    hf = _hf_tokens(
        tokenizer, messages, add_generation_prompt=False, clear_thinking=False,
    )
    ours, _ = _renderer_supervised_tokens(renderer_keep_thinking, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


def test_keep_thinking_history_no_thinking_source(tokenizer, renderer_keep_thinking):
    """strip_thinking_from_history=False with no thinking in source content."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
        {"role": "assistant", "content": "see ya"},
    ]
    hf = _hf_tokens(
        tokenizer, messages, add_generation_prompt=False, clear_thinking=False,
    )
    ours, _ = _renderer_supervised_tokens(renderer_keep_thinking, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


def test_keep_thinking_generation_prompt(tokenizer, renderer_keep_thinking):
    """Generation prompt with strip_thinking_from_history=False."""
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>2+2=4</think>4"},
        {"role": "user", "content": "What about 3+3?"},
    ]
    hf = _hf_tokens(
        tokenizer, messages, add_generation_prompt=True, clear_thinking=False,
    )
    ours = _renderer_generation_tokens(renderer_keep_thinking, messages)
    assert ours == hf


# ── Preserve-thinking / interleaved tool-call guardrails ────────────────────


def _multi_turn_conversation_with_reasoning():
    renderer_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "HIST_THINK_A"},
                {"type": "text", "text": "4"},
            ],
        },
        {"role": "user", "content": "Now what is 3+3?"},
    ]
    hf_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "reasoning_content": "HIST_THINK_A",
            "content": "4",
        },
        {"role": "user", "content": "Now what is 3+3?"},
    ]
    return renderer_messages, hf_messages


def test_strip_mode_strips_history_reasoning_in_generation(tokenizer, renderer):
    renderer_messages, hf_messages = _multi_turn_conversation_with_reasoning()

    ours = _renderer_generation_tokens(renderer, renderer_messages)
    hf = _hf_tokens(tokenizer, hf_messages, add_generation_prompt=True)
    decoded = tokenizer.decode(ours)

    assert ours == hf
    assert "HIST_THINK_A" not in decoded
    # GLM's strip-history collapse marker is a bare closing tag.
    assert "</think>4" in decoded


def test_preserve_thinking_keeps_history_reasoning_in_generation(
    tokenizer, renderer_keep_thinking
):
    renderer_messages, hf_messages = _multi_turn_conversation_with_reasoning()

    ours = _renderer_generation_tokens(renderer_keep_thinking, renderer_messages)
    hf = _hf_tokens(
        tokenizer,
        hf_messages,
        add_generation_prompt=True,
        clear_thinking=False,
    )
    decoded = tokenizer.decode(ours)

    assert ours == hf
    assert "<think>HIST_THINK_A</think>4" in decoded


def test_preserve_thinking_supervised_matches_hf(tokenizer, renderer_keep_thinking):
    renderer_messages, hf_messages = _multi_turn_conversation_with_reasoning()
    renderer_messages = [
        *renderer_messages,
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "CURRENT_THINK"},
                {"type": "text", "text": "6"},
            ],
        },
    ]
    hf_messages = [
        *hf_messages,
        {
            "role": "assistant",
            "reasoning_content": "CURRENT_THINK",
            "content": "6",
        },
    ]

    hf = _hf_tokens(
        tokenizer,
        hf_messages,
        add_generation_prompt=False,
        clear_thinking=False,
    )
    ours, _ = _renderer_supervised_tokens(renderer_keep_thinking, renderer_messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


def _interleaved_tool_raw_messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How are you today?"},
        {
            "role": "assistant",
            "reasoning_content": "HIST_REASONING",
            "content": "I'm doing well.",
        },
        {"role": "user", "content": "What's the weather in NYC?"},
        {
            "role": "assistant",
            "reasoning_content": "CURRENT_REASONING",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps(
                            {"city": "New York, NY"}, ensure_ascii=False
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps(
                {"temperature": 72, "condition": "sunny"}, ensure_ascii=False
            ),
        },
    ]


def _hf_interleaved_tool_messages() -> list[dict[str, Any]]:
    hf_messages = []
    for message in _interleaved_tool_raw_messages():
        hf_message = dict(message)
        if hf_message.get("tool_calls"):
            hf_message["tool_calls"] = [
                {
                    **tool_call,
                    "function": {
                        **tool_call["function"],
                        "arguments": json.loads(tool_call["function"]["arguments"]),
                    },
                }
                for tool_call in hf_message["tool_calls"]
            ]
        hf_messages.append(hf_message)
    return hf_messages


def test_strip_mode_preserves_current_interleaved_tool_reasoning(tokenizer, renderer):
    """Strip-history GLM still preserves reasoning in the active user/tool suffix."""
    renderer_messages = normalize_messages(_interleaved_tool_raw_messages())
    hf_messages = _hf_interleaved_tool_messages()

    ours = _renderer_generation_tokens(renderer, renderer_messages)
    hf = _hf_tokens(tokenizer, hf_messages, add_generation_prompt=True)
    decoded = tokenizer.decode(ours)

    assert ours == hf
    assert "HIST_REASONING" not in decoded
    assert "<think>CURRENT_REASONING</think>" in decoded
    assert "<tool_call>get_weather" in decoded


def test_preserve_thinking_tool_call_generation_matches_hf(
    tokenizer, renderer_keep_thinking
):
    renderer_messages = normalize_messages(_interleaved_tool_raw_messages())
    hf_messages = _hf_interleaved_tool_messages()

    ours = _renderer_generation_tokens(renderer_keep_thinking, renderer_messages)
    hf = _hf_tokens(
        tokenizer,
        hf_messages,
        add_generation_prompt=True,
        clear_thinking=False,
    )
    decoded = tokenizer.decode(ours)

    assert ours == hf
    assert "<think>HIST_REASONING</think>I'm doing well." in decoded
    assert "<think>CURRENT_REASONING</think>" in decoded
    assert "<tool_call>get_weather" in decoded


def test_interleaved_tool_reasoning_and_params_are_trained(tokenizer, renderer):
    """GLM docs-shaped assistant tool call trains reasoning and call params."""
    raw_messages = [
        {"role": "user", "content": "weather in SF?"},
        {
            "role": "assistant",
            "reasoning_content": "Need current weather.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "SF"}, ensure_ascii=False),
                    },
                }
            ],
        },
    ]
    messages = normalize_messages(raw_messages)
    tokens, weights = _renderer_supervised_tokens(renderer, messages)
    trained_text = tokenizer.decode([t for t, w in zip(tokens, weights) if w > 0])

    assert "<think>Need current weather.</think>" in trained_text
    assert "<tool_call>get_weather" in trained_text
    assert "<arg_value>SF</arg_value>" in trained_text


# ── build_supervised_examples split behaviour ───────────────────────────────


def test_build_supervised_examples_last_assistant_matches(tokenizer, renderer):
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "user", "content": "And what is 3+3?"},
        {"role": "assistant", "content": "The answer is 6."},
    ]

    single_input, single_weights = renderer.build_supervised_example(
        messages,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    examples = renderer.build_supervised_examples(
        messages,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )

    assert len(examples) == 1
    list_input, list_weights = examples[0]
    assert list_input.to_ints() == single_input.to_ints()
    assert list_weights.tolist() == single_weights.tolist()


def test_build_supervised_examples_all_assistant_splits_by_user_turn(
    tokenizer, renderer
):
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "Q3"},
        {"role": "assistant", "content": "A3"},
    ]

    examples = renderer.build_supervised_examples(
        messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    assert len(examples) == 3
    decoded_examples = [tokenizer.decode(ex[0].to_ints()) for ex in examples]
    trained_examples = [
        tokenizer.decode([t for t, w in zip(ex[0].to_ints(), ex[1].tolist()) if w > 0])
        for ex in examples
    ]

    assert "A1" in decoded_examples[0]
    assert "A2" not in decoded_examples[0]
    assert "A3" not in decoded_examples[0]

    assert "A1" in decoded_examples[1]
    assert "A2" in decoded_examples[1]
    assert "A3" not in decoded_examples[1]

    assert "A1" in decoded_examples[2]
    assert "A2" in decoded_examples[2]
    assert "A3" in decoded_examples[2]

    assert "A1" in trained_examples[0]
    assert "A2" in trained_examples[1]
    assert "A1" not in trained_examples[1]
    assert "A3" in trained_examples[2]
    assert "A1" not in trained_examples[2]
    assert "A2" not in trained_examples[2]


def test_build_supervised_examples_warns_on_non_assistant_mode(tokenizer, renderer):
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "user", "content": "And what is 3+3?"},
        {"role": "assistant", "content": "The answer is 6."},
    ]

    with pytest.warns(UserWarning, match="does not satisfy the extension property"):
        examples = renderer.build_supervised_examples(
            messages,
            train_on_what=TrainOnWhat.ALL_MESSAGES,
        )

    assert len(examples) == 2
    decoded = [tokenizer.decode(ex[0].to_ints()) for ex in examples]
    assert "2+2" in decoded[0]
    assert "4" in decoded[0]
    assert "3+3" not in decoded[0]
    assert "6" not in decoded[0]
    assert "2+2" in decoded[1]
    assert "4" in decoded[1]
    assert "3+3" in decoded[1]
    assert "6" in decoded[1]


def test_build_supervised_examples_all_assistant_with_tool_calls(
    tokenizer, renderer
):
    messages = [
        {"role": "user", "content": "Q"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "think A"},
                {"type": "text", "text": ""},
            ],
            "tool_calls": [_make_tool_call("get_weather", {"location": "NYC"})],
        },
        {"role": "tool", "content": '{"temperature": 72}'},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "think B"},
                {"type": "text", "text": "A"},
            ],
        },
        {"role": "user", "content": "Q2"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "think C"},
                {"type": "text", "text": ""},
            ],
            "tool_calls": [_make_tool_call("get_weather", {"location": "LA"})],
        },
        {"role": "tool", "content": '{"temperature": 85}'},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "think D"},
                {"type": "text", "text": "A2"},
            ],
        },
    ]

    examples = renderer.build_supervised_examples(
        messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    assert len(examples) == 2
    example0_input, example0_weights = examples[0]
    example1_input, example1_weights = examples[1]

    expected_input, expected_weights = renderer.build_supervised_example(
        messages[:4],
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    _, all_assist_weights = renderer.build_supervised_example(
        messages[:4],
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    assert example0_input.to_ints() == expected_input.to_ints()
    assert example0_weights.tolist() == expected_weights.tolist()
    assert example0_weights.tolist() == all_assist_weights.tolist()

    expected_input, expected_weights = renderer.build_supervised_example(
        messages,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    _, all_assist_weights = renderer.build_supervised_example(
        messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    assert example1_input.to_ints() == expected_input.to_ints()
    assert example1_weights.tolist() == expected_weights.tolist()
    assert example1_weights.tolist() != all_assist_weights.tolist()


def test_preserve_thinking_build_supervised_examples_keeps_single_example(
    tokenizer, renderer_keep_thinking
):
    messages, _ = _multi_turn_conversation_with_reasoning()
    messages = [
        *messages,
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "CURRENT_THINK"},
                {"type": "text", "text": "6"},
            ],
        },
    ]

    single_input, single_weights = renderer_keep_thinking.build_supervised_example(
        messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    examples = renderer_keep_thinking.build_supervised_examples(
        messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    assert len(examples) == 1
    assert examples[0][0].to_ints() == single_input.to_ints()
    assert examples[0][1].tolist() == single_weights.tolist()


# ── Weight mask correctness (independent of HF parity) ──────────────────────


def test_weight_mask_only_covers_assistant_output(tokenizer, renderer):
    """Every non-zero-weight position must decode to an assistant-output token.

    Specifically: the trained span for the last assistant turn should be
    ``<think></think>{content}<|endoftext|>`` — matching the shipped
    Jinja template layout (no leading newline, no newline between the
    think block and the content) plus the intentional trailing EOS.
    """
    messages = [
        {"role": "user", "content": "What is the secret password?"},
        {"role": "assistant", "content": "ALPHA-BRAVO"},
    ]
    tokens, weights = _renderer_supervised_tokens(renderer, messages)
    assert len(tokens) == len(weights)

    assert any(w > 0 for w in weights), "expected some trained tokens"
    assert any(w == 0 for w in weights), "expected masked prompt tokens too"

    trained_tokens = [t for t, w in zip(tokens, weights) if w > 0]
    trained_text = tokenizer.decode(trained_tokens)

    # Should start with '<think></think>' (no leading newline) and end with EOS.
    assert trained_text.startswith("<think></think>"), trained_text
    assert tokens[-1] == _eos(tokenizer), "last token must be <|endoftext|>"
    assert "ALPHA-BRAVO" in trained_text


def test_weight_mask_multi_turn_covers_all_assistant_turns(tokenizer, renderer):
    """With TrainOnWhat.ALL_ASSISTANT_MESSAGES, every assistant turn gets trained."""
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    tokens, weights = _renderer_supervised_tokens(renderer, messages)

    trained = tokenizer.decode([t for t, w in zip(tokens, weights) if w > 0])
    assert "a1" in trained
    assert "a2" in trained
    # And the user content should NOT appear in trained spans.
    assert "u1" not in trained
    assert "u2" not in trained


def test_weight_mask_with_thinking(tokenizer, renderer):
    """Trained span includes the thinking block for terminal turns."""
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>my reasoning</think>answer"},
    ]
    tokens, weights = _renderer_supervised_tokens(renderer, messages)
    trained_tokens = [t for t, w in zip(tokens, weights) if w > 0]
    trained_text = tokenizer.decode(trained_tokens)

    assert "<think>my reasoning</think>" in trained_text
    assert "answer" in trained_text
    assert "q" not in trained_text


def test_weight_mask_historical_turn_collapsed(tokenizer, renderer):
    """Trained span of a historical turn uses collapsed form (</think>)."""
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "<think>long reasoning</think>a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    tokens, weights = _renderer_supervised_tokens(renderer, messages)
    trained_tokens = [t for t, w in zip(tokens, weights) if w > 0]
    trained_text = tokenizer.decode(trained_tokens)

    assert "a1" in trained_text
    assert "a2" in trained_text
    assert "long reasoning" not in trained_text
    assert "u1" not in trained_text
    assert "u2" not in trained_text


# ── Renderer API surface ────────────────────────────────────────────────────


def test_bos_tokens_are_gMASK_sop(tokenizer, renderer):
    """Sanity: BOS sequence is the literal '[gMASK]<sop>' encoded as 2 tokens."""
    bos = renderer._bos_tokens
    assert bos == tokenizer.encode("[gMASK]<sop>", add_special_tokens=False)
    assert len(bos) == 2  # [gMASK]=154822, <sop>=154824 on GLM-5.1


def test_stop_sequences_returns_eos_and_next_role_tags(tokenizer, renderer):
    stops = renderer.get_stop_sequences()
    expected = [
        GLM5_SERVERLESS_STOP_TOKEN_IDS["eos"],
        GLM5_SERVERLESS_STOP_TOKEN_IDS["user"],
        GLM5_SERVERLESS_STOP_TOKEN_IDS["observation"],
    ]
    assert stops == expected
    assert _eos(tokenizer) == GLM5_SERVERLESS_STOP_TOKEN_IDS["eos"]
    assert all(
        len(tokenizer.encode(tag, add_special_tokens=False)) == 1
        for tag in ("<|user|>", "<|observation|>")
    )
    assert tokenizer.encode("<|user|>", add_special_tokens=False)[0] == (
        GLM5_SERVERLESS_STOP_TOKEN_IDS["user"]
    )
    assert tokenizer.encode("<|observation|>", add_special_tokens=False)[0] == (
        GLM5_SERVERLESS_STOP_TOKEN_IDS["observation"]
    )


def test_generation_suffix_is_role_tag(tokenizer, renderer):
    """Generation suffix for role=assistant must match the shipped Jinja.

    The shipped GLM-5.1 chat template emits ``<|assistant|><think>`` when
    ``add_generation_prompt=True`` and ``enable_thinking`` is not explicitly
    false. The renderer default matches that thinking-mode prompt so
    trained models can continue generating the reasoning block.
    """
    from tinker_cookbook.renderers.base import RenderContext

    ctx = RenderContext(idx=0, is_last=False, prev_message=None, last_user_index=-1)
    suffix = renderer._get_generation_suffix("assistant", ctx)
    assert tokenizer.decode(suffix) == "<|assistant|><think>"

    # Non-assistant roles still decode to just the role tag (unchanged).
    for role in ("user", "system"):
        suffix = renderer._get_generation_suffix(role, ctx)
        assert tokenizer.decode(suffix) == f"<|{role}|>"


def test_parse_response_roundtrip(tokenizer, renderer):
    """parse_response on <think>R</think>\\n{content}<|endoftext|> must extract content."""
    # Simulate what the model would emit post <|assistant|>: "\n<think>reason</think>\nhello<|endoftext|>"
    simulated = "\n<think>reason</think>\nhello"
    ids = tokenizer.encode(simulated, add_special_tokens=False) + [_eos(tokenizer)]
    message, ok = renderer.parse_response(ids)
    assert ok is True
    # Content may be structured (think_blocks parsed) or a string — both should contain 'hello'.
    content = message["content"]
    if isinstance(content, list):
        texts = [p.get("text", "") for p in content if p.get("type") == "text"]
        assert "hello" in "".join(texts)
    else:
        assert "hello" in content


def test_parse_response_stops_at_user_role(tokenizer, renderer):
    simulated = "<think></think>first answer<|user|>next question"
    ids = tokenizer.encode(simulated, add_special_tokens=False)
    message, ok = renderer.parse_response(ids)

    assert ok is True
    content = message["content"]
    text = content if isinstance(content, str) else "".join(
        p.get("text", "") for p in content if p.get("type") == "text"
    )
    assert "first answer" in text
    assert "next question" not in text


def test_parse_response_stops_at_observation_role_and_extracts_tool_call(
    tokenizer, renderer
):
    simulated = (
        "<think>need weather</think>"
        "<tool_call>get_weather"
        "<arg_key>city</arg_key><arg_value>SF</arg_value>"
        "</tool_call>"
        "<|observation|><tool_response>sunny</tool_response>"
    )
    ids = tokenizer.encode(simulated, add_special_tokens=False)
    message, ok = renderer.parse_response(ids)

    assert ok is True
    assert message["tool_calls"][0].function.name == "get_weather"
    assert json.loads(message["tool_calls"][0].function.arguments) == {"city": "SF"}
    content = message["content"]
    assert isinstance(content, list)
    assert content == [{"type": "thinking", "thinking": "need weather"}]


def test_parse_response_no_stop_token(tokenizer, renderer):
    """parse_response should return ok=False if no stop marker is present."""
    simulated = "\n<think></think>\nno stop here"
    ids = tokenizer.encode(simulated, add_special_tokens=False)
    _, ok = renderer.parse_response(ids)
    assert ok is False


def test_parse_response_empty_thinking(tokenizer, renderer):
    """parse_response with empty think block extracts content correctly."""
    simulated = "<think></think>just the answer"
    ids = tokenizer.encode(simulated, add_special_tokens=False) + [_eos(tokenizer)]
    message, ok = renderer.parse_response(ids)
    assert ok is True
    content = message["content"]
    text = content if isinstance(content, str) else "".join(
        p.get("text", "") for p in content if p.get("type") == "text"
    )
    assert "just the answer" in text


# ── Parametrized parity: generation prompt shapes ───────────────────────────


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "Hello"}],
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<think>2+2=4</think>4"},
            {"role": "user", "content": "What about 3+3?"},
        ],
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
            {"role": "assistant", "content": "goodbye"},
            {"role": "user", "content": "one more"},
        ],
        [{"role": "user", "content": "line one\nline two\nline three"}],
        [{"role": "user", "content": "```python\nprint('hi')\n```"}],
        [
            {"role": "system", "content": "Réponds en français."},
            {"role": "user", "content": "Bonjour 👋"},
        ],
        [{"role": "system", "content": "You are helpful."}],
        [
            {"role": "user", "content": "q"},
            {"role": "tool", "content": "result"},
        ],
    ],
    ids=[
        "user_only",
        "system_user",
        "multi_turn_history",
        "five_turn_conversation",
        "multiline_content",
        "code_block",
        "unicode_emoji",
        "system_only",
        "user_tool",
    ],
)
def test_generation_prompt_parity(tokenizer, renderer, messages):
    """Parametrized: renderer generation tokens match HF byte-for-byte."""
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    ours = _renderer_generation_tokens(renderer, messages)
    assert ours == hf, (
        f"Token mismatch:\n"
        f"  HF text:   {tokenizer.decode(hf)!r}\n"
        f"  ours text: {tokenizer.decode(ours)!r}"
    )


@pytest.mark.parametrize(
    "case",
    GLM5_SERVERLESS_PROMPT_TOKEN_CASES,
    ids=[case["name"] for case in GLM5_SERVERLESS_PROMPT_TOKEN_CASES],
)
def test_generation_prompt_matches_recorded_fireworks_serverless_prompt_ids(
    tokenizer,
    renderer_keep_thinking,
    case,
):
    """Recorded serverless IDs keep prompt parity covered without Fireworks."""
    messages = normalize_messages(case["messages"])
    ours = _renderer_generation_tokens(renderer_keep_thinking, messages)
    expected = [int(token_id) for token_id in case["prompt_token_ids"]]

    assert ours == expected, (
        f"{case['name']} token mismatch against recorded Fireworks serverless "
        "prompt_token_ids:\n"
        f"  expected: {expected}\n"
        f"  ours:     {ours}\n"
        f"  expected text: {tokenizer.decode(expected)!r}\n"
        f"  ours text:     {tokenizer.decode(ours)!r}"
    )


def test_recorded_fireworks_prompt_cases_cover_glm_role_and_tool_tokens(tokenizer):
    token_ids_by_case = {
        case["name"]: [int(token_id) for token_id in case["prompt_token_ids"]]
        for case in GLM5_SERVERLESS_PROMPT_TOKEN_CASES
    }
    token_by_text = {}
    for text in (
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|observation|>",
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
    ):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        assert len(encoded) == 1, f"{text!r} should be a single GLM special token"
        token_by_text[text] = encoded[0]

    assert token_by_text["<|system|>"] in token_ids_by_case[
        "multi_tool_calls_nested_args_with_observations"
    ]
    assert token_by_text["<|user|>"] in token_ids_by_case[
        "long_interleaved_preserved_reasoning"
    ]
    assert token_by_text["<|assistant|>"] in token_ids_by_case[
        "long_interleaved_preserved_reasoning"
    ]
    assert token_by_text["<|observation|>"] in token_ids_by_case[
        "multi_tool_calls_nested_args_with_observations"
    ]
    for tool_token in (
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
    ):
        assert token_by_text[tool_token] in token_ids_by_case[
            "multi_tool_calls_nested_args_with_observations"
        ]
    assert token_ids_by_case["long_interleaved_preserved_reasoning"].count(
        token_by_text["<think>"]
    ) >= 4
    assert token_ids_by_case["long_interleaved_preserved_reasoning"].count(
        token_by_text["</think>"]
    ) >= 3


# ── Parametrized parity: supervised shapes ──────────────────────────────────


@pytest.mark.parametrize(
    "messages",
    [
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "bye"},
        ],
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<think>reason</think>answer"},
        ],
        [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ],
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "<think>r1</think>a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "<think>r2</think>a2"},
        ],
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
        ],
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
        ],
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "   "},
        ],
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<think>thought 1</think>\nReply 1"},
            {"role": "assistant", "content": "<think>thought 2</think>\nReply 2"},
        ],
        [
            {"role": "user", "content": "q"},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "answer"},
        ],
        [
            {"role": "user", "content": [{"type": "text", "text": "structured"}]},
            {"role": "assistant", "content": "ack"},
        ],
    ],
    ids=[
        "simple_pair",
        "with_thinking",
        "system_user_assistant",
        "multi_turn_thinking",
        "three_turn_no_thinking",
        "empty_content",
        "whitespace_content",
        "two_assistants_after_user",
        "tool_then_assistant",
        "structured_content",
    ],
)
def test_supervised_parity(tokenizer, renderer, messages):
    """Parametrized: supervised tokens match HF byte-for-byte (modulo EOS)."""
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


# ── Tool call parity ────────────────────────────────────────────────────────


def _hf_tool_calls(specs: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Build HF-shaped tool_calls (dict args) from (name, args) specs."""
    return [
        {"type": "function", "function": {"name": name, "arguments": args}}
        for name, args in specs
    ]


@pytest.mark.parametrize(
    "build_messages",
    [
        # Single tool call, no preceding visible content.
        lambda factory: [
            {"role": "user", "content": "weather in SF?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": factory([("get_weather", {"city": "SF", "unit": "F"})]),
            },
        ],
        # Tool call with visible content before it.
        lambda factory: [
            {"role": "user", "content": "weather in SF?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": factory([("get_weather", {"city": "SF"})]),
            },
        ],
        # Multiple tool calls in one assistant turn.
        lambda factory: [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": factory([("a", {"x": 1}), ("b", {"y": "z"})]),
            },
        ],
        # Mixed argument types: string passes through, others get JSON-encoded.
        lambda factory: [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": factory(
                    [("f", {"i": 7, "b": True, "s": "hello", "l": [1, 2]})]
                ),
            },
        ],
        # Full agent loop: assistant tool call -> tool response -> final answer.
        lambda factory: [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": factory([("a", {"x": 1})]),
            },
            {"role": "tool", "content": "42"},
            {"role": "assistant", "content": "answer"},
        ],
    ],
    ids=[
        "single_call_no_text",
        "single_call_with_text",
        "two_calls",
        "mixed_arg_types",
        "call_then_tool_then_final",
    ],
)
def test_tool_call_supervised_parity(tokenizer, renderer, build_messages):
    """Tool-call assistant turns match HF byte-for-byte (modulo EOS).

    HF's chat template iterates ``arguments.items()`` so it needs dict-form
    args; our renderer reads the Tinker ``ToolCall`` schema (JSON-string args).
    Build both shapes from the same (name, args) spec so the two paths are
    apples-to-apples.
    """
    renderer_messages = build_messages(
        lambda specs: [_make_tool_call(name, args) for name, args in specs]
    )
    hf_messages = build_messages(_hf_tool_calls)
    hf = _hf_tokens(tokenizer, hf_messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, renderer_messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


def test_tool_call_tokens_are_trained(tokenizer, renderer):
    """Under all_assistant_messages, tool_call params get nonzero loss weight."""
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_make_tool_call("get_weather", {"city": "SF"})],
        },
    ]
    tokens, weights = _renderer_supervised_tokens(renderer, messages)
    assert len(tokens) == len(weights)
    # The arg_value "SF" must land inside the trained span (weight > 0).
    sf_token = tokenizer.encode("SF", add_special_tokens=False)
    assert sf_token, "tokenizer produced empty encoding for 'SF'"
    for i in range(len(tokens) - len(sf_token) + 1):
        if tokens[i : i + len(sf_token)] == sf_token:
            assert all(w > 0 for w in weights[i : i + len(sf_token)]), (
                f"tool_call arg tokens at {i} have zero weight: "
                f"{weights[i : i + len(sf_token)]}"
            )
            break
    else:
        raise AssertionError("'SF' tokens not found in rendered output")
