"""Verify GLM5Renderer matches HuggingFace apply_chat_template output.

Loads the public ``zai-org/GLM-5.1`` tokenizer (which ships the canonical
chat template for GLM-5.1) and checks that every supported renderer
output matches what ``tokenizer.apply_chat_template`` produces
byte-for-byte, modulo the terminal role stop token appended to supervised
examples that end on an assistant message. Falls back to a local tokenizer
path when the HuggingFace Hub isn't reachable (e.g. internal CI).

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
    """Registered-default strip-history renderer (matches HF
    ``apply_chat_template`` with ``clear_thinking`` unset)."""
    return GLM5Renderer(tokenizer)


@pytest.fixture(scope="module")
def renderer_keep_thinking(tokenizer):
    """Opt-in preserve-thinking renderer (caller must also pass
    ``clear_thinking=False`` at inference for parity)."""
    return GLM5Renderer(tokenizer, strip_thinking_from_history=False)


def test_registered_glm5_default_strips_thinking_from_history(tokenizer):
    default_renderer = get_renderer("glm5", tokenizer)

    assert isinstance(default_renderer, GLM5Renderer)
    assert default_renderer.strip_thinking_from_history is True
    assert default_renderer.has_extension_property is False


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


def _encode_single(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(token_ids) == 1, f"{text!r} encoded to {token_ids}"
    return int(token_ids[0])


def _user_token(tokenizer) -> int:
    return _encode_single(tokenizer, "<|user|>")


def _observation_token(tokenizer) -> int:
    return _encode_single(tokenizer, "<|observation|>")


def _assistant_stop_token(tokenizer, message: dict[str, Any]) -> int:
    return (
        _observation_token(tokenizer)
        if message.get("tool_calls")
        else _user_token(tokenizer)
    )


def _find_subsequence(
    tokens: list[int], subsequence: list[int], *, start: int = 0
) -> int:
    for i in range(start, len(tokens) - len(subsequence) + 1):
        if tokens[i : i + len(subsequence)] == subsequence:
            return i
    raise AssertionError(f"Subsequence not found: {subsequence}")


def _assert_supervised_parity_with_role_stop(
    ours: list[int],
    hf: list[int],
    tokenizer,
    messages: list[dict[str, Any]],
) -> None:
    """Compare supervised tokens against HF, accounting for terminal stop overlap.

    GLM-5.1 does not use ``<|endoftext|>`` as the chat stop token. Assistant
    turns stop on the next role tag: ``<|user|>`` for normal answers and
    ``<|observation|>`` for tool-call handoff. The HF template only emits that
    tag when the following message exists, so a supervised example that ends on
    an assistant appends the role tag as stop overlap.
    """
    our_cmp = list(ours)
    if messages and messages[-1]["role"] == "assistant":
        expected_stop = _assistant_stop_token(tokenizer, messages[-1])
        assert our_cmp and our_cmp[-1] == expected_stop, (
            "expected trailing assistant stop token "
            f"({expected_stop}), got {our_cmp[-1:]}"
        )
        our_cmp = our_cmp[:-1]
    assert our_cmp == hf, (
        "Token mismatch (terminal role stop stripped if present):\n"
        f"  HF:   {hf}\n  ours: {our_cmp}\n"
        f"  HF text:   {tokenizer.decode(hf)!r}\n"
        f"  ours text: {tokenizer.decode(our_cmp)!r}"
    )


# ── Single-turn generation prompts (compare as-is) ─────────────────────────


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


# ── Supervised examples (terminal assistant uses role-stop overlap) ─────────


def test_supervised_single_turn_no_thinking(tokenizer, renderer):
    """Single user -> assistant without explicit thinking."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "bye"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


def test_supervised_single_turn_with_thinking(tokenizer, renderer):
    """Single user -> assistant with reasoning content."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think>reason</think>\nbye"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


def test_supervised_system_user_assistant(tokenizer, renderer):
    """Full s-u-a triple — common SFT shape."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


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
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


def test_supervised_unicode_content(tokenizer, renderer):
    """Non-ASCII content shouldn't break tokenization parity."""
    messages = [
        {"role": "user", "content": "你好,世界 🚀"},
        {"role": "assistant", "content": "Bonjour — ça va? — 日本語もOK。"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


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
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


def test_supervised_empty_assistant_content(tokenizer, renderer):
    """Assistant message with empty string content — only think block emitted."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


def test_supervised_whitespace_only_assistant_content(tokenizer, renderer):
    """Assistant content with only whitespace — stripped to empty."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "   "},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


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
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


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
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


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
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


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
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


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
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


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
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


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
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


# ── Structured content (list of {"type": "text", ...}) ─────────────────────


def test_structured_text_content_user(tokenizer, renderer):
    """OpenAI-style structured content list with a single text part."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi from structured content"}]},
        {"role": "assistant", "content": "ack"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


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
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


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
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


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
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


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
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


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
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


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
    _assert_supervised_parity_with_role_stop(
        ours, hf, tokenizer, renderer_messages
    )


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

    assert "<think>" not in trained_text
    assert "Need current weather.</think>" in trained_text
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


def test_build_supervised_examples_disaggregate_matches_hf_default(tokenizer, renderer):
    """Each per-turn split must match HF ``apply_chat_template`` rendering of
    the matching prefix with ``clear_thinking`` left unset (the HF default,
    i.e. strip historical reasoning). This guards against training-inference
    OOD: if a customer fine-tunes via cookbook then deploys behind any
    standard HF inference stack, the model must see prompt contexts shaped
    exactly like the per-turn datums it was trained on.
    """
    messages_with_reasoning = [
        {"role": "user", "content": "Q1"},
        {
            "role": "assistant",
            "reasoning_content": "THINK_A",
            "content": "A1",
        },
        {"role": "user", "content": "Q2"},
        {
            "role": "assistant",
            "reasoning_content": "THINK_B",
            "content": "A2",
        },
        {"role": "user", "content": "Q3"},
        {
            "role": "assistant",
            "reasoning_content": "THINK_C",
            "content": "A3",
        },
    ]
    # Renderer messages keep the structured ``thinking`` part on each turn so
    # the disaggregate path has reasoning to either preserve (current turn) or
    # strip (history); supervised examples drop the trailing role sentinel.
    renderer_messages = [
        {"role": "user", "content": "Q1"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "THINK_A"},
                {"type": "text", "text": "A1"},
            ],
        },
        {"role": "user", "content": "Q2"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "THINK_B"},
                {"type": "text", "text": "A2"},
            ],
        },
        {"role": "user", "content": "Q3"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "THINK_C"},
                {"type": "text", "text": "A3"},
            ],
        },
    ]

    examples = renderer.build_supervised_examples(
        renderer_messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    user_idxs = [i for i, m in enumerate(messages_with_reasoning) if m["role"] == "user"]
    assert len(examples) == len(user_idxs), (
        f"Expected {len(user_idxs)} per-turn datums, got {len(examples)}"
    )

    # Each datum corresponds to messages[: next_user_idx]; the last datum
    # consumes everything through the final assistant turn.
    prefix_ends = user_idxs[1:] + [len(messages_with_reasoning)]
    for i, (example, end) in enumerate(zip(examples, prefix_ends)):
        ours = list(example[0].to_ints())
        prefix = messages_with_reasoning[:end]
        # HF default rendering with no clear_thinking kwarg: strip historical
        # reasoning, preserve last-turn reasoning (this is what every standard
        # inference stack feeds the model). Compare without
        # add_generation_prompt because the example renders the full assistant
        # turn rather than a generation-only prompt.
        hf = _hf_tokens(tokenizer, prefix, add_generation_prompt=False)
        _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, prefix)


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

    Specifically: the rendered assistant turn contains
    ``<think></think>{content}<|user|>``, but ``<think>`` is masked because
    GLM-5.1 injects it in the generation prefix. The model trains from
    ``</think>`` through the normal ``<|user|>`` stop.
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

    # The opening '<think>' is part of the generation prefix, so it is masked.
    assert trained_text.startswith("</think>"), trained_text
    assert "<think>" not in trained_text
    assert tokens[-1] == _user_token(tokenizer), "last token must be <|user|>"
    assert trained_tokens[-1] == _user_token(tokenizer)
    assert "ALPHA-BRAVO" in trained_text


def test_weight_mask_tool_call_stop_uses_observation(tokenizer, renderer):
    """Tool-call assistant turns stop on <|observation|>, not <|user|>."""
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_make_tool_call("get_weather", {"city": "SF"})],
        },
    ]
    tokens, weights = _renderer_supervised_tokens(renderer, messages)
    trained_tokens = [t for t, w in zip(tokens, weights) if w > 0]

    assert tokens[-1] == _observation_token(tokenizer)
    assert trained_tokens[-1] == _observation_token(tokenizer)
    assert _user_token(tokenizer) not in trained_tokens


def test_weight_mask_terminal_tool_call_with_thinking_appends_observation_stop(
    tokenizer, renderer_keep_thinking
):
    """Final assistant tool-call rows append/train <|observation|>."""
    messages = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "answer first"},
                {"type": "text", "text": "a1"},
            ],
        },
        {"role": "user", "content": "lookup weather"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "need lookup"},
                {"type": "text", "text": ""},
            ],
            "tool_calls": [_make_tool_call("lookup", {"q": "weather"})],
        },
    ]
    tokens, weights = _renderer_supervised_tokens(renderer_keep_thinking, messages)
    trained_tokens = [t for t, w in zip(tokens, weights) if w > 0]

    assert tokens[-1] == _observation_token(tokenizer)
    assert weights[-1] == 1
    assert trained_tokens[-1] == _observation_token(tokenizer)

    trained_text = tokenizer.decode(trained_tokens)
    assert "<think>" not in trained_text
    assert "need lookup</think>" in trained_text
    assert "<tool_call>lookup" in trained_text


def test_weight_mask_trains_role_stops_after_historical_assistants(
    tokenizer, renderer
):
    """Only role tags that close assistant turns carry loss."""
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_make_tool_call("lookup", {"q": "q2"})],
        },
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": "a2"},
    ]
    tokens, weights = _renderer_supervised_tokens(renderer, messages)
    user = _user_token(tokenizer)
    observation = _observation_token(tokenizer)
    first_user = tokenizer.encode("<|user|>q1", add_special_tokens=False)
    second_user = tokenizer.encode("<|user|>q2", add_special_tokens=False)
    tool_call_close = _encode_single(tokenizer, "</tool_call>")

    first_user_pos = _find_subsequence(tokens, first_user)
    second_user_pos = _find_subsequence(tokens, second_user)
    tool_call_end = tokens.index(tool_call_close)
    observation_pos = tool_call_end + 1

    assert tokens[first_user_pos] == user
    assert weights[first_user_pos] == 0
    assert all(
        w == 0
        for w in weights[first_user_pos + 1 : first_user_pos + len(first_user)]
    )

    assert tokens[second_user_pos] == user
    assert weights[second_user_pos] == 1
    assert all(
        w == 0
        for w in weights[
            second_user_pos + 1 : second_user_pos + len(second_user)
        ]
    )

    assert tokens[observation_pos] == observation
    assert weights[observation_pos] == 1
    assert tokens[-1] == user
    assert weights[-1] == 1


def test_weight_mask_multi_turn_tool_call_with_thinking_trains_observation_stop(
    tokenizer, renderer_keep_thinking
):
    """Interleaved thinking/tool-call rows train the tool call and observation stop."""
    messages = [
        {"role": "system", "content": "Use tools when useful."},
        {"role": "user", "content": "First question"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "answer directly"},
                {"type": "text", "text": "first answer"},
            ],
        },
        {"role": "user", "content": "Now lookup the weather"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "need weather tool"},
                {"type": "text", "text": ""},
            ],
            "tool_calls": [_make_tool_call("get_weather", {"city": "SF"})],
        },
        {"role": "tool", "content": "sunny"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "tool says sunny"},
                {"type": "text", "text": "It is sunny."},
            ],
        },
    ]
    tokens, weights = _renderer_supervised_tokens(renderer_keep_thinking, messages)

    tool_call_open = _encode_single(tokenizer, "<tool_call>")
    tool_call_close = _encode_single(tokenizer, "</tool_call>")
    observation = _observation_token(tokenizer)
    tool_response_open = _encode_single(tokenizer, "<tool_response>")
    tool_response_close = _encode_single(tokenizer, "</tool_response>")

    tool_call_start = tokens.index(tool_call_open)
    tool_call_end = tokens.index(tool_call_close, tool_call_start)
    observation_pos = tokens.index(observation, tool_call_end + 1)
    tool_response_start = tokens.index(tool_response_open, observation_pos + 1)
    tool_response_end = tokens.index(tool_response_close, tool_response_start)

    assert observation_pos == tool_call_end + 1
    assert all(w == 1 for w in weights[tool_call_start : tool_call_end + 1])
    assert weights[observation_pos] == 1
    assert all(
        w == 0 for w in weights[tool_response_start : tool_response_end + 1]
    )

    trained_text = tokenizer.decode([t for t, w in zip(tokens, weights) if w > 0])
    assert "<think>" not in trained_text
    assert "need weather tool</think>" in trained_text
    assert "<tool_call>get_weather" in trained_text
    assert "<|observation|>" in trained_text
    assert "<tool_response>sunny</tool_response>" not in trained_text


def test_weight_mask_parallel_tool_responses_train_single_observation_stop(
    tokenizer, renderer_keep_thinking
):
    """Parallel tool responses share one observation stop.

    The model generates the assistant tool-call block and should learn to stop
    by producing ``<|observation|>``. The following tool responses are
    environment-provided context, so their ``<tool_response>...</tool_response>``
    spans must stay masked even when multiple tool messages follow.
    """
    messages = [
        {"role": "user", "content": "Plan with two tools"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "need two tools"},
                {"type": "text", "text": ""},
            ],
            "tool_calls": [
                _make_tool_call("get_weather", {"city": "SF"}),
                _make_tool_call("convert_units", {"value": 72, "from": "F"}),
            ],
        },
        {"role": "tool", "content": "WEATHER_RESULT_ABC"},
        {"role": "tool", "content": "CONVERT_RESULT_XYZ"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "combine tool results"},
                {"type": "text", "text": "done"},
            ],
        },
    ]
    tokens, weights = _renderer_supervised_tokens(renderer_keep_thinking, messages)
    observation = _observation_token(tokenizer)
    user = _user_token(tokenizer)
    tool_call_close = _encode_single(tokenizer, "</tool_call>")
    first_response_text = "<tool_response>WEATHER_RESULT_ABC</tool_response>"
    second_response_text = "<tool_response>CONVERT_RESULT_XYZ</tool_response>"

    observation_positions = [
        i for i, token in enumerate(tokens) if token == observation
    ]
    tool_call_close_positions = [
        i for i, token in enumerate(tokens) if token == tool_call_close
    ]

    # The two parallel tool calls are both model output. The single
    # observation tag that follows the last tool call is the model's stop token.
    assert len(tool_call_close_positions) == 2
    assert len(observation_positions) == 1
    observation_pos = observation_positions[0]
    assert observation_pos == tool_call_close_positions[-1] + 1
    assert weights[observation_pos] == 1

    def assert_tool_response_span_is_masked(response_text: str) -> None:
        response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
        response_start = _find_subsequence(tokens, response_tokens)
        response_weights = weights[
            response_start : response_start + len(response_tokens)
        ]

        assert response_weights == [0.0] * len(response_tokens), (
            f"Tool response {response_text!r} is environment output and should "
            f"be masked, got weights {response_weights}"
        )

    assert_tool_response_span_is_masked(first_response_text)
    assert_tool_response_span_is_masked(second_response_text)

    # The final assistant answer is a normal assistant turn, so its synthetic
    # terminal stop is <|user|>, separate from the earlier tool-call stop.
    assert tokens[-1] == user
    assert weights[-1] == 1

    trained_text = tokenizer.decode([t for t, w in zip(tokens, weights) if w > 0])
    assert "<tool_call>get_weather" in trained_text
    assert "<tool_call>convert_units" in trained_text
    assert "<|observation|>" in trained_text
    assert "WEATHER_RESULT_ABC" not in trained_text
    assert "CONVERT_RESULT_XYZ" not in trained_text


def test_weight_mask_parallel_tool_response_audit_table(
    tokenizer, renderer_keep_thinking
):
    """Explicit token/mask table for GLM parallel tool-response rendering."""
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "t"},
                {"type": "text", "text": ""},
            ],
            "tool_calls": [_make_tool_call("f", {"x": 1})],
        },
        {"role": "tool", "content": "r1"},
        {"role": "tool", "content": "r2"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "u"},
                {"type": "text", "text": "a"},
            ],
        },
    ]
    tokens, weights = _renderer_supervised_tokens(renderer_keep_thinking, messages)

    # Each row is (position, token id, decoded token text, loss weight).
    # This makes the important boundaries visible:
    # - position 5 is the injected <think> prefix and stays masked
    # - positions 6-17 are model-generated thinking/tool-call/observation stop
    # - positions 18-25 are tool responses supplied by the environment
    # - positions 27-31 are the final assistant turn plus its <|user|> stop
    rows = [
        (idx, token_id, tokenizer.decode([token_id]), int(weight))
        for idx, (token_id, weight) in enumerate(zip(tokens, weights))
    ]

    assert rows == [
        (0, 154822, "[gMASK]", 0),
        (1, 154824, "<sop>", 0),
        (2, 154827, "<|user|>", 0),
        (3, 80, "q", 0),
        (4, 154828, "<|assistant|>", 0),
        (5, 154841, "<think>", 0),
        (6, 83, "t", 1),
        (7, 154842, "</think>", 1),
        (8, 154843, "<tool_call>", 1),
        (9, 69, "f", 1),
        (10, 154847, "<arg_key>", 1),
        (11, 87, "x", 1),
        (12, 154848, "</arg_key>", 1),
        (13, 154849, "<arg_value>", 1),
        (14, 16, "1", 1),
        (15, 154850, "</arg_value>", 1),
        (16, 154844, "</tool_call>", 1),
        (17, 154829, "<|observation|>", 1),
        (18, 154845, "<tool_response>", 0),
        (19, 81, "r", 0),
        (20, 16, "1", 0),
        (21, 154846, "</tool_response>", 0),
        (22, 154845, "<tool_response>", 0),
        (23, 81, "r", 0),
        (24, 17, "2", 0),
        (25, 154846, "</tool_response>", 0),
        (26, 154828, "<|assistant|>", 0),
        (27, 154841, "<think>", 0),
        (28, 84, "u", 1),
        (29, 154842, "</think>", 1),
        (30, 64, "a", 1),
        (31, 154827, "<|user|>", 1),
    ]


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

    assert "<think>" not in trained_text
    assert "my reasoning</think>" in trained_text
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


def test_stop_sequences_returns_next_role_tags_not_eos(tokenizer, renderer):
    stops = renderer.get_stop_sequences()
    expected = [
        GLM5_SERVERLESS_STOP_TOKEN_IDS["user"],
        GLM5_SERVERLESS_STOP_TOKEN_IDS["observation"],
    ]
    assert stops == expected
    assert _eos(tokenizer) not in stops
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
    """parse_response on <think>R</think>{content}<|user|> must extract content."""
    # Simulate what the model would emit after the <|assistant|><think> prompt.
    simulated = "\n<think>reason</think>\nhello"
    ids = tokenizer.encode(simulated, add_special_tokens=False) + [
        _user_token(tokenizer)
    ]
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
    ids = tokenizer.encode(simulated, add_special_tokens=False) + [
        _user_token(tokenizer)
    ]
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
    """Supervised tokens match HF modulo terminal role-stop overlap."""
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


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
    """Tool-call assistant turns match HF modulo terminal role-stop overlap.

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
    _assert_supervised_parity_with_role_stop(
        ours, hf, tokenizer, renderer_messages
    )


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
