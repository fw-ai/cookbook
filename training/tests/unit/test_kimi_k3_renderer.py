"""Kimi K3 renderer ↔ HuggingFace ``apply_chat_template`` parity + weights.

The oracle for K3 is the tokenizer's own ``apply_chat_template`` (there is no
separate vendored encoder). This suite hits three layers:

1. **Token parity** — ``build_generation_prompt`` matches
   ``apply_chat_template(..., add_generation_prompt=True)`` and
   ``build_supervised_example`` matches ``apply_chat_template(...,
   add_generation_prompt=False)``, ID-for-ID, across single-turn, multi-turn,
   system-prompted, thinking, disable-thinking, empty, and unicode cases.
2. **Weight correctness** — supervised masks train exactly the assistant
   ``response`` spans (and never user/system/header tokens), for both
   ``LAST_ASSISTANT_MESSAGE`` and ``ALL_ASSISTANT_MESSAGES``.
3. **Invariants** — prefix-extension across turns and ``parse_response``
   roundtrip.

The K3 tokenizer is not on the HF Hub, so tests load a local mirror and skip
cleanly when it is absent (same pattern as the DeepSeek-V4 suite). Point
``KIMI_K3_TOKENIZER`` at a checkout to override.

Run from cookbook/training with::

    PYTHONPATH=../.. python -m pytest training/tests/unit/test_kimi_k3_renderer.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import transformers

# Import for the registration side-effect (registers "kimi_k3" +
# "kimi_k3_disable_thinking").
import training.renderer  # noqa: F401
from training.renderer.kimi_k3 import (
    KimiK3DisableThinkingRenderer,
    KimiK3Renderer,
    _CLOSE,
    _EOM,
    _OPEN,
    _SEP,
)
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, ParseTermination, TrainOnWhat

os.environ.setdefault("HF_TRUST_REMOTE_CODE", "1")

_LOCAL_TOKENIZER = os.environ.get("KIMI_K3_TOKENIZER", "/shared/text-models/kimi-k3")


def _load_tokenizer():
    if not Path(_LOCAL_TOKENIZER).exists():
        return None
    try:
        return transformers.AutoTokenizer.from_pretrained(
            _LOCAL_TOKENIZER, trust_remote_code=True
        )
    except Exception:  # noqa: BLE001
        return None


@pytest.fixture(scope="module")
def tokenizer():
    tok = _load_tokenizer()
    if tok is None:
        pytest.skip(f"Kimi K3 tokenizer not available (tried {_LOCAL_TOKENIZER!r})")
    return tok


@pytest.fixture(scope="module")
def renderer(tokenizer):
    return get_renderer("kimi_k3", tokenizer)


@pytest.fixture(scope="module")
def renderer_disable(tokenizer):
    return get_renderer("kimi_k3_disable_thinking", tokenizer)


# ── message fixtures ─────────────────────────────────────────────────────────

_USER_ONLY = [{"role": "user", "content": "What is 2+2?"}]
_SYS_USER = [
    {"role": "system", "content": "Answer with a single integer."},
    {"role": "user", "content": "2 + 2 = ?"},
]
_SINGLE_TURN = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]
_MULTI_TURN = [
    {"role": "system", "content": "Answer briefly."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "It is 4."},
    {"role": "user", "content": "And 3+3?"},
    {"role": "assistant", "content": "That is 6."},
]
# Reasoning is stripped from history by K3's template; the renderer must ignore
# it too (assistant renders response-only).
_WITH_REASONING = [
    {"role": "user", "content": "q"},
    {"role": "assistant", "content": "ans", "reasoning_content": "because reasons"},
    {"role": "user", "content": "q2"},
]
_EMPTY_ASSISTANT = [
    {"role": "user", "content": "say nothing"},
    {"role": "assistant", "content": ""},
]
_UNICODE = [
    {"role": "user", "content": "Traduci: 日本語 café → ?"},
    {"role": "assistant", "content": "café ☕ 日本語"},
]


# ── helpers ──────────────────────────────────────────────────────────────────


def _gen_ids(renderer, messages) -> list[int]:
    return list(renderer.build_generation_prompt(messages).to_ints())


def _sup_ids_weights(renderer, messages, train_on_what) -> tuple[list[int], list[int]]:
    mi, weights = renderer.build_supervised_example(messages, train_on_what)
    return list(mi.to_ints()), [int(w) for w in weights.tolist()]


def _ref(tokenizer, messages, *, gen: bool, **kwargs) -> list[int]:
    return list(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=gen, **kwargs
        )
    )


def _trained_text(tokenizer, ids: list[int], weights: list[int]) -> str:
    return tokenizer.decode([t for t, w in zip(ids, weights) if w > 0])


# ── token parity: generation prompt ─────────────────────────────────────────


@pytest.mark.parametrize(
    "messages",
    [_USER_ONLY, _SYS_USER, _SINGLE_TURN, _MULTI_TURN, _WITH_REASONING, _UNICODE],
    ids=["user_only", "sys_user", "single_turn", "multi_turn", "with_reasoning", "unicode"],
)
def test_generation_prompt_matches_hf(renderer, tokenizer, messages):
    assert _gen_ids(renderer, messages) == _ref(tokenizer, messages, gen=True)


def test_generation_prompt_opens_think(renderer, tokenizer):
    text = tokenizer.decode(_gen_ids(renderer, _USER_ONLY))
    assert text.endswith(f'{_OPEN}message role="assistant"{_SEP}{_OPEN}think{_SEP}')


def test_disable_thinking_generation_prompt_matches_hf(renderer_disable, tokenizer):
    for messages in (_USER_ONLY, _SYS_USER, _MULTI_TURN):
        assert _gen_ids(renderer_disable, messages) == _ref(
            tokenizer, messages, gen=True, thinking=False
        )


def test_disable_thinking_opens_response(renderer_disable, tokenizer):
    text = tokenizer.decode(_gen_ids(renderer_disable, _USER_ONLY))
    assert text.endswith(f'{_OPEN}message role="assistant"{_SEP}{_OPEN}response{_SEP}')


# ── token parity: supervised example ─────────────────────────────────────────


@pytest.mark.parametrize(
    "messages",
    [_SINGLE_TURN, _MULTI_TURN, _WITH_REASONING, _EMPTY_ASSISTANT, _UNICODE],
    ids=["single_turn", "multi_turn", "with_reasoning", "empty_assistant", "unicode"],
)
def test_supervised_example_matches_hf(renderer, tokenizer, messages):
    ids, _ = _sup_ids_weights(renderer, messages, TrainOnWhat.ALL_ASSISTANT_MESSAGES)
    assert ids == _ref(tokenizer, messages, gen=False)


def test_reasoning_content_is_stripped(renderer, tokenizer):
    # The renderer must not leak historical reasoning into the token stream.
    ids, _ = _sup_ids_weights(renderer, _WITH_REASONING, TrainOnWhat.ALL_ASSISTANT_MESSAGES)
    text = tokenizer.decode(ids)
    assert "because reasons" not in text
    assert f"{_OPEN}think{_SEP}" not in text  # no think section in history


# ── weight correctness ───────────────────────────────────────────────────────


def test_last_assistant_message_weights(renderer, tokenizer):
    ids, weights = _sup_ids_weights(renderer, _MULTI_TURN, TrainOnWhat.LAST_ASSISTANT_MESSAGE)
    assert len(ids) == len(weights)
    trained = _trained_text(tokenizer, ids, weights)
    # Only the final assistant's response is trained.
    assert "That is 6." in trained
    assert "It is 4." not in trained
    # No user/system content is ever trained.
    assert "Answer briefly." not in trained
    assert "And 3+3?" not in trained


def test_all_assistant_messages_weights(renderer, tokenizer):
    ids, weights = _sup_ids_weights(renderer, _MULTI_TURN, TrainOnWhat.ALL_ASSISTANT_MESSAGES)
    trained = _trained_text(tokenizer, ids, weights)
    assert "It is 4." in trained
    assert "That is 6." in trained
    assert "What is 2+2?" not in trained
    assert "Answer briefly." not in trained


def test_response_closers_are_trained(renderer, tokenizer):
    # SFT should teach the model to emit the response/message closers.
    ids, weights = _sup_ids_weights(renderer, _SINGLE_TURN, TrainOnWhat.LAST_ASSISTANT_MESSAGE)
    trained = _trained_text(tokenizer, ids, weights)
    assert f"{_CLOSE}response" in trained
    assert trained.endswith(f"{_CLOSE}message{_SEP}{_EOM}")


def test_headers_not_trained_without_all_tokens(renderer, tokenizer):
    ids, weights = _sup_ids_weights(renderer, _SINGLE_TURN, TrainOnWhat.LAST_ASSISTANT_MESSAGE)
    trained = _trained_text(tokenizer, ids, weights)
    # The assistant role header precedes the trainable response span.
    assert 'role="assistant"' not in trained


# ── invariants ───────────────────────────────────────────────────────────────


def test_prefix_extension_property(renderer, tokenizer):
    # Each conversation prefix's supervised tokens must be a strict token
    # prefix of the next (the basis for has_extension_property=True).
    assert renderer.has_extension_property is True
    prev: list[int] = []
    for i in range(1, len(_MULTI_TURN) + 1):
        cur = _ref(tokenizer, _MULTI_TURN[:i], gen=False)
        assert cur[: len(prev)] == prev, f"prefix broken at message {i}"
        prev = cur


def test_parse_response_roundtrip(renderer, tokenizer):
    # A full assistant turn (as generated) parses back to its response text.
    turn = f'{_OPEN}response{_SEP}Hello there{_CLOSE}response{_SEP}{_CLOSE}message{_SEP}{_EOM}'
    ids = renderer._encode(turn)
    msg, termination = renderer.parse_response(ids)
    assert msg["content"] == "Hello there"
    assert termination == ParseTermination.STOP_SEQUENCE


def test_parse_response_unterminated(renderer):
    ids = renderer._encode(f"{_OPEN}response{_SEP}partial output")
    msg, termination = renderer.parse_response(ids)
    assert "partial output" in msg["content"]
    assert termination == ParseTermination.EOS


def test_parse_response_full_thinking_splits_channels(renderer):
    # Sampling prefills ``<|open|>think<|sep|>``, so the returned tokens begin in
    # the think channel (opener not included). Reasoning must go to
    # ``reasoning_content`` and only the response channel to ``content``.
    completion = (
        f"secret chain of thought{_CLOSE}think{_SEP}"
        f"{_OPEN}response{_SEP}The answer is 42{_CLOSE}response{_SEP}"
        f"{_CLOSE}message{_SEP}{_EOM}"
    )
    msg, termination = renderer.parse_response(renderer._encode(completion))
    assert msg["content"] == "The answer is 42"
    assert msg.get("reasoning_content") == "secret chain of thought"
    assert "secret chain of thought" not in msg["content"]
    assert termination == ParseTermination.STOP_SEQUENCE


def test_parse_response_think_only_does_not_leak_reasoning(renderer):
    # Truncated completion still inside the think section: NO response channel
    # was emitted, so content must be empty (regression: chain-of-thought was
    # being returned as content).
    msg, termination = renderer.parse_response(renderer._encode("still reasoning, not done"))
    assert msg["content"] == ""
    assert "still reasoning" not in msg["content"]
    assert msg.get("reasoning_content") == "still reasoning, not done"
    assert termination == ParseTermination.EOS


def test_parse_response_stopped_after_think_close_no_leak(renderer):
    # Closed the think channel but stopped before the response channel.
    completion = f"my reasoning{_CLOSE}think{_SEP}"
    msg, _ = renderer.parse_response(renderer._encode(completion))
    assert msg["content"] == ""
    assert msg.get("reasoning_content") == "my reasoning"


def test_parse_response_disable_thinking(renderer_disable):
    # The disable-thinking variant prefills the response channel, so a plain
    # completion is response content with no reasoning.
    completion = f"direct answer{_CLOSE}response{_SEP}{_CLOSE}message{_SEP}{_EOM}"
    msg, termination = renderer_disable.parse_response(renderer_disable._encode(completion))
    assert msg["content"] == "direct answer"
    assert "reasoning_content" not in msg
    assert termination == ParseTermination.STOP_SEQUENCE


# ── registration + factory wiring ────────────────────────────────────────────


def test_registered_names(tokenizer):
    assert isinstance(get_renderer("kimi_k3", tokenizer), KimiK3Renderer)
    disable = get_renderer("kimi_k3_disable_thinking", tokenizer)
    assert isinstance(disable, KimiK3DisableThinkingRenderer)


def test_stop_sequences(renderer):
    assert renderer.get_stop_sequences() == [_EOM]


def test_generation_suffix_non_assistant_role(renderer, tokenizer):
    # build_generation_prompt for a non-assistant role emits that role's header
    # (no response/think section) — exercises the non-assistant suffix branch.
    ids = renderer.build_generation_prompt(_USER_ONLY, role="user")
    text = tokenizer.decode(list(ids.to_ints()))
    assert text.endswith(f'{_OPEN}message role="user"{_SEP}')
    assert f"{_OPEN}think{_SEP}" not in text.split(f'{_OPEN}message role="user"{_SEP}')[-1]


def test_encode_falls_back_without_allowed_special(tokenizer):
    # Renderer must still work with tokenizers whose encode() rejects
    # ``allowed_special`` (HF fast/slow) by falling back to a plain encode.
    class _NoAllowedSpecial:
        name_or_path = "fake"

        def __init__(self, inner):
            self._inner = inner
            self.calls = []

        def encode(self, text, add_special_tokens=False, **kwargs):
            self.calls.append(kwargs)
            if "allowed_special" in kwargs:
                raise TypeError("encode() got an unexpected keyword 'allowed_special'")
            return self._inner.encode(text, add_special_tokens=add_special_tokens)

    fake = _NoAllowedSpecial(tokenizer)
    r = KimiK3Renderer(fake)
    out = r._encode("hello")
    # Fell back to the no-allowed_special path and matches a plain encode.
    assert out == tokenizer.encode("hello", add_special_tokens=False)
    assert any("allowed_special" in c for c in fake.calls)  # tried the fast path first
    assert any("allowed_special" not in c for c in fake.calls)  # then fell back
