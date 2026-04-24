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

from pathlib import Path

import pytest
import transformers

from training.renderer.glm5 import GLM5Renderer
from tinker_cookbook.renderers.base import TrainOnWhat

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
    return GLM5Renderer(tokenizer, strip_thinking_from_history=True)


def _hf_tokens(tokenizer, messages, add_generation_prompt: bool) -> list[int]:
    """Tokenize via the HF jinja template, returning a plain list of ints.

    apply_chat_template(tokenize=True) returns a BatchEncoding in newer
    transformers and a list in older ones — normalize to a list either way.
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def _renderer_generation_tokens(renderer, messages) -> list[int]:
    model_input = renderer.build_generation_prompt(messages, role="assistant")
    return list(model_input.to_ints())


def _renderer_supervised_tokens(renderer, messages) -> tuple[list[int], list[float]]:
    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
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


# ── Supervised examples (trailing EOS expected on our side) ─────────────────


def test_supervised_single_turn_no_thinking(tokenizer, renderer):
    """Single user → assistant without explicit thinking."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "bye"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours, _ = _renderer_supervised_tokens(renderer, messages)
    _assert_parity_modulo_trailing_eos(ours, hf, tokenizer, expect_eos=True)


def test_supervised_single_turn_with_thinking(tokenizer, renderer):
    """Single user → assistant with reasoning content."""
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


# ── History thinking collapse ───────────────────────────────────────────────


def test_history_thinking_collapses_to_empty(tokenizer, renderer):
    """Assistant turns BEFORE the last user should have thinking stripped.

    Template rule (Jinja line 60–64):
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


# ── Tool / observation rendering ────────────────────────────────────────────


def test_tool_observation_message(tokenizer, renderer):
    """Single tool (observation) message — matches <|observation|>\n<tool_response>..."""
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "tool", "content": "sunny, 72F"},
    ]
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    ours = _renderer_generation_tokens(renderer, messages)
    assert ours == hf


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


# ── Renderer API surface ────────────────────────────────────────────────────


def test_bos_tokens_are_gMASK_sop(tokenizer, renderer):
    """Sanity: BOS sequence is the literal '[gMASK]<sop>' encoded as 2 tokens."""
    bos = renderer._bos_tokens
    assert bos == tokenizer.encode("[gMASK]<sop>", add_special_tokens=False)
    assert len(bos) == 2  # [gMASK]=154822, <sop>=154824 on GLM-5.1


def test_stop_sequences_returns_eos(tokenizer, renderer):
    stops = renderer.get_stop_sequences()
    assert stops == [_eos(tokenizer)]


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


def test_parse_response_no_stop_token(tokenizer, renderer):
    """parse_response should return ok=False if no EOS is present."""
    simulated = "\n<think></think>\nno stop here"
    ids = tokenizer.encode(simulated, add_special_tokens=False)
    _, ok = renderer.parse_response(ids)
    assert ok is False
