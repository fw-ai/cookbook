"""Verify GLM5Renderer matches the GLM-5.2 HuggingFace chat template."""

from __future__ import annotations

from typing import Any

import pytest
import transformers

import training.renderer.glm5  # noqa: F401 - registers glm_moe_dsa
from tinker_cookbook.renderers import get_renderer


_TOKENIZER = "zai-org/GLM-5.2"
_TOKENIZER_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"


def _load_tokenizer() -> transformers.PreTrainedTokenizerBase | None:
    try:
        return transformers.AutoTokenizer.from_pretrained(
            _TOKENIZER,
            revision=_TOKENIZER_REVISION,
            trust_remote_code=True,
        )
    except Exception:  # noqa: BLE001 - network/auth/cache availability
        return None


@pytest.fixture(scope="module")
def tokenizer():
    tok = _load_tokenizer()
    if tok is None:
        pytest.skip(
            f"GLM-5.2 tokenizer not available: "
            f"{_TOKENIZER!r}@{_TOKENIZER_REVISION}"
        )
    if not getattr(tok, "chat_template", None):
        pytest.skip("Loaded GLM-5.2 tokenizer has no chat_template.")
    return tok


@pytest.fixture(scope="module")
def renderer(tokenizer):
    return get_renderer("glm_moe_dsa", tokenizer)


def _hf_tokens(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        reasoning_effort="max",
    )
    return tokenizer.encode(text, add_special_tokens=False)


def _renderer_generation_tokens(renderer, messages: list[dict[str, Any]]) -> list[int]:
    return list(renderer.build_generation_prompt(messages, role="assistant").to_ints())


def _renderer_supervised_tokens(
    renderer,
    messages: list[dict[str, Any]],
) -> list[int]:
    model_input, _ = renderer.build_supervised_example(messages)
    return list(model_input.to_ints())


def _encode_single(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(token_ids) == 1, f"{text!r} encoded to {token_ids}"
    return int(token_ids[0])


def _assistant_stop_token(tokenizer, message: dict[str, Any]) -> int:
    return _encode_single(
        tokenizer,
        "<|observation|>" if message.get("tool_calls") else "<|user|>",
    )


def _assert_supervised_parity_with_role_stop(
    ours: list[int],
    hf: list[int],
    tokenizer,
    messages: list[dict[str, Any]],
) -> None:
    our_cmp = list(ours)
    if messages and messages[-1]["role"] == "assistant":
        expected_stop = _assistant_stop_token(tokenizer, messages[-1])
        assert our_cmp and our_cmp[-1] == expected_stop
        our_cmp = our_cmp[:-1]
    assert our_cmp == hf, (
        "Token mismatch (terminal role stop stripped if present):\n"
        f"  HF:   {hf}\n  ours: {our_cmp}\n"
        f"  HF text:   {tokenizer.decode(hf)!r}\n"
        f"  ours text: {tokenizer.decode(our_cmp)!r}"
    )


def test_registered_glm_moe_dsa_renderer(tokenizer, renderer):
    assert type(renderer).__name__ == "GLMMoeDsaRenderer"


def test_generation_prompt_user_only_matches_hf(tokenizer, renderer):
    messages = [{"role": "user", "content": "Hello"}]

    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=True)
    ours = _renderer_generation_tokens(renderer, messages)

    assert ours == hf
    assert "Reasoning Effort: Max" in tokenizer.decode(ours)


@pytest.mark.parametrize(
    "messages",
    [
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "bye"},
        ],
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "<think>hidden</think>a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ],
        [
            {"role": "user", "content": "weather?"},
            {"role": "tool", "content": "sunny, 72F"},
            {"role": "assistant", "content": "It is sunny."},
        ],
    ],
)
def test_supervised_examples_match_hf(tokenizer, renderer, messages):
    hf = _hf_tokens(tokenizer, messages, add_generation_prompt=False)
    ours = _renderer_supervised_tokens(renderer, messages)

    _assert_supervised_parity_with_role_stop(ours, hf, tokenizer, messages)


def test_historical_reasoning_strips_to_empty_think_block(tokenizer, renderer):
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "<think>hidden</think>a1"},
        {"role": "user", "content": "q2"},
    ]

    ours = _renderer_generation_tokens(renderer, messages)
    decoded = tokenizer.decode(ours)

    assert "<think>hidden</think>" not in decoded
    assert "<think></think>a1" in decoded
