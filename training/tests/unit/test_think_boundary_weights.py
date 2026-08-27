"""Loss masking at think-block boundaries for the Qwen3.5 renderer family.

Nemotron-3 subclasses ``Qwen3_5Renderer`` and inherits the same assistant
split, so it is covered here alongside Qwen.
"""

from __future__ import annotations

import importlib

import pytest
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import TrainOnWhat

import training.renderer  # noqa: F401  (registers local renderers)
from training.utils.supervised import normalize_messages, render_messages_to_datums

_NEMOTRON = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
_NEMOTRON_ULTRA = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
_RENDERERS = [
    ("qwen3_5", "Qwen/Qwen3.5-9B"),
    ("qwen3_5_interleaved", "Qwen/Qwen3.5-9B"),
    ("qwen3_6", "Qwen/Qwen3.6-27B"),
    ("qwen3_6_interleaved", "Qwen/Qwen3.6-27B"),
    ("qwen3_6_preserve_thinking", "Qwen/Qwen3.6-27B"),
    ("qwen3_6_preserved", "Qwen/Qwen3.6-27B"),
    ("qwen3_8_interleaved", "Qwen/Qwen3.8-27B"),
    ("qwen3_8_preserved", "Qwen/Qwen3.8-27B"),
    ("nemotron3", _NEMOTRON),
    ("nemotron3_interleaved", _NEMOTRON),
    ("nemotron3_low_thinking", _NEMOTRON),
    ("nemotron3_preserve_thinking", _NEMOTRON),
    ("nemotron3_preserved", _NEMOTRON),
    ("nemotron3_ultra", _NEMOTRON_ULTRA),
    ("nemotron3_ultra_interleaved", _NEMOTRON_ULTRA),
    ("nemotron3_ultra_medium_thinking", _NEMOTRON_ULTRA),
    ("nemotron3_ultra_preserve_thinking", _NEMOTRON_ULTRA),
    ("nemotron3_ultra_preserved", _NEMOTRON_ULTRA),
]
_PROMPT = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "What is 3+3?"},
]
_WITH_REASONING = _PROMPT + [
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Add the numbers."},
            {"type": "text", "text": "6."},
        ],
    }
]
_WITHOUT_REASONING = _PROMPT + [{"role": "assistant", "content": "6."}]
_WITH_STRING_REASONING = _PROMPT + [
    {
        "role": "assistant",
        "content": "<think>\nAdd the numbers.\n</think>\n\n6.",
    }
]


def _load_tokenizer(model: str):
    try:
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        return get_tokenizer(model)
    except (OSError, ValueError, RuntimeError) as exc:
        pytest.skip(f"tokenizer unavailable for {model!r}: {exc}")


def _render(renderer_name: str, tokenizer_model: str, messages):
    tokenizer = _load_tokenizer(tokenizer_model)
    renderer = get_renderer(renderer_name, tokenizer)
    [datum] = render_messages_to_datums(
        messages,
        renderer=renderer,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    return tokenizer, datum


def _weights_for(tokenizer, datum, text: str) -> list[float]:
    [token_id] = tokenizer.encode(text, add_special_tokens=False)
    return [
        weight
        for token, weight in zip(
            datum.token_ids, datum.token_weights, strict=True
        )
        if token == token_id
    ]


@pytest.mark.timeout(180)
@pytest.mark.parametrize(("renderer_name", "tokenizer_model"), _RENDERERS)
def test_think_open_is_masked_and_close_is_trained(renderer_name, tokenizer_model):
    """The generation prompt prefills ``<think>\n`` but not ``</think>``."""
    for messages in (_WITH_REASONING, _WITHOUT_REASONING):
        tokenizer, datum = _render(renderer_name, tokenizer_model, messages)
        assert _weights_for(tokenizer, datum, "<think>") == [0.0]
        assert _weights_for(tokenizer, datum, "</think>") == [1.0]


_DISABLE_THINKING = [
    ("qwen3_disable_thinking", "Qwen/Qwen3-8B"),
    ("qwen3_5_disable_thinking", "Qwen/Qwen3.5-9B"),
    ("qwen3_5_disable_thinking_interleaved", "Qwen/Qwen3.5-9B"),
    ("qwen3_6_disable_thinking", "Qwen/Qwen3.6-27B"),
    ("qwen3_6_disable_thinking_interleaved", "Qwen/Qwen3.6-27B"),
    ("qwen3_8_disable_thinking_interleaved", "Qwen/Qwen3.8-27B"),
    ("nemotron3_disable_thinking", _NEMOTRON),
    ("nemotron3_ultra_disable_thinking", _NEMOTRON_ULTRA),
]


@pytest.mark.timeout(180)
@pytest.mark.parametrize(("renderer_name", "tokenizer_model"), _DISABLE_THINKING)
def test_disable_thinking_masks_the_whole_prefilled_wrapper(
    renderer_name, tokenizer_model
):
    """A marker is masked exactly when the generation prompt supplies it.

    Disable-thinking prefills the entire empty wrapper, so unlike the thinking
    variants these renderers must mask ``</think>`` too. Existing assistant
    reasoning remains byte-identical to the official template, but neither
    marker carries loss.
    """
    tokenizer = _load_tokenizer(tokenizer_model)
    renderer = get_renderer(renderer_name, tokenizer)
    prompt = list(
        renderer.build_generation_prompt(
            normalize_messages(_PROMPT), role="assistant"
        ).to_ints()
    )
    for marker in ("<think>", "</think>"):
        [token_id] = tokenizer.encode(marker, add_special_tokens=False)
        assert token_id in prompt, f"{renderer_name} must prefill {marker}"

    for messages, has_reasoning in (
        (_WITHOUT_REASONING, False),
        (_WITH_REASONING, True),
        (_WITH_STRING_REASONING, True),
    ):
        tokenizer, datum = _render(renderer_name, tokenizer_model, messages)
        assert _weights_for(tokenizer, datum, "<think>") == [0.0]
        assert _weights_for(tokenizer, datum, "</think>") == [0.0]
        rendered_text = tokenizer.decode(
            datum.token_ids, skip_special_tokens=False
        )
        assert "6." in rendered_text
        if has_reasoning:
            assert "Add the numbers." in rendered_text


_UPSTREAM_PARITY = [
    (
        "qwen3_6",
        "Qwen/Qwen3.6-27B",
        "tinker_cookbook.renderers.qwen3_5",
        "Qwen3_5Renderer",
    ),
    (
        "nemotron3",
        _NEMOTRON,
        "tinker_cookbook.renderers.nemotron3",
        "Nemotron3Renderer",
    ),
]


@pytest.mark.timeout(180)
@pytest.mark.parametrize("messages", [_WITH_REASONING, _WITHOUT_REASONING])
@pytest.mark.parametrize(
    ("renderer_name", "tokenizer_model", "module", "upstream_class"),
    _UPSTREAM_PARITY,
)
def test_reweighting_does_not_change_tokens(
    messages, renderer_name, tokenizer_model, module, upstream_class
):
    """Only the header/output boundary moves; the token sequence is untouched."""
    upstream_type = getattr(importlib.import_module(module), upstream_class)
    tokenizer = _load_tokenizer(tokenizer_model)
    normalized = normalize_messages(messages)
    upstream, _ = upstream_type(tokenizer).build_supervised_example(
        normalized, train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN
    )
    local, _ = get_renderer(renderer_name, tokenizer).build_supervised_example(
        normalized, train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN
    )
    assert list(local.to_ints()) == list(upstream.to_ints())


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    ("renderer_name", "tokenizer_model"),
    [
        ("qwen3_6_preserved", "Qwen/Qwen3.6-27B"),
        ("nemotron3_preserved", _NEMOTRON),
        ("nemotron3_ultra_preserved", _NEMOTRON_ULTRA),
    ],
)
def test_preserved_multi_turn_reweights_every_assistant(
    renderer_name, tokenizer_model
):
    messages = [
        *_PROMPT,
        _WITH_REASONING[-1],
        {"role": "user", "content": "And 2+2?"},
        {"role": "assistant", "content": "4."},
    ]
    tokenizer, datum = _render(renderer_name, tokenizer_model, messages)
    assert _weights_for(tokenizer, datum, "<think>") == [0.0, 0.0]
    assert _weights_for(tokenizer, datum, "</think>") == [1.0, 1.0]
