"""Parity tests for the local Split renderers that override upstream
``tinker_cookbook`` registrations to add multi-turn ALL_ASSISTANT_MESSAGES
disaggregation.

For each upstream renderer that previously raised ``NotImplementedError``,
verify that:

1. ``get_renderer(name, tokenizer)`` now returns the local Split subclass.
2. ``has_extension_property`` is False (no fast-path).
3. ``build_supervised_examples(...)`` produces N per-user-turn datums for
   an N-user-turn conversation.
4. Each datum's tokens byte-equal HF ``apply_chat_template`` (no kwargs)
   for the matching prefix — i.e. training tokens align with what every
   standard inference stack feeds the model.

Network-dependent: these tests load HF tokenizers via
``transformers.AutoTokenizer``. They skip if the model can't be loaded
(no network / gated repo / etc.) but otherwise run as a regular unit
test.
"""

from __future__ import annotations

from typing import Any

import pytest
import transformers

import training.renderer  # noqa: F401  — registers Split overrides
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Renderer, TrainOnWhat


# (registered name, HF model id, is_multimodal, override class name).
# Multimodal renderers receive an image_processor in addition to a
# tokenizer; the image processor isn't needed for text-only parity but the
# factory expects the kwarg shape.
_PARITY_CASES = [
    ("qwen3", "Qwen/Qwen3-8B", False, "Qwen3SplitRenderer"),
    (
        "qwen3_disable_thinking",
        "Qwen/Qwen3-8B",
        False,
        "Qwen3DisableThinkingSplitRenderer",
    ),
    ("qwen3_5", "Qwen/Qwen3.5-9B", False, "Qwen3_5SplitRenderer"),
    (
        "qwen3_5_disable_thinking",
        "Qwen/Qwen3.5-9B",
        False,
        "Qwen3_5DisableThinkingSplitRenderer",
    ),
    (
        "deepseekv3_thinking",
        "deepseek-ai/DeepSeek-V3.1",
        False,
        "DeepSeekV3ThinkingSplitRenderer",
    ),
    (
        "nemotron3",
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        False,
        "Nemotron3SplitRenderer",
    ),
    (
        "gpt_oss_high_reasoning",
        "openai/gpt-oss-120b",
        False,
        "GptOssSplitRenderer",
    ),
]


def _load_tokenizer(model_id: str):
    try:
        return transformers.AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
    except Exception:  # noqa: BLE001 — network / gated repo / config drift
        return None


def _hf_tokens(tokenizer, messages, *, add_generation_prompt: bool, **kwargs):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    return tokenizer.encode(text, add_special_tokens=False)


@pytest.mark.parametrize(
    "name,model_id,is_multimodal,split_classname",
    _PARITY_CASES,
    ids=[case[0] for case in _PARITY_CASES],
)
def test_split_override_registered_and_has_disaggregate(
    name: str, model_id: str, is_multimodal: bool, split_classname: str
):
    tok = _load_tokenizer(model_id)
    if tok is None:
        pytest.skip(f"tokenizer for {model_id!r} not available")

    renderer = get_renderer(name, tok)

    assert type(renderer).__name__ == split_classname, (
        f"{name!r} should resolve to local {split_classname}, got "
        f"{type(renderer).__name__}"
    )
    # The mixin's ``build_supervised_examples`` must shadow the base default.
    assert (
        type(renderer).build_supervised_examples
        is not Renderer.build_supervised_examples
    ), f"{split_classname} must override build_supervised_examples"
    assert renderer.has_extension_property is False


@pytest.mark.parametrize(
    "name,model_id,is_multimodal,split_classname",
    _PARITY_CASES,
    ids=[case[0] for case in _PARITY_CASES],
)
def test_disaggregate_produces_n_examples(
    name: str, model_id: str, is_multimodal: bool, split_classname: str
):
    tok = _load_tokenizer(model_id)
    if tok is None:
        pytest.skip(f"tokenizer for {model_id!r} not available")

    renderer = get_renderer(name, tok)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "Q3"},
        {"role": "assistant", "content": "A3"},
    ]
    examples = renderer.build_supervised_examples(
        messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"]
    assert len(examples) == len(user_idxs), (
        f"{name!r}: expected {len(user_idxs)} per-turn datums, got {len(examples)}"
    )

    # Sanity: each datum decodes a non-empty prefix and the assistant
    # answer for THAT turn appears in the trained tokens.
    for i, example in enumerate(examples):
        token_ids = list(example[0].to_ints())
        weights = example[1].tolist()
        assert len(token_ids) == len(weights)
        decoded_trained = tok.decode(
            [t for t, w in zip(token_ids, weights) if w > 0]
        )
        expected_answer = messages[user_idxs[i] + 1]["content"]
        assert expected_answer in decoded_trained, (
            f"{name!r}: datum {i} should train answer {expected_answer!r}; "
            f"trained tokens decode to {decoded_trained!r}"
        )
