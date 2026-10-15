"""Post-removal gates for Fireworks renderer/model-routing ownership."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import training.renderer as fireworks_renderers
import training.renderer._model_info as native_model_info
import training.renderer._registry as native_renderers
import training.renderer.model_info as fireworks_model_info


class _CharacterTokenizer:
    """Small reversible tokenizer for dependency-free renderer parity checks."""

    name_or_path = "characterization/tokenizer"
    bos_token = "<B>"
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, tokens: list[int], **_: Any) -> str:
        return "".join(chr(token) for token in tokens)


_BASELINE_CUSTOM_RENDERERS = (
    "deepseek_v4",
    "deepseek_v4_disable_thinking",
    "deepseek_v41",
    "gemma4",
    "gemma4_thinking",
    "glm5",
    "glm5_interleaved",
    "glm5_preserve_thinking",
    "glm_moe_dsa",
    "glm_moe_dsa_interleaved",
    "glm_moe_dsa_preserve_thinking",
    "glm53",
    "glm53_interleaved",
    "glm53_preserve_thinking",
    "glm53_flash",
    "glm53_flash_interleaved",
    "glm53_flash_preserve_thinking",
    "kimi_k25_interleaved",
    "kimi_k26_interleaved",
    "kimi_k26_preserve_thinking",
    "kimi_k27_code",
    "kimi_k27_code_preserved",
    "kimi_k3",
    "kimi_k3_disable_thinking",
    "minimax_m2",
    "minimax_m3",
    "mistral",
    "muse_glimmer",
    "qwen2_5",
    "qwen3",
    "qwen3_disable_thinking",
    "qwen3_vl",
    "qwen3_vl_instruct",
    "qwen3_5",
    "qwen3_5_disable_thinking",
    "qwen3_6",
    "qwen3_6_disable_thinking",
    "qwen3_6_preserve_thinking",
    "qwen3_5_interleaved",
    "qwen3_5_disable_thinking_interleaved",
    "qwen3_6_interleaved",
    "qwen3_6_disable_thinking_interleaved",
    "qwen3_6_preserved",
    "qwen3_8",
    "qwen3_8_interleaved",
    "qwen3_8_disable_thinking_interleaved",
    "qwen3_8_preserved",
    "deepseekv3_thinking",
    "kimi_k25",
    "nemotron3",
    "nemotron3_interleaved",
    "nemotron3_low_thinking",
    "nemotron3_disable_thinking",
    "nemotron3_preserve_thinking",
    "nemotron3_preserved",
    "nemotron3_ultra",
    "nemotron3_ultra_interleaved",
    "nemotron3_ultra_medium_thinking",
    "nemotron3_ultra_disable_thinking",
    "nemotron3_ultra_preserve_thinking",
    "nemotron3_ultra_preserved",
    "gpt_oss_no_sysprompt",
    "gpt_oss_low_reasoning",
    "gpt_oss_medium_reasoning",
    "gpt_oss_high_reasoning",
)


def test_custom_registry_matches_pre_cutover_snapshot() -> None:
    assert tuple(fireworks_renderers.get_registered_renderer_names()) == _BASELINE_CUSTOM_RENDERERS


def test_role_colon_runtime_contract() -> None:
    tokenizer = _CharacterTokenizer()
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "2 + 2?"},
        {"role": "assistant", "content": "4"},
    ]
    vendored = native_renderers.get_renderer("role_colon", tokenizer)
    fireworks = fireworks_renderers.get_renderer("role_colon", tokenizer)

    assert fireworks.get_stop_sequences() == vendored.get_stop_sequences()
    assert (
        fireworks.build_generation_prompt(messages).to_ints() == vendored.build_generation_prompt(messages).to_ints()
    )

    vendored_input, vendored_weights = vendored.build_supervised_example(
        messages, native_renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    fireworks_input, fireworks_weights = fireworks.build_supervised_example(
        messages, fireworks_renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    assert fireworks_input.to_ints() == vendored_input.to_ints()
    assert fireworks_weights.tolist() == vendored_weights.tolist()

    response = tokenizer.encode(" four\n\nUser:")
    vendored_message, vendored_termination = vendored.parse_response(response)
    fireworks_message, fireworks_termination = fireworks.parse_response(response)
    assert fireworks_message == vendored_message
    assert fireworks_termination.value == vendored_termination.value


def test_nemotron35_lightning_recommends_compact_thinking_renderers() -> None:
    assert fireworks_model_info.get_recommended_renderer_names(
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    ) == ["nemotron3_ultra", "nemotron3_ultra_disable_thinking"]


@pytest.mark.parametrize("name", ["not-a-renderer", "", "qwen-unknown"])
def test_unknown_renderer_error_contract(name: str) -> None:
    tokenizer = _CharacterTokenizer()
    with pytest.raises(ValueError) as vendored_error:
        native_renderers.get_renderer(name, tokenizer)
    with pytest.raises(ValueError) as fireworks_error:
        fireworks_renderers.get_renderer(name, tokenizer)
    assert str(fireworks_error.value) == str(vendored_error.value)
