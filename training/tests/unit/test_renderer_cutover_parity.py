"""Characterization gates for the renderer/model-routing ownership cutover."""

from __future__ import annotations

import dataclasses
import importlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

import training.renderer as fireworks_renderers
import training.renderer.model_info as fireworks_model_info

# The pinned package is loaded dynamically only as the temporary differential
# oracle. Static imports from its renderer/routing surface are forbidden below
# and this oracle goes away when the final package dependency is removed.
legacy_model_info = importlib.import_module("tinker_cookbook.model_info")
legacy_renderers = importlib.import_module("tinker_cookbook.renderers")

_DEPENDENCY_OWNED_IMPORT = re.compile(
    r"(?m)^[ \t]*(?:"
    r"from[ \t]+tinker_cookbook\.(?:model_info|renderers)(?:\.[A-Za-z_]\w*)*[ \t]+import\b"
    r"|import[ \t]+tinker_cookbook\.(?:model_info|renderers)(?:\.[A-Za-z_]\w*)*\b"
    r"|from[ \t]+tinker_cookbook[ \t]+import[ \t]+[^#\n]*\b(?:model_info|renderers)\b"
    r")"
)


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


def test_role_colon_runtime_behavior_matches_pinned_package() -> None:
    tokenizer = _CharacterTokenizer()
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "2 + 2?"},
        {"role": "assistant", "content": "4"},
    ]
    legacy = legacy_renderers.get_renderer("role_colon", tokenizer)
    fireworks = fireworks_renderers.get_renderer("role_colon", tokenizer)

    assert fireworks.get_stop_sequences() == legacy.get_stop_sequences()
    assert fireworks.build_generation_prompt(messages).to_ints() == legacy.build_generation_prompt(
        messages
    ).to_ints()

    legacy_input, legacy_weights = legacy.build_supervised_example(
        messages, legacy_renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    fireworks_input, fireworks_weights = fireworks.build_supervised_example(
        messages, fireworks_renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    assert fireworks_input.to_ints() == legacy_input.to_ints()
    assert fireworks_weights.tolist() == legacy_weights.tolist()

    response = tokenizer.encode(" four\n\nUser:")
    legacy_message, legacy_termination = legacy.parse_response(response)
    fireworks_message, fireworks_termination = fireworks.parse_response(response)
    assert fireworks_message == legacy_message
    assert fireworks_termination.value == legacy_termination.value


def test_model_routing_matches_pinned_package() -> None:
    legacy_maps = (
        legacy_model_info.get_llama_info(),
        legacy_model_info.get_qwen_info(),
        legacy_model_info.get_deepseek_info(),
        legacy_model_info.get_gpt_oss_info(),
        legacy_model_info.get_moonshot_info(),
        legacy_model_info.get_nvidia_info(),
    )
    fireworks_maps = (
        fireworks_model_info.get_llama_info(),
        fireworks_model_info.get_qwen_info(),
        fireworks_model_info.get_deepseek_info(),
        fireworks_model_info.get_gpt_oss_info(),
        fireworks_model_info.get_moonshot_info(),
        fireworks_model_info.get_nvidia_info(),
    )
    assert [
        {name: dataclasses.asdict(attributes) for name, attributes in model_map.items()}
        for model_map in fireworks_maps
    ] == [
        {name: dataclasses.asdict(attributes) for name, attributes in model_map.items()}
        for model_map in legacy_maps
    ]


def test_no_consumer_imports_dependency_owned_renderer_or_routing_modules() -> None:
    repository_root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    tracked_source_files = subprocess.run(
        ["git", "ls-files", "*.py", "*.ipynb"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    forbidden: list[str] = []
    for relative_path in tracked_source_files:
        if "training/_vendor/tinker_cookbook_0_4_3/" in relative_path:
            continue
        path = repository_root / relative_path
        source = path.read_text(encoding="utf-8")
        if path.suffix == ".ipynb":
            notebook = json.loads(source)
            source = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook.get("cells", [])
                if cell.get("cell_type") == "code"
            )
        for match in _DEPENDENCY_OWNED_IMPORT.finditer(source):
            line_number = source.count("\n", 0, match.start()) + 1
            forbidden.append(f"{relative_path}:{line_number}")

    assert not forbidden, "dependency-owned renderer/routing imports remain: " + ", ".join(forbidden)


@pytest.mark.parametrize("name", ["not-a-renderer", "", "qwen-unknown"])
def test_unknown_renderer_error_shape_matches_pinned_package(name: str) -> None:
    tokenizer = _CharacterTokenizer()
    with pytest.raises(ValueError) as legacy_error:
        legacy_renderers.get_renderer(name, tokenizer)
    with pytest.raises(ValueError) as fireworks_error:
        fireworks_renderers.get_renderer(name, tokenizer)
    assert str(fireworks_error.value) == str(legacy_error.value)
