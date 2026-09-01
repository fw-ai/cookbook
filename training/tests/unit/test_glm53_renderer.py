"""Verify GLM-5.3 and GLM-5.3-Flash against pinned HF templates."""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
import tinker
import transformers
from PIL import Image
from tinker_cookbook.exceptions import RendererError

import training.renderer.glm5  # noqa: F401 - registers glm53
from training.renderer import get_renderer
from training.renderer.glm5 import Glm53FlashImageTokenCounter
from training.utils.supervised import (
    build_tool_prefixed_messages,
    normalize_messages,
)
from training._vendor.tinker_cookbook_0_4_3.renderers.base import TrainOnWhat


_TOKENIZER = "zai-org/GLM-5.3"
_TOKENIZER_REVISION = "935644c05e76fc198714f4cca449fd8b970ff6d7"
_FLASH_TOKENIZER = "zai-org/GLM-5.3-Flash"
_FLASH_TOKENIZER_REVISION = "03eb5366286afd40d2221b1d9c63a6dd1ba4832e"
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "time",
            "description": "Get time",
            "parameters": {
                "type": "object",
                "properties": {"zone": {"type": "string"}},
                "required": ["zone"],
            },
        },
    },
]


def _load_tokenizer(
    model_name: str,
    revision: str,
    *,
    path_env: str,
) -> transformers.PreTrainedTokenizerBase | None:
    tokenizer_source = os.environ.get(path_env, model_name)
    try:
        return transformers.AutoTokenizer.from_pretrained(
            tokenizer_source,
            revision=revision if tokenizer_source == model_name else None,
            trust_remote_code=True,
        )
    except Exception:  # noqa: BLE001 - network/auth/cache availability
        return None


@pytest.fixture(scope="module")
def tokenizer():
    tok = _load_tokenizer(
        _TOKENIZER,
        _TOKENIZER_REVISION,
        path_env="GLM53_TOKENIZER_PATH",
    )
    if tok is None:
        pytest.skip(
            f"GLM-5.3 tokenizer not available: "
            f"{_TOKENIZER!r}@{_TOKENIZER_REVISION}"
        )
    if not getattr(tok, "chat_template", None):
        pytest.skip("Loaded GLM-5.3 tokenizer has no chat template.")
    return tok


@pytest.fixture(scope="module")
def flash_tokenizer():
    tok = _load_tokenizer(
        _FLASH_TOKENIZER,
        _FLASH_TOKENIZER_REVISION,
        path_env="GLM53_FLASH_TOKENIZER_PATH",
    )
    if tok is None:
        pytest.skip(
            f"GLM-5.3-Flash tokenizer not available: "
            f"{_FLASH_TOKENIZER!r}@{_FLASH_TOKENIZER_REVISION}"
        )
    if not getattr(tok, "chat_template", None):
        pytest.skip("Loaded GLM-5.3-Flash tokenizer has no chat template.")
    return tok


@pytest.fixture(scope="module")
def renderer(tokenizer):
    return get_renderer("glm53", tokenizer)


@pytest.fixture(scope="module")
def flash_renderer(flash_tokenizer: Any) -> Any:
    return get_renderer("glm53_flash", flash_tokenizer)


@pytest.fixture(scope="module")
def image_renderer(flash_tokenizer: Any) -> Any:
    return get_renderer(
        "glm53_flash",
        flash_tokenizer,
        image_processor=Glm53FlashImageTokenCounter(),
    )


def _hf_tokens(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        **kwargs,
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


def _generation_tokens(
    renderer,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> list[int]:
    normalized = build_tool_prefixed_messages(
        messages,
        renderer=renderer,
        tools=tools,
    )
    return list(renderer.build_generation_prompt(normalized).to_ints())


def _generation_input(
    renderer: Any,
    messages: list[dict[str, Any]],
) -> tinker.ModelInput:
    return renderer.build_generation_prompt(normalize_messages(messages))


def _expand_image_chunks(
    tokenizer: Any,
    model_input: tinker.ModelInput,
) -> list[int]:
    begin_image = tokenizer.convert_tokens_to_ids("<|begin_of_image|>")
    image = tokenizer.convert_tokens_to_ids("<|image|>")
    end_image = tokenizer.convert_tokens_to_ids("<|end_of_image|>")
    expanded: list[int] = []
    for chunk in model_input.chunks:
        if isinstance(chunk, tinker.types.EncodedTextChunk):
            expanded.extend(int(token) for token in chunk.tokens)
        elif isinstance(chunk, tinker.types.ImageChunk):
            expanded.append(begin_image)
            expanded.extend([image] * int(chunk.expected_tokens))
            expanded.append(end_image)
        else:  # pragma: no cover - renderer emits only these two chunk types
            raise TypeError(type(chunk))
    return expanded


def _expand_hf_image_tokens(
    tokenizer: Any,
    tokens: list[int],
    expected_tokens: list[int],
) -> list[int]:
    image = tokenizer.convert_tokens_to_ids("<|image|>")
    expanded: list[int] = []
    image_index = 0
    for token in tokens:
        if token == image:
            assert image_index < len(expected_tokens)
            expanded.extend([image] * expected_tokens[image_index])
            image_index += 1
        else:
            expanded.append(token)
    assert image_index == len(expected_tokens)
    return expanded


def _supervised_tokens(renderer, messages: list[dict[str, Any]]) -> list[int]:
    model_input, _ = renderer.build_supervised_example(normalize_messages(messages))
    return list(model_input.to_ints())


def _without_terminal_role_stop(tokenizer, tokens: list[int]) -> list[int]:
    user = tokenizer.encode("<|user|>", add_special_tokens=False)
    assert len(user) == 1
    assert tokens[-1] == user[0]
    return tokens[:-1]


def _parallel_tool_messages(
    results: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "weather and time"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": {"city": "Paris"},
                    },
                },
                {
                    "id": "call_time",
                    "type": "function",
                    "function": {
                        "name": "time",
                        "arguments": {"zone": "UTC"},
                    },
                },
            ],
        },
        *[
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
            for tool_call_id, content in results
        ],
    ]


def test_registered_glm53_renderer(tokenizer, renderer):
    assert type(renderer).__name__ == "GLM53Renderer"
    assert renderer.has_extension_property is True
    assert renderer.supports_per_message_rendering is False


def test_registered_glm53_flash_renderer(flash_tokenizer, flash_renderer):
    assert type(flash_renderer).__name__ == "GLM53FlashRenderer"
    assert flash_renderer.has_extension_property is True
    assert flash_renderer.supports_per_message_rendering is False


def test_flash_and_text_only_templates_share_text_wire_contract(
    tokenizer: Any,
    flash_tokenizer: Any,
) -> None:
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "weather and time"},
        {
            "role": "assistant",
            "reasoning_content": "I should call both tools.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": {"city": "Paris"},
                    },
                },
                {
                    "id": "call_time",
                    "type": "function",
                    "function": {"name": "time", "arguments": {"zone": "UTC"}},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_time", "content": "12:00"},
        {"role": "tool", "tool_call_id": "call_weather", "content": "sunny"},
    ]
    assert _hf_tokens(
        tokenizer,
        messages,
        add_generation_prompt=True,
        tools=_TOOLS,
    ) == _hf_tokens(
        flash_tokenizer,
        messages,
        add_generation_prompt=True,
        tools=_TOOLS,
    )


def test_flash_and_text_only_templates_intentionally_diverge_for_images(
    tokenizer: Any,
    flash_tokenizer: Any,
) -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect "},
                {"type": "image", "image": "unused-by-template"},
            ],
        }
    ]
    text_only = tokenizer.apply_chat_template(messages, tokenize=False)
    flash = flash_tokenizer.apply_chat_template(messages, tokenize=False)
    assert "unable to process this image" in text_only
    assert "<|begin_of_image|><|image|><|end_of_image|>" in flash


@pytest.mark.parametrize(
    ("height", "width", "expected_patches", "expected_tokens"),
    [
        (14, 14, 64, 16),
        (64, 128, 72, 18),
        (224, 224, 256, 64),
        (720, 1280, 4784, 1196),
        (10_000, 10_000, 31_684, 7921),
    ],
)
def test_image_token_counter_matches_hf_geometry(
    height: int,
    width: int,
    expected_patches: int,
    expected_tokens: int,
) -> None:
    counter = Glm53FlashImageTokenCounter()
    patches = counter.get_number_of_image_patches(height, width)
    assert patches == expected_patches
    assert patches // counter.merge_size**2 == expected_tokens


def test_image_token_counter_loads_nested_processor_config(tmp_path: Any) -> None:
    processor_config = {
        "image_processor": {
            "image_processor_type": "Glm5NextImageProcessor",
            "patch_size": 7,
            "temporal_patch_size": 2,
            "merge_size": 4,
            "min_image_tokens": 8,
            "max_image_tokens": 512,
        },
        "processor_class": "Glm5NextProcessor",
    }
    (tmp_path / "processor_config.json").write_text(
        json.dumps(processor_config),
        encoding="utf-8",
    )
    counter = Glm53FlashImageTokenCounter.from_pretrained(str(tmp_path))
    assert counter.patch_size == 7
    assert counter.temporal_patch_size == 2
    assert counter.merge_size == 4
    assert counter.min_image_tokens == 8
    assert counter.max_image_tokens == 512


def test_generation_prompt_preserves_interleaved_images_with_hf_parity(
    flash_tokenizer: Any,
    image_renderer: Any,
) -> None:
    first = Image.new("RGB", (224, 224), "red")
    second = Image.new("RGB", (128, 64), "blue")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first="},
                {"type": "image", "image": first},
                {"type": "text", "text": ";second="},
                {"type": "image", "image": second},
                {"type": "text", "text": ";compare"},
            ],
        }
    ]

    model_input = _generation_input(image_renderer, messages)
    image_chunks = [
        chunk
        for chunk in model_input.chunks
        if isinstance(chunk, tinker.types.ImageChunk)
    ]
    assert [chunk.expected_tokens for chunk in image_chunks] == [64, 18]
    assert _expand_image_chunks(
        flash_tokenizer, model_input
    ) == _expand_hf_image_tokens(
        flash_tokenizer,
        _hf_tokens(flash_tokenizer, messages, add_generation_prompt=True),
        [64, 18],
    )


def test_tool_image_preserves_hf_wrapper_and_is_context_only(
    flash_tokenizer: Any,
    image_renderer: Any,
) -> None:
    image = Image.new("RGB", (224, 224), "green")
    messages = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_capture",
                    "type": "function",
                    "function": {"name": "capture", "arguments": {}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_capture",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image", "image": image},
                {"type": "text", "text": "after"},
            ],
        },
        {"role": "assistant", "content": "done"},
    ]

    model_input, weights = image_renderer.build_supervised_example(
        normalize_messages(messages),
        train_on_what=TrainOnWhat.ALL_TOKENS,
    )
    image_chunks = [
        chunk
        for chunk in model_input.chunks
        if isinstance(chunk, tinker.types.ImageChunk)
    ]
    assert [chunk.expected_tokens for chunk in image_chunks] == [64]

    cursor = 0
    for chunk in model_input.chunks:
        chunk_weights = weights[cursor : cursor + chunk.length]
        if isinstance(chunk, tinker.types.ImageChunk):
            assert chunk_weights.eq(0).all()
        cursor += chunk.length
    assert cursor == len(weights)

    ours = _expand_image_chunks(flash_tokenizer, model_input)
    assert ours[-1] == flash_tokenizer.convert_tokens_to_ids("<|user|>")
    assert ours[:-1] == _expand_hf_image_tokens(
        flash_tokenizer,
        _hf_tokens(flash_tokenizer, messages, add_generation_prompt=False),
        [64],
    )


def test_image_content_requires_processor(
    flash_renderer: Any,
) -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (32, 32))}
            ],
        }
    ]
    with pytest.raises(RendererError, match="requires an image processor"):
        _generation_input(flash_renderer, messages)


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "hello"}],
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "<think>old</think>a1"},
            {"role": "user", "content": "q2"},
        ],
    ],
)
def test_generation_prompt_matches_hf(tokenizer, renderer, messages):
    assert _generation_tokens(renderer, messages) == _hf_tokens(
        tokenizer,
        messages,
        add_generation_prompt=True,
    )


def test_enable_thinking_false_is_ignored_like_hf(tokenizer, renderer):
    messages = [{"role": "user", "content": "hello"}]
    assert _generation_tokens(renderer, messages) == _hf_tokens(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def test_default_preserves_historical_reasoning(tokenizer, renderer):
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "<think>old</think>a1"},
        {"role": "user", "content": "q2"},
    ]
    decoded = tokenizer.decode(_generation_tokens(renderer, messages))
    assert "<think>old</think>a1" in decoded


def test_supervised_example_matches_hf_modulo_training_stop(tokenizer, renderer):
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "reasoning_content": "work", "content": "a1"},
    ]
    ours = _without_terminal_role_stop(
        tokenizer,
        _supervised_tokens(renderer, messages),
    )
    assert ours == _hf_tokens(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )


def test_parallel_tool_results_follow_call_order(tokenizer, renderer):
    messages = _parallel_tool_messages(
        [("call_time", "12:00"), ("call_weather", "sunny")]
    )
    ours = _generation_tokens(renderer, messages, tools=_TOOLS)
    assert ours == _hf_tokens(
        tokenizer,
        messages,
        add_generation_prompt=True,
        tools=_TOOLS,
    )
    rendered = tokenizer.decode(ours)
    assert rendered.index("sunny") < rendered.index("12:00")


def test_duplicate_tool_result_ids_fall_back_to_source_order(tokenizer, renderer):
    messages = _parallel_tool_messages(
        [("call_time", "first"), ("call_time", "second")]
    )
    ours = _generation_tokens(renderer, messages, tools=_TOOLS)
    assert ours == _hf_tokens(
        tokenizer,
        messages,
        add_generation_prompt=True,
        tools=_TOOLS,
    )
    rendered = tokenizer.decode(ours)
    assert rendered.index("first") < rendered.index("second")
