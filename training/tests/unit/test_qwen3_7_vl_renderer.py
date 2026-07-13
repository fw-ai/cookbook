"""Exact-checkpoint Qwen3.7 VL renderer parity and loss-mask coverage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import tinker
from PIL import Image
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import TrainOnWhat
from tinker_cookbook.tokenizer_utils import get_tokenizer

import training.renderer  # noqa: F401 - registers cookbook-local renderers
from training.utils.supervised import normalize_messages, prepare_messages_with_tools


_MODEL = Path("/shared/qwen3p7-plus-vl-think/hf")
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_region",
            "description": "Inspect a named region in the image.",
            "parameters": {
                "type": "object",
                "properties": {"region": {"type": "string"}},
                "required": ["region"],
            },
        },
    }
]


pytestmark = pytest.mark.skipif(
    not (_MODEL / "tokenizer_config.json").exists(),
    reason="exact qwen3p7-plus-vl-think snapshot is not available",
)


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(str(_MODEL))


@pytest.fixture(scope="module")
def image_processor():
    return get_image_processor(str(_MODEL))


@pytest.fixture(scope="module")
def image() -> Image.Image:
    return Image.new("RGB", (256, 192), color=(42, 117, 193))


def _structural_tokens(model_input: tinker.ModelInput, tokenizer: Any) -> list[int]:
    """Replace each binary image chunk with the template's one image-pad token."""
    image_pad = tokenizer.encode("<|image_pad|>", add_special_tokens=False)
    assert len(image_pad) == 1

    tokens: list[int] = []
    for chunk in model_input.chunks:
        if isinstance(chunk, tinker.types.EncodedTextChunk):
            tokens.extend(chunk.tokens)
        else:
            assert isinstance(chunk, tinker.types.ImageChunk)
            assert chunk.expected_tokens > 0
            tokens.extend(image_pad)
    return tokens


def _assert_generation_parity(
    *,
    renderer_name: str,
    tokenizer: Any,
    image_processor: Any,
    messages: list[dict],
    tools: list[dict] | None = None,
    **template_kwargs: Any,
) -> tinker.ModelInput:
    renderer = get_renderer(
        renderer_name,
        tokenizer,
        image_processor=image_processor,
    )
    normalized = prepare_messages_with_tools(
        messages,
        renderer=renderer,
        tools=tools,
    )
    model_input = renderer.build_generation_prompt(normalized, role="assistant")
    actual = _structural_tokens(model_input, tokenizer)
    hf_result = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        **template_kwargs,
    )
    expected = list(
        hf_result.input_ids if hasattr(hf_result, "input_ids") else hf_result
    )
    assert actual == expected
    assert (
        tokenizer.decode(
            actual,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).encode()
        == tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        ).encode()
    )
    return model_input


@pytest.mark.parametrize(
    ("renderer_name", "template_kwargs"),
    [
        ("qwen3_7_vl", {}),
        ("qwen3_7_vl_disable_thinking", {"enable_thinking": False}),
    ],
)
def test_vl_single_turn_thinking_modes_are_byte_identical(
    renderer_name,
    template_kwargs,
    tokenizer,
    image_processor,
    image,
):
    model_input = _assert_generation_parity(
        renderer_name=renderer_name,
        tokenizer=tokenizer,
        image_processor=image_processor,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What is shown?"},
                ],
            }
        ],
        **template_kwargs,
    )
    image_chunks = [
        chunk
        for chunk in model_input.chunks
        if isinstance(chunk, tinker.types.ImageChunk)
    ]
    assert len(image_chunks) == 1
    assert image_chunks[0].expected_tokens > 0


def test_vl_preserved_thinking_multiturn_and_two_images_are_byte_identical(
    tokenizer,
    image_processor,
    image,
):
    _assert_generation_parity(
        renderer_name="qwen3_7_vl_preserve_thinking",
        tokenizer=tokenizer,
        image_processor=image_processor,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe the color."},
                ],
            },
            {
                "role": "assistant",
                "reasoning_content": "The dominant color is blue.",
                "content": "It is blue.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Is this the same image?"},
                ],
            },
        ],
        preserve_thinking=True,
    )


def test_vl_parallel_tool_calls_and_image_tool_result_are_byte_identical(
    tokenizer,
    image_processor,
    image,
):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Inspect the top and bottom."},
            ],
        },
        {
            "role": "assistant",
            "reasoning_content": "I need to inspect both regions.",
            "content": "Checking both.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "inspect_region",
                        "arguments": {"region": "top"},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "inspect_region",
                        "arguments": {"region": "bottom"},
                    },
                },
            ],
        },
        {"role": "tool", "content": "Top: blue"},
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "  Bottom crop: "},
                {"type": "image", "image": image},
                {"type": "text", "text": "  "},
            ],
        },
    ]
    _assert_generation_parity(
        renderer_name="qwen3_7_vl",
        tokenizer=tokenizer,
        image_processor=image_processor,
        messages=messages,
        tools=_TOOLS,
    )


def test_vl_supervised_chunks_weights_and_closed_template_align(
    tokenizer,
    image_processor,
    image,
):
    renderer = get_renderer(
        "qwen3_7_vl_preserve_thinking",
        tokenizer,
        image_processor=image_processor,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Name the dominant color."},
            ],
        },
        {
            "role": "assistant",
            "reasoning_content": "The pixels are mostly blue.",
            "content": "Blue.",
        },
    ]
    normalized = normalize_messages(messages)
    model_input, weights = renderer.build_supervised_example(
        normalized,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    actual = _structural_tokens(model_input, tokenizer)
    hf_result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        preserve_thinking=True,
    )
    expected = list(
        hf_result.input_ids if hasattr(hf_result, "input_ids") else hf_result
    )
    assert actual == expected
    assert len(weights) == sum(chunk.length for chunk in model_input.chunks)
    assert weights.max().item() == 1
    assert weights.min().item() == 0
    assert any(
        isinstance(chunk, tinker.types.ImageChunk) for chunk in model_input.chunks
    )


def test_vl_rejects_system_images_like_the_checkpoint_template(
    tokenizer,
    image_processor,
    image,
):
    renderer = get_renderer(
        "qwen3_7_vl",
        tokenizer,
        image_processor=image_processor,
    )
    messages = normalize_messages(
        [
            {
                "role": "system",
                "content": [{"type": "image", "image": image}],
            },
            {"role": "user", "content": "Describe it."},
        ]
    )
    with pytest.raises(ValueError, match="System message cannot contain images"):
        renderer.build_generation_prompt(messages)
