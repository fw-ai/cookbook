"""High-value DeepSeek-V4.1 supervised-renderer contracts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import tinker
from PIL import Image
from training.renderer.deepseek_v41 import (
    DeepseekV41ImageTokenCounter,
    DeepseekV41Renderer,
)
from training._vendor.tinker_cookbook_0_4_3.renderers.base import TrainOnWhat
from training.utils.supervised import (
    build_tool_prefixed_messages,
    normalize_messages,
    render_messages_to_datums,
    resolve_renderer_name,
)

_EOS = "<｜end▁of▁sentence｜>"
_EOS_ID = 0x110000


class _Tokenizer:
    name_or_path = "deepseek-v41-test"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [_EOS_ID] if text == _EOS else [ord(character) for character in text]

    def decode(self, tokens: list[int], **_: object) -> str:
        return "".join(
            _EOS if int(token) == _EOS_ID else chr(int(token)) for token in tokens
        )


class _MergingTokenizer(_Tokenizer):
    """Merges separator suffixes into one token, as the real BPE vocabulary does."""

    _MERGES = {".\n\n": 0x110001, ">\n\n": 0x110002}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text == _EOS:
            return [_EOS_ID]
        tokens, index = [], 0
        while index < len(text):
            merged = next(
                (
                    item
                    for item in self._MERGES.items()
                    if text.startswith(item[0], index)
                ),
                None,
            )
            tokens.append(merged[1] if merged else ord(text[index]))
            index += len(merged[0]) if merged else 1
        return tokens


def _publisher_encoder():
    root = Path(__file__).resolve().parents[5]
    path = root / "py/fireworks/models/deepseek4_next/reference/encoding/encoding.py"
    spec = importlib.util.spec_from_file_location("deepseek_v41_oracle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _renderer(*, images: bool = False) -> DeepseekV41Renderer:
    counter = (
        DeepseekV41ImageTokenCounter(
            patch_size=2,
            downsample_ratio=2,
            max_image_tokens=64,
            min_pixels=1,
            max_wh_ratio=None,
        )
        if images
        else None
    )
    return DeepseekV41Renderer(_Tokenizer(), image_processor=counter)


def _text(model_input: tinker.ModelInput) -> str:
    return _Tokenizer().decode(list(model_input.to_ints()))


@pytest.mark.parametrize(
    "model",
    [
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "DeepSeekV4.1",
        "custom/deepseekv41-finetune",
        "accounts/fireworks/models/deepseek-v41-flash",
    ],
)
def test_v41_aliases_route_before_v4(model: str) -> None:
    assert resolve_renderer_name(model) == "deepseek_v41"


def test_default_effort_initial_and_mid_system_match_publisher() -> None:
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Be "},
                {"type": "text", "text": "precise."},
            ],
            "response_format": {"type": "json_object"},
        },
        {"role": "user", "content": "Two plus two?"},
        {
            "role": "assistant",
            "reasoning_content": "Add.",
            "content": "Four.",
        },
        {"role": "system", "content": "Now answer briefly."},
    ]
    expected = _publisher_encoder().encode_messages(
        messages,
        thinking_mode="thinking",
    )
    actual = _text(_renderer().build_generation_prompt(normalize_messages(messages)))
    assert actual == expected
    assert _renderer().get_stop_sequences() == [_EOS_ID]


def test_tool_order_and_mid_system_last_turn_mask() -> None:
    tools = [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
        for name in ("first", "second")
    ]
    messages = [
        {"role": "system", "content": "Use tools.", "tools": tools},
        {"role": "user", "content": "Run both."},
        {
            "role": "assistant",
            "reasoning_content": "OLD_REASONING",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(json.dumps({"city": "Paris"}))
                        if name == "first"
                        else "{}",
                    },
                }
                for call_id, name in (("a", "first"), ("b", "second"))
            ],
        },
        {"role": "tool", "tool_call_id": "b", "content": "SECOND_RESULT"},
        {"role": "tool", "tool_call_id": "a", "content": "FIRST_RESULT"},
        {"role": "system", "content": "Answer now."},
        {
            "role": "assistant",
            "reasoning_content": "NEW_REASONING",
            "content": "Done.",
        },
    ]
    renderer = _renderer()
    model_input, weights = renderer.build_supervised_example(
        normalize_messages(messages),
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    text = _text(model_input)
    assert text == _publisher_encoder().encode_messages(
        messages, thinking_mode="thinking"
    )
    trained = renderer.tokenizer.decode(
        [
            token
            for token, weight in zip(
                model_input.to_ints(), weights.tolist(), strict=True
            )
            if weight
        ]
    )
    assert "OLD_REASONING" not in trained
    assert "NEW_REASONING" in trained and "Done." in trained
    assert "FIRST_RESULT" not in trained and "SECOND_RESULT" not in trained


@pytest.mark.parametrize(
    "messages",
    [
        [
            {"role": "user", "content": "Draft."},
            {"role": "assistant", "reasoning_content": "First.", "content": "A."},
            {"role": "assistant", "reasoning_content": "Retry.", "content": "B."},
        ],
        [
            {"role": "system", "content": "System only."},
            {"role": "assistant", "reasoning_content": "Think.", "content": "Answer."},
        ],
        [{"role": "assistant", "reasoning_content": "Think.", "content": "Answer."}],
    ],
)
def test_assistant_header_matches_publisher(
    messages: list[dict[str, object]],
) -> None:
    actual = _text(
        _renderer().build_supervised_example(normalize_messages(messages))[0]
    )
    expected = _publisher_encoder().encode_messages(messages, thinking_mode="thinking")
    assert actual == expected


def test_embedded_think_is_not_duplicated() -> None:
    messages = normalize_messages(
        [
            {"role": "user", "content": "Answer."},
            {"role": "assistant", "content": "<think>ONLY_ONCE</think>Visible."},
        ]
    )
    rendered = _text(_renderer().build_supervised_example(messages)[0])
    assert rendered.count("ONLY_ONCE") == 1
    assert "<think>ONLY_ONCE</think>Visible." in rendered


@pytest.mark.parametrize(
    "raw_messages",
    [
        [
            {"role": "system", "content": "Initial instruction."},
            {"role": "assistant", "content": "One."},
            {"role": "system", "content": "New instruction."},
            {"role": "assistant", "content": "Two."},
            {"role": "system", "content": "Trailing instruction."},
        ],
        [
            {"role": "system", "content": "Initial instruction."},
            {"role": "assistant", "content": "One."},
            {"role": "system", "content": "Trailing instruction."},
        ],
    ],
)
def test_mid_system_is_a_supervised_turn_boundary(
    raw_messages: list[dict[str, object]],
) -> None:
    messages = normalize_messages(raw_messages)
    rendered = render_messages_to_datums(
        messages,
        renderer=_renderer(),
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    assert len(rendered) == sum(
        message["role"] == "assistant" for message in raw_messages
    )
    assert all(sum(item.datum.loss_fn_inputs["weights"].data) > 0 for item in rendered)


def test_top_level_tools_preserve_structured_system_metadata() -> None:
    tools = [
        {
            "type": "function",
            "function": {"name": "search", "parameters": {"type": "object"}},
        }
    ]
    raw_messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Use "},
                {"type": "text", "text": "tools."},
            ],
            "response_format": {"type": "json_object"},
        },
        {"role": "user", "content": "Find it."},
    ]
    prefixed = build_tool_prefixed_messages(
        raw_messages, renderer=_renderer(), tools=tools
    )
    actual = _text(_renderer().build_generation_prompt(prefixed))
    expected_messages = [dict(raw_messages[0], tools=tools), raw_messages[1]]
    expected = _publisher_encoder().encode_messages(
        expected_messages, thinking_mode="thinking"
    )
    assert actual == expected


@pytest.mark.parametrize(
    "message",
    [
        {"role": "system", "content": "<｜deepseek_image｜>"},
        {
            "role": "assistant",
            "reasoning_content": "<｜deepseek_image｜>",
            "content": "answer",
        },
    ],
)
def test_raw_image_placeholder_is_rejected(message: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="image placeholders"):
        _renderer().build_supervised_example(normalize_messages([message]))


def test_images_are_ordered_counted_and_zero_loss() -> None:
    red = Image.new("RGB", (4, 4), "red")
    blue = Image.new("RGB", (4, 4), "blue")
    model_input, weights = _renderer(images=True).build_supervised_example(
        normalize_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first"},
                        {"type": "image", "image": red},
                        {"type": "text", "text": "second"},
                        {"type": "image", "image": blue},
                    ],
                },
                {"role": "assistant", "content": "Done."},
            ]
        )
    )
    images = [
        chunk
        for chunk in model_input.chunks
        if isinstance(chunk, tinker.types.ImageChunk)
    ]
    assert [chunk.expected_tokens for chunk in images] == [4, 4]
    offset = 0
    for chunk in model_input.chunks:
        if isinstance(chunk, tinker.types.ImageChunk):
            assert not weights[offset : offset + chunk.length].any()
        offset += chunk.length


@pytest.mark.parametrize(
    "messages",
    [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look."},
                    {"type": "text", "text": "Here."},
                    {"type": "image", "image": "red"},
                    {"type": "text", "text": "What?"},
                ],
            },
            {"role": "assistant", "content": "Red."},
        ],
        [
            {
                "role": "system",
                "content": "Use tools.",
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": name, "parameters": {"type": "object"}},
                    }
                    for name in ("first", "second")
                ],
            },
            {"role": "user", "content": "Run both."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": name,
                        "type": "function",
                        "function": {"name": name, "arguments": "{}"},
                    }
                    for name in ("first", "second")
                ],
            },
            {"role": "tool", "tool_call_id": "first", "content": "ONE"},
            {"role": "tool", "tool_call_id": "second", "content": "TWO"},
            {"role": "assistant", "content": "Done."},
        ],
    ],
)
def test_block_separators_tokenize_like_the_publisher_prompt(
    messages: list[dict[str, object]],
) -> None:
    image = Image.new("RGB", (4, 4), "red")
    rendered = [
        {
            **message,
            "content": [
                dict(part, image=image) if part["type"] == "image" else part
                for part in message["content"]
            ],
        }
        if isinstance(message["content"], list)
        else message
        for message in messages
    ]
    renderer = DeepseekV41Renderer(
        _MergingTokenizer(),
        image_processor=_renderer(images=True).image_processor,
    )
    model_input, _ = renderer.build_supervised_example(normalize_messages(rendered))
    segments: list[list[int]] = [[]]
    for chunk in model_input.chunks:
        if isinstance(chunk, tinker.types.ImageChunk):
            segments.append([])
        else:
            segments[-1].extend(chunk.tokens)
    publisher = [
        {
            **message,
            "content": [
                {"type": "image", "data": "red"} if part["type"] == "image" else part
                for part in message["content"]
            ],
        }
        if isinstance(message["content"], list)
        else message
        for message in messages
    ]
    prompt = _publisher_encoder().encode_messages(publisher, thinking_mode="thinking")
    assert segments == [
        renderer.tokenizer.encode(part) for part in prompt.split("<｜deepseek_image｜>")
    ]


def test_valid_and_malformed_tool_parsing() -> None:
    renderer = _renderer()

    def assert_unparsed(text: str, error: str) -> None:
        message, _ok = renderer.parse_response(renderer.tokenizer.encode(text + _EOS))
        [unparsed] = message["unparsed_tool_calls"]
        assert unparsed.error == error

    valid = (
        'Reason.</think>Done.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="search">\n'
        '<｜DSML｜ parameter name="q" string="true">fireworks</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>" + _EOS
    )
    message, ok = renderer.parse_response(renderer.tokenizer.encode(valid))
    assert ok
    assert message["tool_calls"][0].function.name == "search"
    assert json.loads(message["tool_calls"][0].function.arguments) == {"q": "fireworks"}

    assert_unparsed("Done.\n\n<｜DSML｜ calls>\nbroken", "Malformed tool_calls block")
    assert_unparsed(
        "Done.\n\n<｜DSML｜ calls>\nbroken\n</｜DSML｜ calls>",
        "Malformed tool_calls body",
    )
    invoke = 'Done.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="search">\n{}\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'
    assert_unparsed(
        invoke.format("<｜DSML｜ parameter broken>"),
        "Malformed tool parameter",
    )
    assert_unparsed(
        invoke.format(
            '<｜DSML｜ parameter name="q" string="true">first</｜DSML｜ parameter>\n'
            '<｜DSML｜ parameter name="q" string="true">second</｜DSML｜ parameter>'
        ),
        "Duplicate tool parameter: q",
    )
