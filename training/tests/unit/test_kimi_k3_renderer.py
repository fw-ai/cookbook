"""Release-oracle tests for the Kimi K3 cookbook renderer.

The release tokenizer's Python ``apply_chat_template`` implementation is the
canonical oracle. No Jinja template is synthesized in this suite or renderer.

Scenario coverage was reviewed against these read-only upstream revisions:

* veRL ``8a694930275061f52ebd538c906ef8819af56dbd``: single/multi-turn
  generation, SFT assistant masks, chat-template kwargs, and VLM rows.
* slime ``8f5e2151943e9ed0bbffaed93741d3473abb58d9``: assistant-only
  multi-turn masks, tool trajectories, and accumulated response spans.
* Miles ``803016a4622a7f7f45c26140cb9bd8e016aad217``: pretokenized
  message-boundary cuts, response lengths, partial rollouts, and tools.
* tinker-cookbook ``3e04119ce293a2b6ba5284e35267c9ba6d27c5da``:
  observation/action extension, response parsing, and streaming equivalence.
* AReaL ``d99124ec15102ca2fcd4960cc8beaef3950c2672``: sequential and
  parallel tool-call/result concatenation, including reordered results.

The oracle loads from the public ``moonshotai/Kimi-K3`` repository at an
immutable reviewed revision. ``KIMI_K3_MODEL_PATH`` remains an optional local
or offline override; no release artifact or credential is vendored.
"""

from __future__ import annotations

import base64
import copy
import io
import json
import os
import random
from typing import Any

import pytest
import tinker
from PIL import Image
from training.renderer import TrainOnWhat, get_renderer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from training.renderer.kimi_k3 import (
    STOP_SEQUENCE,
    KimiK3Conversation,
    KimiK3RenderOptions,
    KimiK3Renderer,
    _canonical_image_bytes,
    _native_message,
    _normalize_reasoning,
    _normalize_tool_choice,
)
from training.utils.rl.rollout.renderer import (
    build_multimodal_completions_prompt_token_ids,
    model_input_to_token_ids,
)
from training.utils.supervised import (
    build_tool_prefixed_messages,
    normalize_messages,
    render_messages_to_datums,
    resolve_renderer_name,
)
from training.utils.tokenizers import load_tokenizer

_KIMI_K3_HF_MODEL = "moonshotai/Kimi-K3"
_KIMI_K3_HF_REVISION = "301be1b88c89c0d3a763da6301352cb8fe399e90"
_LOCAL_RELEASE_PATH = os.environ.get("KIMI_K3_MODEL_PATH")
_RELEASE_SOURCE = _LOCAL_RELEASE_PATH or _KIMI_K3_HF_MODEL
_RELEASE_REVISION = None if _LOCAL_RELEASE_PATH else _KIMI_K3_HF_REVISION

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]


def _call(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


PARALLEL_TOOL_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "Use tools and answer briefly."},
    {"role": "user", "content": "Weather in Paris and 6 * 7?"},
    {
        "role": "assistant",
        "reasoning_content": "I need both tools.",
        "content": "",
        "tool_calls": [
            _call("weather-1", "get_weather", {"city": "Paris"}),
            _call("math-1", "calculate", {"expression": "6 * 7"}),
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "math-1",
        "content": json.dumps({"result": 42}),
    },
    {
        "role": "tool",
        "tool_call_id": "weather-1",
        "content": json.dumps({"temperature": 21, "unit": "celsius"}),
    },
    {
        "role": "assistant",
        "reasoning_content": "Both results are available.",
        "content": "Paris is 21°C and 6 * 7 is 42.",
    },
    {"role": "user", "content": "Repeat only the number. 🔢"},
    {"role": "assistant", "content": "42"},
]


@pytest.fixture(scope="module")
def tokenizer():
    try:
        loaded = load_tokenizer(_RELEASE_SOURCE, _RELEASE_REVISION, True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        pytest.skip(f"Kimi K3 HF tokenizer unavailable: {exc}")
    assert loaded.chat_template is None
    assert type(loaded).apply_chat_template.__module__.startswith(
        "transformers_modules."
    )
    return loaded


@pytest.fixture(scope="module")
def renderer(tokenizer) -> KimiK3Renderer:
    loaded = get_renderer("kimi_k3", tokenizer)
    assert isinstance(loaded, KimiK3Renderer)
    return loaded


@pytest.fixture(scope="module")
def image_processor():
    load_kwargs: dict[str, Any] = (
        {"local_files_only": True}
        if _LOCAL_RELEASE_PATH
        else {"revision": _KIMI_K3_HF_REVISION}
    )
    try:
        processor_cls = get_class_from_dynamic_module(
            "kimi_k3_vision_processing.KimiK3VisionProcessor",
            _RELEASE_SOURCE,
            **load_kwargs,
        )
        processor = processor_cls.from_pretrained(
            _RELEASE_SOURCE,
            **load_kwargs,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        pytest.skip(f"Kimi K3 HF image processor unavailable: {exc}")
    processor.preserve_image_mode = True
    return processor


@pytest.fixture(scope="module")
def vision_renderer(tokenizer, image_processor) -> KimiK3Renderer:
    return KimiK3Renderer(tokenizer, image_processor=image_processor)


def _ids(model_input: tinker.ModelInput) -> list[int]:
    return [int(token) for token in model_input.to_ints()]


def _oracle_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] | None = None,
    options: KimiK3RenderOptions | None = None,
    image_prompts: list[str] | None = None,
) -> list[int]:
    kwargs = (options or KimiK3RenderOptions()).native_kwargs()
    if image_prompts is not None:
        kwargs["image_prompts"] = image_prompts
    return [
        int(token)
        for token in tokenizer.apply_chat_template(
            copy.deepcopy(messages),
            tools=copy.deepcopy(tools),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
    ]


def _conversation(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    options: KimiK3RenderOptions | None = None,
) -> KimiK3Conversation:
    return KimiK3Conversation(messages, tools=tools, options=options)


def _supervised(
    renderer: KimiK3Renderer,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
) -> tuple[list[int], list[float]]:
    model_input, weights = renderer.build_supervised_example(
        _conversation(messages, tools=tools),
        train_on_what=train_on_what,
    )
    return _ids(model_input), [float(weight) for weight in weights.tolist()]


def _last_action(
    renderer: KimiK3Renderer,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    tokens, weights = _supervised(renderer, messages, tools=tools)
    trained = [index for index, weight in enumerate(weights) if weight == 1.0]
    assert trained
    first, last = trained[0], trained[-1]
    assert all(weight == 0.0 for weight in weights[:first])
    assert all(weight == 1.0 for weight in weights[first : last + 1])
    assert weights[last + 1 :] == [0.0]
    return tokens[:first], tokens[first : last + 1], tokens[last + 1 :]


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([], id="empty"),
        pytest.param(
            [{"role": "assistant", "content": "Already answered."}],
            id="assistant-only",
        ),
        pytest.param(
            [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
                {"role": "system", "content": "Retry with one word."},
            ],
            id="multiple-users-system-retry",
        ),
    ],
)
def test_generation_matches_release_python_oracle(
    tokenizer,
    renderer,
    messages,
) -> None:
    actual = _ids(
        renderer.build_generation_prompt(_conversation(messages), role="assistant")
    )
    assert actual == _oracle_ids(
        tokenizer,
        messages,
        add_generation_prompt=True,
    )


@pytest.mark.parametrize(
    "options",
    [
        pytest.param(KimiK3RenderOptions(), id="thinking-max"),
        pytest.param(
            KimiK3RenderOptions(thinking=True, thinking_effort="low"),
            id="thinking-low",
        ),
        pytest.param(
            KimiK3RenderOptions(thinking=True, thinking_effort="high"),
            id="thinking-high",
        ),
        pytest.param(
            KimiK3RenderOptions(thinking=False, thinking_effort=None),
            id="thinking-disabled",
        ),
        pytest.param(
            KimiK3RenderOptions(tool_choice="required"),
            id="tool-required",
        ),
        pytest.param(
            KimiK3RenderOptions(tool_choice="none"),
            id="tool-none",
        ),
        pytest.param(
            KimiK3RenderOptions(
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "integer"}},
                            "required": ["answer"],
                        },
                    },
                }
            ),
            id="response-schema",
        ),
    ],
)
def test_request_controls_match_release_python_oracle(
    tokenizer,
    renderer,
    options,
) -> None:
    messages = [{"role": "user", "content": "What is 6 * 7?"}]
    actual = _ids(
        renderer.build_generation_prompt(
            _conversation(messages, tools=TOOLS, options=options)
        )
    )
    assert actual == _oracle_ids(
        tokenizer,
        messages,
        tools=TOOLS,
        options=options,
        add_generation_prompt=True,
    )


def test_verl_slime_areal_multiturn_assistant_loss_spans(
    tokenizer,
    renderer,
) -> None:
    tokens, all_assistant_weights = _supervised(
        renderer,
        PARALLEL_TOOL_MESSAGES,
        tools=TOOLS,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    trained_text = tokenizer.decode(
        [
            token
            for token, weight in zip(tokens, all_assistant_weights, strict=True)
            if weight == 1.0
        ]
    )
    for expected in (
        "I need both tools.",
        "Paris is 21°C and 6 * 7 is 42.",
        "42",
    ):
        assert expected in trained_text
    assert "Weather in Paris" not in trained_text
    assert "temperature" not in trained_text

    _, last_weights = _supervised(
        renderer,
        PARALLEL_TOOL_MESSAGES,
        tools=TOOLS,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    assert sum(last_weights) < sum(all_assistant_weights)


def test_sft_production_path_preserves_tools_and_loss_mask(
    tokenizer,
    renderer,
) -> None:
    rendered = render_messages_to_datums(
        PARALLEL_TOOL_MESSAGES,
        renderer=renderer,
        tools=TOOLS,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        include_loss_mask=True,
    )
    assert len(rendered) == 1
    row = rendered[0]
    assert row.token_ids == _oracle_ids(
        tokenizer,
        PARALLEL_TOOL_MESSAGES,
        tools=TOOLS,
        add_generation_prompt=False,
    )
    assert len(row.token_ids) == len(row.token_weights)
    assert row.token_weights[-1] == 0.0

    production_messages = build_tool_prefixed_messages(
        PARALLEL_TOOL_MESSAGES,
        renderer=renderer,
        tools=TOOLS,
    )
    direct_input, direct_weights = renderer.build_supervised_example(
        production_messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    assert row.token_ids == _ids(direct_input)
    assert row.token_weights == direct_weights.tolist()


def test_renderer_prompt_matches_python_template(tokenizer, renderer) -> None:
    assert tokenizer.chat_template is None
    messages = [{"role": "user", "content": "Give one number."}]
    model_input = renderer.build_generation_prompt(messages)
    assert model_input_to_token_ids(model_input) == _oracle_ids(
        tokenizer,
        messages,
        add_generation_prompt=True,
    )
    assert renderer.get_stop_sequences() == [STOP_SEQUENCE]


def test_renderer_resolution_selects_kimi_k3() -> None:
    assert resolve_renderer_name("accounts/example/models/kimi-k3") == "kimi_k3"
    assert resolve_renderer_name("/trusted/local/KIMI_K3_release") == "kimi_k3"
    with pytest.raises(ValueError, match="Could not infer a renderer"):
        resolve_renderer_name("accounts/example/models/kimi-k30")
    with pytest.raises(ValueError, match="Could not infer a renderer"):
        resolve_renderer_name("/trusted/local/kimi_k30_release")


def test_miles_all_pretokenized_message_boundary_cuts(
    tokenizer,
    renderer,
) -> None:
    """Every history cut is stable in both completed and generation form."""
    for cut in range(len(PARALLEL_TOOL_MESSAGES) + 1):
        prefix = PARALLEL_TOOL_MESSAGES[:cut]
        for add_generation_prompt in (False, True):
            trace = renderer.render_trace(
                _conversation(prefix, tools=TOOLS),
                add_generation_prompt=add_generation_prompt,
            )
            assert list(trace.materialized_one_pad_ids) == _oracle_ids(
                tokenizer,
                prefix,
                tools=TOOLS,
                add_generation_prompt=add_generation_prompt,
            ), (cut, add_generation_prompt)
            assert _ids(trace.model_input) == list(trace.expanded_token_ids)


def test_tinker_action_to_next_observation_extension(
    renderer,
) -> None:
    assistant_indices = [
        index
        for index, message in enumerate(PARALLEL_TOOL_MESSAGES)
        if message["role"] == "assistant"
    ]
    for position, assistant_index in enumerate(assistant_indices):
        history = PARALLEL_TOOL_MESSAGES[:assistant_index]
        through_assistant = PARALLEL_TOOL_MESSAGES[: assistant_index + 1]
        observation, action, trailing = _last_action(
            renderer,
            through_assistant,
            tools=TOOLS,
        )
        assert observation == _ids(
            renderer.build_generation_prompt(_conversation(history, tools=TOOLS))
        )
        assert action
        assert len(trailing) == 1

        if position + 1 < len(assistant_indices):
            next_assistant = assistant_indices[position + 1]
            next_observation = _ids(
                renderer.build_generation_prompt(
                    _conversation(
                        PARALLEL_TOOL_MESSAGES[:next_assistant],
                        tools=TOOLS,
                    )
                )
            )
            assert next_observation[: len(observation + action + trailing)] == (
                observation + action + trailing
            )


def test_final_and_streaming_parser_for_every_assistant(
    renderer,
) -> None:
    for index, expected in enumerate(PARALLEL_TOOL_MESSAGES):
        if expected["role"] != "assistant":
            continue
        _, action, _ = _last_action(
            renderer,
            PARALLEL_TOOL_MESSAGES[: index + 1],
            tools=TOOLS,
        )
        parsed, termination = renderer.parse_response(action)
        assert termination.is_clean
        streamed = list(renderer.parse_response_streaming(action))
        assert streamed[-1] == parsed

        visible = "".join(
            str(part.get("text", ""))
            for part in parsed["content"]
            if part.get("type") == "text"
        )
        thinking = "".join(
            str(part.get("thinking", ""))
            for part in parsed["content"]
            if part.get("type") == "thinking"
        )
        assert visible == expected.get("content", "")
        assert thinking == expected.get("reasoning_content", "")
        expected_calls = expected.get("tool_calls") or []
        parsed_calls = parsed.get("tool_calls") or []
        assert [call.id for call in parsed_calls] == [
            f"{call['function']['name']}:{call_index}"
            for call_index, call in enumerate(expected_calls)
        ]
        assert [call.function.name for call in parsed_calls] == [
            call["function"]["name"] for call in expected_calls
        ]
        assert [json.loads(call.function.arguments) for call in parsed_calls] == [
            call["function"]["arguments"] for call in expected_calls
        ]


@pytest.mark.parametrize("seed", [1, 42, 123, 456, 999])
def test_deterministic_random_conversations_match_oracle(
    seed,
    tokenizer,
    renderer,
) -> None:
    rng = random.Random(seed)
    values = ["plain", " leading", "line\nbreak", "你好", "emoji 🚀", "\tindent"]
    messages: list[dict[str, Any]] = [{"role": "system", "content": f"seed={seed}"}]
    for turn in range(rng.randint(1, 5)):
        messages.append(
            {
                "role": "user",
                "content": f"{turn}:{rng.choice(values)}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "reasoning_content": f"reason-{seed}-{turn}",
                "content": rng.choice(values),
            }
        )

    supervised, _ = _supervised(
        renderer,
        messages,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    assert supervised == _oracle_ids(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )
    generation_messages = messages + [
        {"role": "user", "content": f"next-{rng.choice(values)}"}
    ]
    assert _ids(renderer.build_generation_prompt(generation_messages)) == _oracle_ids(
        tokenizer,
        generation_messages,
        add_generation_prompt=True,
    )


def _encoded_image(
    mode: str,
    image_format: str,
) -> bytes:
    image = Image.new(
        mode,
        (35, 49),
        (20, 40, 60, 128) if mode == "RGBA" else (20, 40, 60),
    )
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("payload", "expected_mode"),
    [
        pytest.param(Image.new("RGB", (28, 42), "red"), "RGB", id="pil-rgb"),
        pytest.param(
            Image.new("RGBA", (28, 42), (1, 2, 3, 127)), "RGBA", id="pil-rgba"
        ),
        pytest.param(_encoded_image("RGB", "JPEG"), "RGB", id="jpeg-bytes"),
        pytest.param(
            "data:image/jpeg;base64,"
            + base64.b64encode(_encoded_image("RGB", "JPEG")).decode(),
            "RGB",
            id="jpeg-data-url",
        ),
    ],
)
def test_vision_symbolic_token_in_and_expanded_representations(
    tokenizer,
    vision_renderer,
    payload,
    expected_mode,
) -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": payload},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    before = copy.deepcopy(messages)
    trace = vision_renderer.render_trace(messages)
    assert messages[0]["content"][1] == before[0]["content"][1]
    assert trace.processed_media[0].media.mode == expected_mode
    assert trace.processed_media[0].expected_tokens > 0

    assert list(trace.symbolic_token_ids) == _oracle_ids(
        tokenizer,
        messages,
        add_generation_prompt=True,
    )
    assert list(trace.token_in_ids) == _oracle_ids(
        tokenizer,
        messages,
        add_generation_prompt=True,
        image_prompts=["<|media_pad|>"],
    )
    assert list(trace.materialized_one_pad_ids) == _oracle_ids(
        tokenizer,
        messages,
        add_generation_prompt=True,
        image_prompts=[trace.processed_media[0].image_prompt],
    )
    media_pad_id = vision_renderer.image_placeholder_token_id
    assert trace.materialized_one_pad_ids.count(media_pad_id) == 1
    assert trace.expanded_token_ids.count(media_pad_id) == (
        trace.processed_media[0].expected_tokens
    )
    image_chunks = [
        chunk
        for chunk in trace.model_input.chunks
        if isinstance(chunk, tinker.types.ImageChunk)
    ]
    assert len(image_chunks) == 1
    assert image_chunks[0].expected_tokens == (trace.processed_media[0].expected_tokens)
    assert trace.model_input.length == len(trace.expanded_token_ids)
    token_in_ids, images = build_multimodal_completions_prompt_token_ids(
        messages,
        trace.model_input,
        tokenizer,
        renderer=vision_renderer,
    )
    assert token_in_ids == list(trace.materialized_one_pad_ids)
    assert len(images) == 1


def test_vision_multiple_images_across_turns_with_tools(
    tokenizer,
    vision_renderer,
) -> None:
    first = Image.new("RGB", (28, 28), "blue")
    second = Image.new("RGBA", (42, 28), (1, 2, 3, 100))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": first},
                {"type": "image", "image": first.copy()},
                {"type": "text", "text": "Compare these."},
            ],
        },
        {"role": "assistant", "content": "They match."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "And this one?"},
                {"type": "image", "image": second},
            ],
        },
    ]
    trace = vision_renderer.render_trace(
        _conversation(messages, tools=TOOLS),
    )
    prompts = [item.image_prompt for item in trace.processed_media]
    assert len(prompts) == 3
    assert list(trace.materialized_one_pad_ids) == _oracle_ids(
        tokenizer,
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
        image_prompts=prompts,
    )
    assert (
        sum(
            isinstance(chunk, tinker.types.ImageChunk)
            for chunk in trace.model_input.chunks
        )
        == 3
    )


def test_vision_tool_image_matches_release_oracle_and_is_context_only(
    tokenizer,
    vision_renderer,
) -> None:
    encoded = base64.b64encode(_encoded_image("RGB", "PNG")).decode()
    messages = [
        {"role": "user", "content": "Inspect the screenshot."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _call("weather-1", "get_weather", {"city": "Paris"}),
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "weather-1",
            "content": [
                {"type": "text", "text": "Captured result:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                },
            ],
        },
        {"role": "assistant", "content": "The screenshot is mostly red."},
    ]

    trace = vision_renderer.render_trace(messages)
    prompts = [item.image_prompt for item in trace.processed_media]
    assert len(prompts) == 1
    assert list(trace.materialized_one_pad_ids) == _oracle_ids(
        tokenizer,
        messages,
        add_generation_prompt=True,
        image_prompts=prompts,
    )

    model_input, weights = vision_renderer.build_supervised_example(messages)
    request, processed = vision_renderer._normalized_request(messages)
    segments, labels = vision_renderer._supervised_segments_and_labels(
        request,
        TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    _, token_ids, _, _ = vision_renderer._materialize_segments(
        segments,
        processed,
        labels,
    )
    media_pad_indexes = [
        index
        for index, token in enumerate(token_ids)
        if token == vision_renderer.image_placeholder_token_id
    ]
    assert media_pad_indexes
    assert all(float(weights[index]) == 0.0 for index in media_pad_indexes)
    assert any(float(weight) == 1.0 for weight in weights)
    assert (
        sum(isinstance(chunk, tinker.types.ImageChunk) for chunk in model_input.chunks)
        == 1
    )


def test_vision_reordered_tool_images_stay_with_their_tool_calls(
    tokenizer,
    vision_renderer,
) -> None:
    weather_image = Image.new("RGB", (28, 42), "red")
    math_image = Image.new("RGB", (56, 28), "blue")
    messages = [
        {"role": "user", "content": "Run both tools."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _call("weather-1", "get_weather", {"city": "Paris"}),
                _call("math-1", "calculate", {"expression": "6 * 7"}),
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "math-1",
            "content": [{"type": "image", "image": math_image}],
        },
        {
            "role": "tool",
            "tool_call_id": "weather-1",
            "content": [{"type": "image", "image": weather_image}],
        },
        {"role": "assistant", "content": "Both results are ready."},
    ]

    request, processed = vision_renderer._normalized_request(messages)
    tool_messages = [
        message for message in request.messages if message.get("role") == "tool"
    ]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "weather-1",
        "math-1",
    ]
    assert [(item.media.width, item.media.height) for item in processed] == [
        weather_image.size,
        math_image.size,
    ]

    trace = vision_renderer.render_trace(messages)
    assert list(trace.materialized_one_pad_ids) == _oracle_ids(
        tokenizer,
        messages,
        add_generation_prompt=True,
        image_prompts=[item.image_prompt for item in trace.processed_media],
    )


@pytest.mark.parametrize(
    "messages,error",
    [
        pytest.param(
            [
                {
                    "role": "assistant",
                    "content": [{"type": "image", "image": b"not-used"}],
                }
            ],
            "user or tool messages",
            id="assistant-image",
        ),
        pytest.param(
            [
                {
                    "role": "system",
                    "content": [{"type": "image", "image": b"not-used"}],
                }
            ],
            "user or tool messages",
            id="system-image",
        ),
        pytest.param(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.png"},
                        }
                    ],
                }
            ],
            "remote URLs",
            id="remote-image",
        ),
        pytest.param(
            [
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio": b"bytes"}],
                }
            ],
            "Unsupported",
            id="audio",
        ),
        pytest.param(
            [
                {
                    "role": "user",
                    "content": [{"type": "video", "video": b"bytes"}],
                }
            ],
            "Unsupported",
            id="video",
        ),
    ],
)
def test_vision_rejects_unsupported_media(
    vision_renderer,
    messages,
    error,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        vision_renderer.render_trace(messages)


# ── Suspicious / previously untested edge cases ─────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(None, (True, "max"), id="default-max"),
        pytest.param(True, (True, "max"), id="true-max"),
        pytest.param(False, (False, None), id="false-disabled"),
        pytest.param("none", (False, None), id="none-disabled"),
        pytest.param("low", (True, "low"), id="low"),
        pytest.param("medium", (True, "high"), id="medium-aliases-high"),
        pytest.param("high", (True, "high"), id="high"),
        pytest.param("xhigh", (True, "max"), id="xhigh-aliases-max"),
        pytest.param("max", (True, "max"), id="max"),
        pytest.param({"effort": "low"}, (True, "low"), id="mapping-effort"),
        pytest.param(
            {"reasoning_effort": "xhigh"},
            (True, "max"),
            id="mapping-reasoning_effort",
        ),
    ],
)
def test_normalize_reasoning_aliases(raw, expected) -> None:
    assert _normalize_reasoning(raw) == expected


def test_normalize_reasoning_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported Kimi K3 reasoning effort"):
        _normalize_reasoning("ultra")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(None, None, id="none"),
        pytest.param("auto", None, id="auto"),
        pytest.param("none", "none", id="tool-none"),
        pytest.param("required", "required", id="tool-required"),
        pytest.param(
            {"type": "function", "function": {"name": "get_weather"}},
            "required",
            id="function-mapping",
        ),
    ],
)
def test_normalize_tool_choice(raw, expected) -> None:
    assert _normalize_tool_choice(raw) == expected


def test_normalize_tool_choice_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported Kimi K3 tool_choice"):
        _normalize_tool_choice("sometimes")


def test_render_options_from_api_aliases_and_deep_sort() -> None:
    disabled = KimiK3RenderOptions.from_api(thinking={"type": "disabled"})
    assert disabled == KimiK3RenderOptions(thinking=False, thinking_effort=None)

    required = KimiK3RenderOptions.from_api(
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"b": 1, "a": 2},
            },
        },
    )
    assert required.tool_choice == "required"
    assert required.response_format == {
        "json_schema": {
            "name": "answer",
            "schema": {"a": 2, "b": 1},
        },
        "type": "json_schema",
    }


def test_native_message_preserves_trainable_for_customized_masks() -> None:
    native = _native_message(
        {
            "role": "assistant",
            "content": "hello",
            "trainable": True,
            "reasoning_content": "plan",
        }
    )
    assert native["trainable"] is True
    assert native["reasoning_content"] == "plan"


def test_native_message_rejects_unsupported_role() -> None:
    with pytest.raises(ValueError, match="Unsupported Kimi K3 message role"):
        _native_message({"role": "function", "content": "x"})


def test_jpeg_rgba_is_rejected_before_encoding() -> None:
    image = Image.new("RGBA", (8, 8), (1, 2, 3, 4))
    with pytest.raises(
        ValueError, match="JPEG Kimi K3 inputs must already be RGB or L"
    ):
        _canonical_image_bytes(image, "jpeg")


def test_media_pad_id_is_derived_from_release_tokenizer(tokenizer, renderer) -> None:
    assert renderer.image_placeholder_token_id == int(
        tokenizer.convert_tokens_to_ids("<|media_pad|>")
    )


def test_empty_think_history_observation_matches_generation(
    tokenizer,
    renderer,
) -> None:
    """Release K3 emits an empty think channel in completed history when thinking
    is enabled, so supervised observations stay KV-cache-aligned with generation.
    """
    messages = [
        {"role": "user", "content": "Say hi."},
        {"role": "assistant", "content": "Hi."},
    ]
    observation, action, trailing = _last_action(renderer, messages)
    assert observation == _ids(
        renderer.build_generation_prompt(_conversation(messages[:1]))
    )
    assert action
    assert len(trailing) == 1
    assert "<|open|>think" in tokenizer.decode(observation)
    assert "<|open|>think" in tokenizer.decode(
        _oracle_ids(tokenizer, messages, add_generation_prompt=False)
    )


def test_customized_train_on_what_honors_per_message_masks(
    tokenizer,
    renderer,
) -> None:
    messages = [
        {"role": "user", "content": "First", "trainable": False},
        {
            "role": "assistant",
            "reasoning_content": "skip me",
            "content": "One",
            "trainable": False,
        },
        {"role": "user", "content": "Second", "trainable": False},
        {
            "role": "assistant",
            "reasoning_content": "keep me",
            "content": "Two",
            "trainable": True,
        },
    ]
    tokens, weights = _supervised(
        renderer,
        messages,
        train_on_what=TrainOnWhat.CUSTOMIZED,
    )
    assert tokens == _oracle_ids(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )
    trained = tokenizer.decode(
        [token for token, weight in zip(tokens, weights, strict=True) if weight == 1.0]
    )
    assert "Two" in trained
    assert "keep me" in trained
    assert "One" not in trained
    assert "First" not in trained


def test_customized_requires_trainable_on_every_message(renderer) -> None:
    with pytest.raises(ValueError, match="trainable"):
        renderer.build_supervised_example(
            [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a", "trainable": True},
            ],
            train_on_what=TrainOnWhat.CUSTOMIZED,
        )


def test_last_assistant_turn_masks_only_after_final_user(tokenizer, renderer) -> None:
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "One"},
        {"role": "user", "content": "Second"},
        {
            "role": "assistant",
            "reasoning_content": "final",
            "content": "Two",
        },
    ]
    tokens, weights = _supervised(
        renderer,
        messages,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN,
    )
    trained = tokenizer.decode(
        [token for token, weight in zip(tokens, weights, strict=True) if weight == 1.0]
    )
    assert "Two" in trained
    assert "final" in trained
    assert "One" not in trained


def test_generation_prefill_appends_literal_text(tokenizer, renderer) -> None:
    messages = [{"role": "user", "content": "Continue"}]
    base = _ids(renderer.build_generation_prompt(messages))
    with_prefill = _ids(
        renderer.build_generation_prompt(messages, prefill="partial draft")
    )
    assert with_prefill[: len(base)] == base
    assert tokenizer.decode(with_prefill[len(base) :]) == "partial draft"


def test_disable_thinking_renderer_matches_disabled_options(tokenizer) -> None:
    disabled = get_renderer("kimi_k3_disable_thinking", tokenizer)
    assert isinstance(disabled, KimiK3Renderer)
    messages = [{"role": "user", "content": "Quiet mode"}]
    actual = _ids(disabled.build_generation_prompt(messages))
    options = KimiK3RenderOptions(thinking=False, thinking_effort=None)
    assert actual == _oracle_ids(
        tokenizer,
        messages,
        options=options,
        add_generation_prompt=True,
    )
    assert "thinking-effort" not in tokenizer.decode(actual)


def test_prepare_conversation_binds_live_request_controls(tokenizer) -> None:
    renderer = get_renderer("kimi_k3_disable_thinking", tokenizer)
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {"type": "object"},
        },
    }

    conversation = renderer.prepare_conversation(
        [{"role": "user", "content": "Use the tool."}],
        tools=TOOLS,
        request_kwargs={
            "tool_choice": "required",
            "response_format": response_format,
        },
    )

    assert isinstance(conversation, KimiK3Conversation)
    assert conversation.tools == tuple(TOOLS)
    assert conversation.options == KimiK3RenderOptions(
        thinking=False,
        thinking_effort=None,
        tool_choice="required",
        response_format=response_format,
    )


def test_disable_thinking_ignores_lone_reasoning_effort(tokenizer) -> None:
    renderer = get_renderer("kimi_k3_disable_thinking", tokenizer)
    conversation = renderer.prepare_conversation(
        [{"role": "user", "content": "Stay quiet."}],
        request_kwargs={"reasoning_effort": "high"},
    )

    assert conversation.options.thinking is False
    assert conversation.options.thinking_effort is None


def test_tool_declare_prefix_matches_envelope_tools(tokenizer, renderer) -> None:
    prefix = renderer.create_conversation_prefix_with_tools(
        TOOLS,
        system_prompt="Use tools carefully.",
    )
    messages = prefix + [{"role": "user", "content": "Weather in Paris?"}]
    actual = _ids(renderer.build_generation_prompt(messages))
    oracle_messages = [
        {"role": "system", "content": "Use tools carefully."},
        {"role": "user", "content": "Weather in Paris?"},
    ]
    assert actual == _oracle_ids(
        tokenizer,
        oracle_messages,
        tools=TOOLS,
        add_generation_prompt=True,
    )
    assert "tool-declare" in tokenizer.decode(actual)


def test_tool_declare_and_envelope_tools_conflict(renderer) -> None:
    prefix = renderer.create_conversation_prefix_with_tools(TOOLS)
    conversation = KimiK3Conversation(
        prefix + [{"role": "user", "content": "hi"}],
        tools=TOOLS,
    )
    with pytest.raises(ValueError, match="both the conversation envelope"):
        renderer.build_generation_prompt(conversation)


def test_top_level_and_dynamic_tools_coexist(tokenizer, renderer) -> None:
    messages = [
        {"role": "system", "content": "", "tools": [TOOLS[1]]},
        {"role": "user", "content": "Use whichever tool is needed."},
    ]
    normalized = normalize_messages(messages)
    actual = _ids(
        renderer.build_generation_prompt(_conversation(normalized, tools=[TOOLS[0]]))
    )

    assert actual == _oracle_ids(
        tokenizer,
        messages,
        tools=[TOOLS[0]],
        add_generation_prompt=True,
    )
    decoded = tokenizer.decode(actual)
    assert "# Tools" in decoded
    assert "## New Tools Available" in decoded


def test_dynamic_tools_survive_production_sft_path(tokenizer, renderer) -> None:
    messages = [
        {"role": "user", "content": "Start."},
        {"role": "system", "content": "", "tools": [TOOLS[0]]},
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("weather-1", "get_weather", {"city": "Paris"})],
        },
    ]

    [rendered] = render_messages_to_datums(
        messages,
        renderer=renderer,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    normalized = normalize_messages(messages)
    observation, action, trailing = _last_action(renderer, normalized)

    assert rendered.token_ids == _oracle_ids(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )
    assert rendered.token_ids == observation + action + trailing
    assert rendered.token_weights == (
        [0.0] * len(observation) + [1.0] * len(action) + [0.0] * len(trailing)
    )
    assert observation == _ids(
        renderer.build_generation_prompt(_conversation(normalized[:-1]))
    )


def test_json_string_tool_arguments_match_oracle(tokenizer, renderer) -> None:
    messages = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "weather-1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "Paris"}),
                    },
                }
            ],
        },
    ]
    actual = _ids(
        renderer.build_generation_prompt(_conversation(messages, tools=TOOLS))
    )
    assert actual == _oracle_ids(
        tokenizer,
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
    )


@pytest.mark.parametrize(
    "response",
    [
        pytest.param([11, 22, 33], id="missing-stop"),
        pytest.param(None, id="double-stop"),
    ],
)
def test_parser_marks_malformed_stop_sequences(tokenizer, renderer, response) -> None:
    stop = list(
        tokenizer._encode_text_piece(
            "<|close|>message<|sep|>",
            allow_special_tokens=True,
        )
    )
    tokens = stop + stop if response is None else response
    parsed, termination = renderer.parse_response(tokens)
    assert not termination.is_clean
    assert parsed["role"] == "assistant"


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            '<|open|>call index="1"<|sep|><|close|>call<|sep|>',
            id="missing-tool-name",
        ),
        pytest.param(
            '<|open|>call tool="calculate" index="1"<|sep|>'
            '<|open|>json type="object"<|sep|>{broken'
            "<|close|>json<|sep|><|close|>call<|sep|>",
            id="invalid-json-block",
        ),
    ],
)
def test_parser_marks_malformed_tool_calls(tokenizer, renderer, call) -> None:
    response = (
        "<|close|>think<|sep|><|open|>response<|sep|>"
        "<|close|>response<|sep|><|open|>tools<|sep|>"
        f"{call}"
        "<|close|>tools<|sep|><|close|>message<|sep|>"
    )
    tokens = list(tokenizer._encode_text_piece(response, allow_special_tokens=True))

    parsed, termination = renderer.parse_response(tokens)

    assert not termination.is_clean
    assert parsed["role"] == "assistant"
    assert not parsed.get("tool_calls")


def test_parser_treats_literal_control_text_as_content(tokenizer, renderer) -> None:
    """Control-looking text must not become structure unless tokenized specially."""
    messages = [
        {"role": "user", "content": "Echo markers"},
        {
            "role": "assistant",
            "reasoning_content": "stay structured",
            "content": "literal <|close|>message<|sep|> text",
        },
    ]
    _, action, _ = _last_action(renderer, messages)
    parsed, termination = renderer.parse_response(action)
    assert termination.is_clean
    visible = "".join(
        str(part.get("text", ""))
        for part in parsed["content"]
        if part.get("type") == "text"
    )
    assert visible == "literal <|close|>message<|sep|> text"


def test_vision_supervised_matches_oracle_and_zeros_image_pads(
    tokenizer,
    vision_renderer,
) -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (28, 42), "red")},
                {"type": "text", "text": "Describe this image."},
            ],
        },
        {
            "role": "assistant",
            "reasoning_content": "Color check.",
            "content": "A red rectangle.",
        },
    ]
    model_input, weights = vision_renderer.build_supervised_example(
        messages,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    request, processed = vision_renderer._normalized_request(messages)
    segments, labels = vision_renderer._supervised_segments_and_labels(
        request,
        TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    _, expanded, one_pad, material_weights = vision_renderer._materialize_segments(
        segments,
        processed,
        labels,
    )
    materialized_ids = vision_renderer._encode_segments(
        vision_renderer._replace_image_segments(segments, processed)
    )
    oracle = _oracle_ids(
        tokenizer,
        messages,
        add_generation_prompt=False,
        image_prompts=[item.image_prompt for item in processed],
    )
    assert one_pad == materialized_ids == oracle
    assert [float(weight) for weight in weights.tolist()] == material_weights
    assert model_input.length == len(expanded) == len(material_weights)
    assert material_weights[-1] == 0.0
    assert sum(
        isinstance(chunk, tinker.types.ImageChunk) for chunk in model_input.chunks
    ) == len(processed)
    media_pad_id = vision_renderer.image_placeholder_token_id
    pad_indexes = [
        index for index, token in enumerate(expanded) if token == media_pad_id
    ]
    assert pad_indexes
    assert all(material_weights[index] == 0.0 for index in pad_indexes)
    assert expanded.count(media_pad_id) == sum(
        item.expected_tokens for item in processed
    )
    assert one_pad.count(media_pad_id) == len(processed)


def test_vision_image_url_data_url_matches_image_type(
    tokenizer,
    vision_renderer,
) -> None:
    encoded = base64.b64encode(_encoded_image("RGB", "JPEG")).decode()
    data_url = f"data:image/jpeg;base64,{encoded}"
    as_image = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": data_url},
                {"type": "text", "text": "Caption?"},
            ],
        }
    ]
    as_image_url = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "Caption?"},
            ],
        }
    ]
    left = vision_renderer.render_trace(as_image)
    right = vision_renderer.render_trace(as_image_url)
    assert list(left.materialized_one_pad_ids) == list(right.materialized_one_pad_ids)
    assert list(left.expanded_token_ids) == list(right.expanded_token_ids)
    assert list(left.materialized_one_pad_ids) == _oracle_ids(
        tokenizer,
        as_image,
        add_generation_prompt=True,
        image_prompts=[left.processed_media[0].image_prompt],
    )


def test_multipart_thinking_content_matches_reasoning_content_field(
    tokenizer,
    renderer,
) -> None:
    via_field = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "reasoning_content": "plan",
            "content": "hello",
        },
    ]
    via_parts = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "plan"},
                {"type": "text", "text": "hello"},
            ],
        },
    ]
    assert _ids(renderer.build_generation_prompt(_conversation(via_field[:1]))) == _ids(
        renderer.build_generation_prompt(_conversation(via_parts[:1]))
    )
    left, _ = _supervised(renderer, via_field)
    right, _ = _supervised(renderer, via_parts)
    assert (
        left
        == right
        == _oracle_ids(
            tokenizer,
            via_field,
            add_generation_prompt=False,
        )
    )
