"""Release-oracle tests for the Kimi K3.1 cookbook renderer.

As for Kimi K3, the checkpoint tokenizer's Python ``apply_chat_template`` is the
canonical oracle (it delegates to the K3.1 ``encoding_k3.py``); no template is
synthesized here. The renderer is a thin adaptation of ``kimi_k3``, so this
suite concentrates on what differs: thinking tiers and the no-effort default,
literal tool-argument bytes, serving-shaped tools, and the token-level parser
(partial stops, request thinking mode, malformed turns). Image handling is the
inherited K3 path and is exercised once end to end.

The oracle loads from the public ``moonshotai/Kimi-K3.1-Pro`` repository at an
immutable reviewed revision. ``KIMI_K31_MODEL_PATH`` is an optional local or
offline override.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import os
import random
from typing import Any

import pytest
from fireworks.training import sdk
from fireworks.training.sdk.sampling import SampledCompletion
from PIL import Image
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from training.renderer import ParseTermination, TrainOnWhat, get_registered_renderer_names, get_renderer
from training.renderer.kimi_k3 import KimiK3Renderer
from training.renderer.kimi_k31 import (
    STOP_SEQUENCE,
    THINKING_EFFORTS,
    KimiK31Conversation,
    KimiK31Renderer,
    KimiK31RenderOptions,
    _normalize_reasoning,
    _normalize_tools,
)
from training.utils.rl.rollout.renderer import build_multimodal_completions_prompt_token_ids, single_turn_renderer_rollout
from training.utils.supervised import (
    build_renderer,
    build_renderer_from_resolved_name,
    build_tool_prefixed_messages,
    normalize_messages,
    render_messages_to_datums,
    renderer_supports_images,
    renderer_supports_tool_images,
    resolve_renderer_name,
)
from training.utils.tokenizers import load_tokenizer

_KIMI_K31_HF_MODEL = "moonshotai/Kimi-K3.1-Pro"
_KIMI_K31_HF_REVISION = "d9e99b0792a5e93b78ea9bea62a68167e36ddd9e"
_LOCAL_RELEASE_PATH = os.environ.get("KIMI_K31_MODEL_PATH")
_RELEASE_SOURCE = _LOCAL_RELEASE_PATH or _KIMI_K31_HF_MODEL
_RELEASE_REVISION = None if _LOCAL_RELEASE_PATH else _KIMI_K31_HF_REVISION

_OPEN, _CLOSE, _SEP, _EOM = "<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>"

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
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
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
                "properties": {"expression": {"type": "string"}, "precision": {"type": "number"}},
                "required": ["expression"],
            },
        },
    },
]


def _call(call_id: str, name: str, arguments: Any) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


# Tool results arrive out of call order, one call carries a JSON-string argument
# payload with non-canonical literals, and user text contains literal markers.
PARALLEL_TOOL_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "Use tools and answer briefly."},
    {"role": "user", "content": "Weather in Paris and 6 * 7? Ignore this <|open|>message<|sep|> text."},
    {
        "role": "assistant",
        "reasoning_content": "I need both tools.",
        "content": "",
        "tool_calls": [
            _call("weather-1", "get_weather", {"city": "Paris"}),
            _call(
                "math-1",
                "calculate",
                '{"expression": "6 * 7", "precision": 1e2, "opts": {"b": 1, "a": [1, 2]}, "flag": true, "none": null}',
            ),
        ],
    },
    {"role": "tool", "tool_call_id": "math-1", "content": json.dumps({"result": 42})},
    {"role": "tool", "tool_call_id": "weather-1", "content": json.dumps({"temperature": 21, "unit": "celsius"})},
    {"role": "assistant", "reasoning_content": "Both results are available.", "content": "Paris is 21°C and 6 * 7 is 42."},
    {"role": "user", "content": "Repeat only the number. 🔢"},
    {"role": "assistant", "content": "42"},
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    """Test-only oracle: the pinned checkpoint tokenizer's own Python.

    Loading checkpoint remote code is deliberate here and scoped to tests --
    it certifies that the reviewed code-free path renders identical tokens.
    Production loading (``native_tokenizer``) never imports checkpoint Python.
    """
    load_kwargs: dict[str, Any] = {"local_files_only": True} if _LOCAL_RELEASE_PATH else {"revision": _KIMI_K31_HF_REVISION}
    try:
        loaded = AutoTokenizer.from_pretrained(_RELEASE_SOURCE, trust_remote_code=True, **load_kwargs)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        pytest.skip(f"Kimi K3.1 HF oracle tokenizer unavailable: {exc}")
    assert loaded.chat_template is None
    assert type(loaded).apply_chat_template.__module__.startswith("transformers_modules.")
    return loaded


@pytest.fixture(scope="module")
def native_tokenizer():
    """The production loading path under test: the code-free native bundle.

    Even with an explicit ``trust_remote_code=True`` request, recognized Kimi
    K3.1 data resolves to the reviewed fast tokenizer; ``auto_map`` and
    checkpoint Python stay inert.
    """
    try:
        loaded = load_tokenizer(_RELEASE_SOURCE, _RELEASE_REVISION, True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        pytest.skip(f"Kimi K3.1 native tokenizer unavailable: {exc}")
    assert loaded.chat_template is None
    assert not type(loaded).__module__.startswith("transformers_modules.")
    return loaded


@pytest.fixture(scope="module")
def renderer(native_tokenizer) -> KimiK31Renderer:
    loaded = get_renderer("kimi_k31", native_tokenizer)
    assert isinstance(loaded, KimiK31Renderer)
    assert isinstance(loaded, KimiK3Renderer)
    return loaded


@pytest.fixture(scope="module")
def image_processor():
    load_kwargs: dict[str, Any] = {"local_files_only": True} if _LOCAL_RELEASE_PATH else {"revision": _KIMI_K31_HF_REVISION}
    try:
        processor_cls = get_class_from_dynamic_module(
            "kimi_k3_vision_processing.KimiK3VisionProcessor", _RELEASE_SOURCE, **load_kwargs
        )
        processor = processor_cls.from_pretrained(_RELEASE_SOURCE, **load_kwargs)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        pytest.skip(f"Kimi K3.1 HF image processor unavailable: {exc}")
    processor.preserve_image_mode = True
    return processor


@pytest.fixture(scope="module")
def vision_renderer(native_tokenizer, image_processor) -> KimiK31Renderer:
    return KimiK31Renderer(native_tokenizer, image_processor=image_processor)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ids(model_input: sdk.ModelInput) -> list[int]:
    return [int(token) for token in model_input.to_ints()]


def _expanded_ids(model_input: sdk.ModelInput, pad_id: int) -> list[int]:
    tokens: list[int] = []
    for chunk in model_input.chunks:
        if isinstance(chunk, sdk.types.ImageChunk):
            tokens.extend([pad_id] * chunk.expected_tokens)
        else:
            tokens.extend(int(token) for token in chunk.tokens)
    return tokens


def _controls(tokenizer, text: str) -> list[int]:
    return [int(token) for token in tokenizer._encode_text_piece(text, allow_special_tokens=True)]


def _oracle_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape messages the way serving hands them to the template.

    Cookbook ``thinking`` parts become the template's top-level
    ``reasoning_content``, and message-level tools are re-serialized like request
    tools (serving's ``format_tools`` path).
    """
    shaped: list[dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        if copied.get("role") == "system" and copied.get("tools"):
            copied["tools"] = list(_normalize_tools(copied["tools"]))
        content = copied.get("content")
        if copied.get("role") == "assistant" and isinstance(content, list):
            thinking = "".join(str(p.get("thinking", "")) for p in content if p.get("type") == "thinking")
            if thinking and not copied.get("reasoning_content"):
                copied["reasoning_content"] = thinking
            copied["content"] = [p for p in content if p.get("type") != "thinking"]
        shaped.append(copied)
    return shaped


def _oracle_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] | None = None,
    options: KimiK31RenderOptions | None = None,
    image_prompts: list[str] | None = None,
) -> list[int]:
    options = options or KimiK31RenderOptions()
    kwargs = options.native_kwargs()
    # The checkpoint template defaults thinking_effort to "max" when the key is
    # absent; the renderer (like serving) renders no effort message unless asked.
    kwargs.setdefault("thinking_effort", None)
    if image_prompts is not None:
        kwargs["image_prompts"] = image_prompts
    return [
        int(token)
        for token in tokenizer.apply_chat_template(
            copy.deepcopy(_oracle_messages(messages)),
            tools=list(_normalize_tools(tools)) if tools else None,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
    ]


def _conversation(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    options: KimiK31RenderOptions | None = None,
) -> KimiK31Conversation:
    return KimiK31Conversation(messages, tools=tools, options=options)


def _supervised(
    renderer: KimiK31Renderer,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
) -> tuple[list[int], list[float]]:
    model_input, weights = renderer.build_supervised_example(_conversation(messages, tools=tools), train_on_what=train_on_what)
    return _ids(model_input), [float(weight) for weight in weights.tolist()]


def _last_action(
    renderer: KimiK31Renderer,
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


def _parsed_to_message(parsed: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(str(part.get("text", "")) for part in parsed["content"] if part.get("type") == "text"),
    }
    thinking = "".join(str(part.get("thinking", "")) for part in parsed["content"] if part.get("type") == "thinking")
    if thinking:
        message["reasoning_content"] = thinking
    if parsed.get("tool_calls"):
        message["tool_calls"] = [
            {"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}}
            for call in parsed["tool_calls"]
        ]
    return message


def _encoded_image(mode: str, image_format: str) -> bytes:
    image = Image.new(mode, (35, 49), (20, 40, 60, 128) if mode == "RGBA" else (20, 40, 60))
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Registration, resolution, tokenizer contract
# ---------------------------------------------------------------------------


def test_registered_names() -> None:
    names = set(get_registered_renderer_names())
    assert {"kimi_k31", "kimi_k31_disable_thinking", *(f"kimi_k31_{effort}" for effort in THINKING_EFFORTS)} <= names


def test_renderer_resolution_selects_kimi_k31() -> None:
    for model in (
        "moonshotai/Kimi-K3.1-Pro",
        "https://huggingface.co/moonshotai/Kimi-K3.1-Pro",
        "accounts/fireworks/models/kimi-k3p1-pro",
        "accounts/example/models/kimi-k3p1-finetune",
        "/trusted/local/kimi_k31_release",
        "/trusted/local/Kimi-K3.1",
    ):
        assert resolve_renderer_name(model) == "kimi_k31", model
    assert resolve_renderer_name("moonshotai/Kimi-K3") == "kimi_k3"
    assert resolve_renderer_name("accounts/fireworks/models/kimi-k3") == "kimi_k3"
    with pytest.raises(ValueError, match="Could not infer a renderer"):
        resolve_renderer_name("accounts/example/models/kimi-k30")
    with pytest.raises(ValueError, match="Could not infer a renderer"):
        resolve_renderer_name("accounts/example/models/kimi-k3p10")


@pytest.mark.parametrize("name", ["kimi_k31", "kimi_k31_disable_thinking", "kimi_k31_high"])
def test_renderer_image_capabilities_for_managed_validation(name: str) -> None:
    assert renderer_supports_images(name) is True
    assert renderer_supports_tool_images(name) is True


def test_rejects_tokenizers_without_the_encoding_contract() -> None:
    class _FastLike:
        backend_tokenizer = object()

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [ord(c) for c in text]

    # Neither a serializable fast backend (the native bundle) nor the legacy
    # checkpoint TikTokenTokenizer: rendering is impossible, so reject.
    with pytest.raises(TypeError, match="TikTokenTokenizer"):
        KimiK31Renderer(_FastLike())


# ---------------------------------------------------------------------------
# Generation parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([], id="empty"),
        pytest.param([{"role": "assistant", "content": "Already answered."}], id="assistant-only"),
        pytest.param(
            [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
                {"role": "system", "content": "Retry with one word."},
            ],
            id="multiple-users-system-retry",
        ),
        pytest.param(
            [
                {"role": "user", "name": "alice", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": " two"}]},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "plan"}, {"type": "text", "text": "done"}]},
            ],
            id="multipart-and-names",
        ),
        pytest.param(PARALLEL_TOOL_MESSAGES, id="parallel-tools"),
    ],
)
def test_generation_matches_release_python_oracle(tokenizer, renderer, messages) -> None:
    actual = _ids(renderer.build_generation_prompt(_conversation(messages, tools=TOOLS)))
    assert actual == _oracle_ids(tokenizer, messages, tools=TOOLS, add_generation_prompt=True)


def test_default_prompt_has_no_effort_message(tokenizer, renderer) -> None:
    messages = [{"role": "user", "content": "Give one number."}]
    actual = _ids(renderer.build_generation_prompt(messages))
    assert actual == _oracle_ids(tokenizer, messages, add_generation_prompt=True)
    decoded = tokenizer.decode(actual)
    assert decoded.endswith(f'{_OPEN}message role="assistant"{_SEP}{_OPEN}think{_SEP}')
    assert "thinking-effort" not in decoded
    # The checkpoint template's own default is "max"; the renderer deliberately differs.
    assert "thinking_effort=max" in tokenizer.decode(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    assert renderer.get_stop_sequences() == [STOP_SEQUENCE]


@pytest.mark.parametrize(
    "options",
    [
        pytest.param(KimiK31RenderOptions(), id="thinking-no-effort"),
        *[pytest.param(KimiK31RenderOptions(thinking=True, thinking_effort=e), id=f"thinking-{e}") for e in THINKING_EFFORTS],
        pytest.param(KimiK31RenderOptions(thinking=False, thinking_effort=None), id="thinking-disabled"),
        pytest.param(KimiK31RenderOptions(tool_choice="required"), id="tool-required"),
        pytest.param(KimiK31RenderOptions(tool_choice="none"), id="tool-none"),
        pytest.param(KimiK31RenderOptions(response_format={"type": "json_object"}), id="json-object"),
        pytest.param(
            KimiK31RenderOptions(
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "schema": {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]},
                    },
                }
            ),
            id="response-schema",
        ),
    ],
)
def test_request_controls_match_release_python_oracle(tokenizer, renderer, options) -> None:
    messages = [{"role": "user", "content": "What is 6 * 7?"}]
    actual = _ids(renderer.build_generation_prompt(_conversation(messages, tools=TOOLS, options=options)))
    assert actual == _oracle_ids(tokenizer, messages, tools=TOOLS, options=options, add_generation_prompt=True)


@pytest.mark.parametrize("effort", THINKING_EFFORTS)
def test_effort_pinned_renderers_match_options(tokenizer, native_tokenizer, effort) -> None:
    pinned = get_renderer(f"kimi_k31_{effort}", native_tokenizer)
    assert isinstance(pinned, KimiK31Renderer)
    messages = [{"role": "user", "content": "Think it over."}]
    options = KimiK31RenderOptions(thinking=True, thinking_effort=effort)
    actual = _ids(pinned.build_generation_prompt(messages))
    assert actual == _oracle_ids(tokenizer, messages, options=options, add_generation_prompt=True)
    assert f"thinking_effort={effort}" in tokenizer.decode(actual)
    # A bare envelope keeps the renderer's pinned effort.
    assert _ids(pinned.build_generation_prompt(_conversation(messages))) == actual


def test_disable_thinking_renderer_matches_disabled_options(tokenizer, native_tokenizer) -> None:
    disabled = get_renderer("kimi_k31_disable_thinking", native_tokenizer)
    messages = [{"role": "user", "content": "Quiet mode"}, {"role": "assistant", "content": "ok"}]
    options = KimiK31RenderOptions(thinking=False, thinking_effort=None)
    actual = _ids(disabled.build_generation_prompt(messages))
    assert actual == _oracle_ids(tokenizer, messages, options=options, add_generation_prompt=True)
    assert "think" not in tokenizer.decode(actual)
    supervised, _ = _supervised(disabled, messages)
    assert supervised == _oracle_ids(tokenizer, messages, options=options, add_generation_prompt=False)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, (True, None)),
        (True, (True, None)),
        ("low", (True, "low")),
        ("medium", (True, "high")),
        ("high", (True, "high")),
        ("max", (True, "max")),
        ("xhigh", (True, "max")),
        ({"effort": "medium"}, (True, "high")),
        ("none", (False, None)),
        (False, (False, None)),
    ],
)
def test_normalize_reasoning(raw, expected) -> None:
    assert _normalize_reasoning(raw) == expected


def test_normalize_reasoning_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="reasoning effort"):
        _normalize_reasoning("ultra")
    with pytest.raises(ValueError, match="reasoning effort"):
        _normalize_reasoning({"effort": "ultra"})
    with pytest.raises(ValueError, match="thinking_effort"):
        KimiK31RenderOptions(thinking_effort="ultra")
    # Aliases are normalized at the API boundary, not native encoder options.
    with pytest.raises(ValueError, match="thinking_effort"):
        KimiK31RenderOptions(thinking_effort="medium")


def test_prepare_conversation_binds_live_request_controls(tokenizer, native_tokenizer, renderer) -> None:
    response_format = {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}}
    conversation = renderer.prepare_conversation(
        [{"role": "user", "content": "Use the tool."}],
        tools=TOOLS,
        request_kwargs={"reasoning_effort": "xhigh", "tool_choice": {"type": "function"}, "response_format": response_format},
    )
    assert isinstance(conversation, KimiK31Conversation)
    assert conversation.tools == tuple(TOOLS)
    assert conversation.options == KimiK31RenderOptions(
        thinking=True, thinking_effort="max", tool_choice="required", response_format=response_format
    )
    medium = renderer.prepare_conversation([{"role": "user", "content": "Think a bit."}], request_kwargs={"reasoning_effort": "medium"})
    # The pinned encoder has no medium tier; it folds into high as in kimi_k3.
    assert medium.options.thinking_effort == "high"
    assert "thinking_effort=high" in tokenizer.decode(_ids(renderer.build_generation_prompt(medium)))

    disabled = get_renderer("kimi_k31_disable_thinking", native_tokenizer)
    quiet = disabled.prepare_conversation([{"role": "user", "content": "Stay quiet."}], request_kwargs={"reasoning_effort": "high"})
    assert quiet.options.thinking is False and quiet.options.thinking_effort is None
    reopened = disabled.prepare_conversation(
        [{"role": "user", "content": "Think."}], request_kwargs={"thinking": {"type": "enabled"}, "reasoning_effort": "low"}
    )
    assert reopened.options.thinking is True and reopened.options.thinking_effort == "low"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_tool_declare_prefix_matches_envelope_tools(tokenizer, renderer) -> None:
    prefix = renderer.create_conversation_prefix_with_tools([tool["function"] for tool in TOOLS], system_prompt="Use tools carefully.")
    messages = prefix + [{"role": "user", "content": "Weather in Paris?"}]
    actual = _ids(renderer.build_generation_prompt(messages))
    oracle_messages = [{"role": "system", "content": "Use tools carefully."}, {"role": "user", "content": "Weather in Paris?"}]
    assert actual == _oracle_ids(tokenizer, oracle_messages, tools=TOOLS, add_generation_prompt=True)
    assert actual == _ids(renderer.build_generation_prompt(_conversation(oracle_messages, tools=TOOLS)))
    assert "tool-declare" in tokenizer.decode(actual)


def test_tool_declare_and_envelope_tools_conflict(renderer) -> None:
    prefix = renderer.create_conversation_prefix_with_tools([tool["function"] for tool in TOOLS])
    with pytest.raises(ValueError, match="both the conversation envelope"):
        renderer.build_generation_prompt(KimiK31Conversation(prefix + [{"role": "user", "content": "hi"}], tools=TOOLS))


def test_tools_are_shaped_like_serving(tokenizer, renderer) -> None:
    """``strict``/``container``/``None`` never reach the prompt, as in Fireworks serving."""
    noisy = [
        {"type": "function", "container": None, "function": {**TOOLS[0]["function"], "strict": True, "description": None}},
        TOOLS[1]["function"],  # bare function spec
    ]
    messages = [{"role": "user", "content": "hi"}]
    actual = _ids(renderer.build_generation_prompt(_conversation(messages, tools=noisy)))
    assert actual == _oracle_ids(tokenizer, messages, tools=noisy, add_generation_prompt=True)
    # Shaping drops exactly the keys serving drops: the noisy list renders like
    # the same tools written cleanly (the first without its None description).
    clean = [
        {"type": "function", "function": {k: v for k, v in TOOLS[0]["function"].items() if k != "description"}},
        TOOLS[1],
    ]
    assert actual == _ids(renderer.build_generation_prompt(_conversation(messages, tools=clean)))
    decoded = tokenizer.decode(actual)
    assert "strict" not in decoded and "container" not in decoded and "null" not in decoded
    assert "calculate" in decoded and "get_weather" in decoded
    assert list(_normalize_tools(TOOLS)) == TOOLS


def test_dynamic_tools_are_shaped_like_serving(tokenizer, renderer) -> None:
    """Message-level tool declarations get the same serving shaping as request tools."""
    noisy_dynamic = {"type": "function", "container": None, "function": {**TOOLS[1]["function"], "strict": True, "description": None}}
    messages = [
        {"role": "user", "content": "Start."},
        {"role": "system", "content": "", "tools": [noisy_dynamic]},
        {"role": "user", "content": "6 * 7?"},
    ]
    actual = _ids(renderer.build_generation_prompt(_conversation(normalize_messages(messages), tools=[TOOLS[0]])))
    assert actual == _oracle_ids(tokenizer, messages, tools=[TOOLS[0]], add_generation_prompt=True)
    decoded = tokenizer.decode(actual)
    assert "## New Tools Available" in decoded and "calculate" in decoded
    assert "strict" not in decoded and "container" not in decoded and "null" not in decoded
    [rendered] = render_messages_to_datums(
        messages + [{"role": "assistant", "content": "42"}], renderer=renderer, tools=[TOOLS[0]], train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
    )
    assert "strict" not in tokenizer.decode(rendered.token_ids)


def test_top_level_and_dynamic_tools_coexist(tokenizer, renderer) -> None:
    messages = [{"role": "system", "content": "", "tools": [TOOLS[1]]}, {"role": "user", "content": "Use whichever tool is needed."}]
    normalized = normalize_messages(messages)
    actual = _ids(renderer.build_generation_prompt(_conversation(normalized, tools=[TOOLS[0]])))
    assert actual == _oracle_ids(tokenizer, messages, tools=[TOOLS[0]], add_generation_prompt=True)
    decoded = tokenizer.decode(actual)
    assert "# Tools" in decoded and "## New Tools Available" in decoded


def test_json_string_tool_arguments_keep_literal_bytes(tokenizer, renderer) -> None:
    messages = PARALLEL_TOOL_MESSAGES[:3]
    actual = _ids(renderer.build_generation_prompt(_conversation(messages, tools=TOOLS)))
    assert actual == _oracle_ids(tokenizer, messages, tools=TOOLS, add_generation_prompt=True)
    decoded = tokenizer.decode(actual)
    assert 'key="precision" type="number"<|sep|>1e2<|close|>' in decoded
    assert 'key="opts" type="object"<|sep|>{"b": 1, "a": [1, 2]}<|close|>' in decoded


# ---------------------------------------------------------------------------
# Supervised examples (inherited K3 masks on the K3.1 encoder)
# ---------------------------------------------------------------------------


def test_supervised_matches_oracle_and_assistant_masks(tokenizer, renderer) -> None:
    tokens, all_assistant_weights = _supervised(renderer, PARALLEL_TOOL_MESSAGES, tools=TOOLS, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
    assert tokens == _oracle_ids(tokenizer, PARALLEL_TOOL_MESSAGES, tools=TOOLS, add_generation_prompt=False)
    trained_text = tokenizer.decode([token for token, weight in zip(tokens, all_assistant_weights, strict=True) if weight == 1.0])
    for expected in ("I need both tools.", "Paris is 21°C and 6 * 7 is 42.", "42", 'key="precision" type="number"<|sep|>1e2'):
        assert expected in trained_text
    for excluded in ("Weather in Paris", "temperature", "tool-declare", 'role="user"', _EOM):
        assert excluded not in trained_text
    assert trained_text.count(STOP_SEQUENCE) == 3
    assert all_assistant_weights[-1] == 0.0
    _, last_weights = _supervised(renderer, PARALLEL_TOOL_MESSAGES, tools=TOOLS, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
    assert 0 < sum(last_weights) < sum(all_assistant_weights)


def test_customized_train_on_what_honors_per_message_masks(tokenizer, renderer) -> None:
    messages = [
        {"role": "user", "content": "First", "trainable": False},
        {"role": "assistant", "reasoning_content": "skip me", "content": "One", "trainable": False},
        {"role": "user", "content": "Second", "trainable": False},
        {"role": "assistant", "reasoning_content": "keep me", "content": "Two", "trainable": True},
    ]
    tokens, weights = _supervised(renderer, messages, train_on_what=TrainOnWhat.CUSTOMIZED)
    assert tokens == _oracle_ids(tokenizer, messages, add_generation_prompt=False)
    trained = tokenizer.decode([token for token, weight in zip(tokens, weights, strict=True) if weight == 1.0])
    assert "Two" in trained and "keep me" in trained and "One" not in trained and "First" not in trained


def test_message_boundary_cuts_match_oracle(tokenizer, renderer) -> None:
    for cut in range(len(PARALLEL_TOOL_MESSAGES) + 1):
        prefix = PARALLEL_TOOL_MESSAGES[:cut]
        for add_generation_prompt in (False, True):
            trace = renderer.render_trace(_conversation(prefix, tools=TOOLS), add_generation_prompt=add_generation_prompt)
            assert list(trace.materialized_one_pad_ids) == _oracle_ids(
                tokenizer, prefix, tools=TOOLS, add_generation_prompt=add_generation_prompt
            ), (cut, add_generation_prompt)


def test_action_to_next_observation_extension(renderer) -> None:
    assistant_indices = [i for i, message in enumerate(PARALLEL_TOOL_MESSAGES) if message["role"] == "assistant"]
    for position, assistant_index in enumerate(assistant_indices):
        history = PARALLEL_TOOL_MESSAGES[:assistant_index]
        observation, action, trailing = _last_action(renderer, PARALLEL_TOOL_MESSAGES[: assistant_index + 1], tools=TOOLS)
        assert observation == _ids(renderer.build_generation_prompt(_conversation(history, tools=TOOLS)))
        assert action
        assert trailing == [renderer._parser.end_of_msg_id]
        if position + 1 < len(assistant_indices):
            next_observation = _ids(
                renderer.build_generation_prompt(_conversation(PARALLEL_TOOL_MESSAGES[: assistant_indices[position + 1]], tools=TOOLS))
            )
            assert next_observation[: len(observation + action + trailing)] == observation + action + trailing


def test_generation_prefill_appends_literal_text(tokenizer, renderer) -> None:
    messages = [{"role": "user", "content": "Continue"}]
    base = _ids(renderer.build_generation_prompt(messages))
    with_prefill = _ids(renderer.build_generation_prompt(messages, prefill="partial <|close|> draft"))
    assert with_prefill[: len(base)] == base
    assert tokenizer.decode(with_prefill[len(base) :]) == "partial <|close|> draft"
    assert renderer._parser.close_id not in with_prefill[len(base) :]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parser_round_trips_every_assistant(renderer) -> None:
    for index, expected in enumerate(PARALLEL_TOOL_MESSAGES):
        if expected["role"] != "assistant":
            continue
        history = PARALLEL_TOOL_MESSAGES[:index]
        _, action, _ = _last_action(renderer, PARALLEL_TOOL_MESSAGES[: index + 1], tools=TOOLS)
        parsed, termination = renderer.parse_response(action)
        assert termination == ParseTermination.STOP_SEQUENCE
        assert list(renderer.parse_response_streaming(action))[-1] == parsed

        visible = "".join(str(part.get("text", "")) for part in parsed["content"] if part.get("type") == "text")
        thinking = "".join(str(part.get("thinking", "")) for part in parsed["content"] if part.get("type") == "thinking")
        assert visible == expected.get("content", "")
        assert thinking == expected.get("reasoning_content", "")
        expected_calls = expected.get("tool_calls") or []
        parsed_calls = parsed.get("tool_calls") or []
        assert [call.id for call in parsed_calls] == [f"{call['function']['name']}:{i}" for i, call in enumerate(expected_calls)]
        assert [call.function.name for call in parsed_calls] == [call["function"]["name"] for call in expected_calls]
        for parsed_call, expected_call in zip(parsed_calls, expected_calls, strict=True):
            raw = expected_call["function"]["arguments"]
            assert json.loads(parsed_call.function.arguments) == (json.loads(raw) if isinstance(raw, str) else raw)
        # The parsed message re-renders to the sampled tokens.
        _, rebuilt_action, _ = _last_action(renderer, history + [_parsed_to_message(parsed)], tools=TOOLS)
        assert rebuilt_action == action


def test_parser_preserves_argument_literals(renderer) -> None:
    _, action, _ = _last_action(renderer, PARALLEL_TOOL_MESSAGES[:3], tools=TOOLS)
    parsed, _ = renderer.parse_response(action)
    assert parsed["tool_calls"][1].function.arguments == (
        '{"expression":"6 * 7","precision":1e2,"opts":{"b": 1, "a": [1, 2]},"flag":true,"none":null}'
    )


def test_parser_accepts_a_partial_trailing_stop(renderer) -> None:
    """Completions keep the stop's leading ``<|close|>`` after the stop string is stripped."""
    stop = renderer._parser.stop_ids
    for index in (2, 5, 7):
        _, action, _ = _last_action(renderer, PARALLEL_TOOL_MESSAGES[: index + 1], tools=TOOLS)
        full_parsed, full_termination = renderer.parse_response(action)
        assert full_termination == ParseTermination.STOP_SEQUENCE
        partial = action[: len(action) - len(stop)] + [renderer._parser.close_id]
        parsed, termination = renderer.parse_response(partial)
        assert termination == ParseTermination.STOP_SEQUENCE
        assert parsed == full_parsed
    _, action, _ = _last_action(renderer, PARALLEL_TOOL_MESSAGES[:3], tools=TOOLS)
    unfinished, termination = renderer.parse_response(action[: len(action) - len(stop)])
    assert termination == ParseTermination.MALFORMED
    assert unfinished.get("tool_calls")


def test_parser_follows_the_request_thinking_mode(tokenizer, native_tokenizer, renderer) -> None:
    """A complete turn is recognized from its structure whatever the renderer default."""
    conversation = renderer.prepare_conversation([{"role": "user", "content": "Say hello."}], request_kwargs={"reasoning_effort": "none"})
    assert tokenizer.decode(_ids(renderer.build_generation_prompt(conversation))).endswith(f"{_OPEN}response{_SEP}")
    parsed, termination = renderer.parse_response(_controls(tokenizer, f"Hello{_CLOSE}response{_SEP}{STOP_SEQUENCE}"))
    assert termination == ParseTermination.STOP_SEQUENCE
    assert parsed["content"] == [{"type": "text", "text": "Hello"}]
    disabled = get_renderer("kimi_k31_disable_thinking", native_tokenizer)
    parsed, termination = disabled.parse_response(
        _controls(tokenizer, f"plan{_CLOSE}think{_SEP}{_OPEN}response{_SEP}Hi{_CLOSE}response{_SEP}{STOP_SEQUENCE}")
    )
    assert termination == ParseTermination.STOP_SEQUENCE
    assert parsed["content"] == [{"type": "thinking", "thinking": "plan"}, {"type": "text", "text": "Hi"}]


def test_truncated_turn_follows_the_request_mode_when_given(tokenizer, native_tokenizer, renderer) -> None:
    """A turn cut before any channel boundary is content only if the request disabled thinking."""
    conversation = renderer.prepare_conversation([{"role": "user", "content": "6 * 7?"}], request_kwargs={"reasoning_effort": "none"})
    assert renderer.thinking_mode_of(conversation) is False
    assert renderer.thinking_mode_of([{"role": "user", "content": "plain list"}]) is None
    truncated = [int(t) for t in tokenizer._encode_text_piece("42", allow_special_tokens=False)]

    parsed, termination = renderer.parse_response(truncated, thinking=renderer.thinking_mode_of(conversation))
    assert termination == ParseTermination.MALFORMED
    assert parsed["content"] == [{"type": "text", "text": "42"}]
    assert list(renderer.parse_response_streaming(truncated, thinking=False))[-1] == parsed
    assert renderer.parse_response(truncated)[0]["content"] == [{"type": "thinking", "thinking": "42"}]
    disabled = get_renderer("kimi_k31_disable_thinking", native_tokenizer)
    assert disabled.parse_response(truncated)[0]["content"] == [{"type": "text", "text": "42"}]
    assert disabled.parse_response(truncated, thinking=True)[0]["content"] == [{"type": "thinking", "thinking": "42"}]


def test_rollout_helper_passes_the_request_thinking_mode(tokenizer, renderer) -> None:
    conversation = renderer.prepare_conversation([{"role": "user", "content": "6 * 7?"}], request_kwargs={"reasoning_effort": "none"})
    truncated = [int(t) for t in tokenizer._encode_text_piece("42", allow_special_tokens=False)]
    seen: dict[str, Any] = {}

    async def sample_with_prompt_tokens(prompt_token_ids: list[int], **kwargs: Any) -> list[SampledCompletion]:
        seen["prompt"] = list(prompt_token_ids)
        return [
            SampledCompletion(
                text="42",
                full_tokens=list(prompt_token_ids) + truncated,
                prompt_len=len(prompt_token_ids),
                finish_reason="length",
                inference_logprobs=[-0.1] * len(truncated),
                sampling_logprobs=[-0.1] * len(truncated),
            )
        ]

    def reward_fn(row: Any, parsed: Any, parse_ok: bool) -> float:
        seen["parsed"] = parsed
        return 1.0

    run = asyncio.run(
        single_turn_renderer_rollout(
            {"id": 1},
            renderer=renderer,
            sample_with_prompt_tokens=sample_with_prompt_tokens,
            message_builder=lambda row: conversation,
            reward_fn=reward_fn,
        )
    )
    assert run is not None
    assert seen["prompt"] == _ids(renderer.build_generation_prompt(conversation))
    assert seen["parsed"]["content"] == [{"type": "text", "text": "42"}]


@pytest.mark.parametrize(
    "response",
    [
        pytest.param([11, 22, 33], id="missing-stop"),
        pytest.param(f"{STOP_SEQUENCE}{STOP_SEQUENCE}", id="double-stop"),
        pytest.param(f"still thinking{STOP_SEQUENCE}", id="unclosed-think"),
        pytest.param(f"{_CLOSE}think{_SEP}{_OPEN}response{_SEP}no close{STOP_SEQUENCE}", id="unclosed-response"),
        pytest.param(
            f"{_CLOSE}think{_SEP}{_OPEN}response{_SEP}{_CLOSE}response{_SEP}{_OPEN}tools{_SEP}"
            f'{_OPEN}call index="1"{_SEP}{_CLOSE}call{_SEP}{_CLOSE}tools{_SEP}{STOP_SEQUENCE}',
            id="missing-tool-name",
        ),
        pytest.param(
            f"{_CLOSE}think{_SEP}{_OPEN}response{_SEP}{_CLOSE}response{_SEP}{_OPEN}tools{_SEP}"
            f'{_OPEN}call tool="calculate" index="1"{_SEP}{_OPEN}json type="object"{_SEP}{{broken'
            f"{_CLOSE}json{_SEP}{_CLOSE}call{_SEP}{_CLOSE}tools{_SEP}{STOP_SEQUENCE}",
            id="invalid-json-block",
        ),
        pytest.param(
            f"{_CLOSE}think{_SEP}{_OPEN}response{_SEP}{_CLOSE}response{_SEP}{_OPEN}tools{_SEP}"
            f'{_OPEN}call tool="calculate" index="1"{_SEP}{_OPEN}argument key="precision" type="number"{_SEP}abc'
            f"{_CLOSE}argument{_SEP}{_CLOSE}call{_SEP}{_CLOSE}tools{_SEP}{STOP_SEQUENCE}",
            id="non-json-number-argument",
        ),
        pytest.param(
            f"{_CLOSE}think{_SEP}{_OPEN}response{_SEP}{_CLOSE}response{_SEP}{_OPEN}tools{_SEP}"
            f'{_OPEN}call tool="calculate" index="1"{_SEP}{_CLOSE}call{_SEP}{_CLOSE}tools{_SEP}junk{STOP_SEQUENCE}',
            id="content-after-tools",
        ),
        pytest.param(f"{_CLOSE}think{_SEP}{_OPEN}response{_SEP}{_CLOSE}response{_SEP}stray{STOP_SEQUENCE}", id="content-after-response"),
    ],
)
def test_parser_marks_malformed_responses(tokenizer, renderer, response) -> None:
    tokens = response if isinstance(response, list) else _controls(tokenizer, response)
    parsed, termination = renderer.parse_response(tokens)
    assert termination == ParseTermination.MALFORMED
    assert parsed["role"] == "assistant"
    assert not parsed.get("tool_calls")


def test_parser_reads_raw_json_block_and_empty_call(tokenizer, renderer) -> None:
    response = (
        f"{_CLOSE}think{_SEP}{_OPEN}response{_SEP}{_CLOSE}response{_SEP}{_OPEN}tools{_SEP}"
        f'{_OPEN}call tool="calculate" index="1"{_SEP}{_OPEN}json type="object"{_SEP}{{"expression": "1+1"}}'
        f"{_CLOSE}json{_SEP}{_CLOSE}call{_SEP}"
        f'{_OPEN}call tool="get_weather" index="2"{_SEP}{_CLOSE}call{_SEP}{_CLOSE}tools{_SEP}{STOP_SEQUENCE}'
    )
    parsed, termination = renderer.parse_response(_controls(tokenizer, response))
    assert termination == ParseTermination.STOP_SEQUENCE
    calls = parsed["tool_calls"]
    assert [call.function.name for call in calls] == ["calculate", "get_weather"]
    assert calls[0].function.arguments == '{"expression": "1+1"}'
    assert calls[1].function.arguments == "{}"


def test_parser_treats_literal_control_text_as_content(renderer) -> None:
    messages = [
        {"role": "user", "content": "Echo markers"},
        {"role": "assistant", "reasoning_content": "stay <|open|> structured", "content": "literal <|close|>message<|sep|> text"},
    ]
    _, action, _ = _last_action(renderer, messages)
    parsed, termination = renderer.parse_response(action)
    assert termination == ParseTermination.STOP_SEQUENCE
    assert parsed["content"] == [
        {"type": "thinking", "thinking": "stay <|open|> structured"},
        {"type": "text", "text": "literal <|close|>message<|sep|> text"},
    ]


def test_parser_flags_control_tokens_inside_text(tokenizer, renderer) -> None:
    parser = renderer._parser
    text = lambda s: [int(t) for t in tokenizer._encode_text_piece(s, allow_special_tokens=False)]  # noqa: E731
    response = _controls(tokenizer, f"{_CLOSE}think{_SEP}{_OPEN}response{_SEP}") + text("a") + [parser.open_id] + text("b")
    response += _controls(tokenizer, f"{_CLOSE}response{_SEP}{STOP_SEQUENCE}")
    parsed, termination = renderer.parse_response(response)
    assert termination == ParseTermination.MALFORMED
    assert parsed["content"] == [{"type": "text", "text": f"a{_OPEN}b"}]


def test_length_truncated_reasoning_is_not_content(tokenizer, renderer) -> None:
    tokens = [int(t) for t in tokenizer._encode_text_piece("half a thought", allow_special_tokens=False)]
    parsed, termination = renderer.parse_response(tokens)
    assert termination == ParseTermination.MALFORMED
    assert parsed["content"] == [{"type": "thinking", "thinking": "half a thought"}]


# ---------------------------------------------------------------------------
# Production SFT path
# ---------------------------------------------------------------------------


def test_sft_production_path_preserves_tools_literals_and_loss_mask(tokenizer, renderer) -> None:
    rendered = render_messages_to_datums(
        PARALLEL_TOOL_MESSAGES, renderer=renderer, tools=TOOLS, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES, include_loss_mask=True
    )
    assert len(rendered) == 1
    row = rendered[0]
    production_messages = build_tool_prefixed_messages(PARALLEL_TOOL_MESSAGES, renderer=renderer, tools=TOOLS)
    direct_input, direct_weights = renderer.build_supervised_example(production_messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
    assert row.token_ids == _ids(direct_input)
    assert row.token_weights == direct_weights.tolist()
    assert row.token_weights[-1] == 0.0
    # Normalization keeps the original argument bytes, so the production row renders
    # like the raw messages and like the checkpoint encoding of those raw messages.
    assert row.token_ids == _oracle_ids(tokenizer, PARALLEL_TOOL_MESSAGES, tools=TOOLS, add_generation_prompt=False)


def test_normalization_keeps_argument_literals_only_for_literal_renderers(renderer) -> None:
    raw = PARALLEL_TOOL_MESSAGES[2]["tool_calls"][1]["function"]["arguments"]
    preserved = normalize_messages(PARALLEL_TOOL_MESSAGES, preserve_tool_argument_literals=True)
    assert preserved[2]["tool_calls"][1].function.arguments == raw
    unicode_row = [{"role": "assistant", "content": "", "tool_calls": [_call("c", "get_weather", {"city": "北京"})]}]
    assert normalize_messages(unicode_row, preserve_tool_argument_literals=True)[0]["tool_calls"][0].function.arguments == '{"city": "北京"}'
    canonical = normalize_messages(PARALLEL_TOOL_MESSAGES)[2]["tool_calls"][1].function.arguments
    assert canonical != raw and json.loads(canonical) == json.loads(raw)
    assert renderer.preserves_tool_argument_literals is True
    assert getattr(KimiK3Renderer, "preserves_tool_argument_literals", False) is False


def test_dynamic_tools_survive_production_sft_path(tokenizer, renderer) -> None:
    messages = [
        {"role": "user", "content": "Start."},
        {"role": "system", "content": "", "tools": [TOOLS[0]]},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [_call("weather-1", "get_weather", {"city": "Paris"})]},
    ]
    [rendered] = render_messages_to_datums(messages, renderer=renderer, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
    normalized = normalize_messages(messages, preserve_tool_argument_literals=True)
    observation, action, trailing = _last_action(renderer, normalized)
    assert rendered.token_ids == _oracle_ids(tokenizer, messages, add_generation_prompt=False)
    assert rendered.token_ids == observation + action + trailing
    assert rendered.token_weights == [0.0] * len(observation) + [1.0] * len(action) + [0.0] * len(trailing)


_WEIGHTED_ROWS = {
    "single_turn_masked_draft": [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "DRAFT", "weight": 0},
        {"role": "assistant", "content": "ANSWER", "weight": 1},
    ],
    "multi_turn_masked_middle": [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1", "weight": 1},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "CALL", "weight": 1},
        {"role": "assistant", "content": "RETRY", "weight": 0},
        {"role": "assistant", "content": "ANSWER", "weight": 1},
    ],
}


@pytest.mark.parametrize("row_id", sorted(_WEIGHTED_ROWS))
def test_weighted_rows_only_withhold_loss(tokenizer, renderer, row_id) -> None:
    row = _WEIGHTED_ROWS[row_id]
    weighted = render_messages_to_datums(row, renderer=renderer, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
    reference = render_messages_to_datums(
        [{k: v for k, v in m.items() if k != "weight"} for m in row], renderer=renderer, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    assert len(weighted) == len(reference) == 1
    assert weighted[0].token_ids == reference[0].token_ids
    trained = tokenizer.decode([t for t, w in zip(weighted[0].token_ids, weighted[0].token_weights) if w > 0])
    assert "ANSWER" in trained and "DRAFT" not in trained and "RETRY" not in trained
    assert not [
        i for i, (w, r) in enumerate(zip(weighted[0].token_weights, reference[0].token_weights)) if w > 0 and r == 0
    ]


# ---------------------------------------------------------------------------
# Vision (inherited K3 path on the K3.1 checkpoint processor)
# ---------------------------------------------------------------------------


def test_vision_matches_oracle_and_zeros_image_pads(tokenizer, vision_renderer) -> None:
    encoded = base64.b64encode(_encoded_image("RGB", "PNG")).decode()
    messages = [
        {"role": "user", "content": [{"type": "image", "image": Image.new("RGB", (28, 42), "red")}, {"type": "text", "text": "Describe."}]},
        {"role": "assistant", "content": "", "tool_calls": [_call("weather-1", "get_weather", {"city": "Paris"})]},
        {
            "role": "tool",
            "tool_call_id": "weather-1",
            "content": [{"type": "text", "text": "Captured:"}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}],
        },
        {"role": "assistant", "reasoning_content": "Color check.", "content": "A red rectangle."},
    ]
    trace = vision_renderer.render_trace(_conversation(messages, tools=TOOLS))
    prompts = [item.image_prompt for item in trace.processed_media]
    assert len(prompts) == 2
    assert list(trace.materialized_one_pad_ids) == _oracle_ids(
        tokenizer, messages, tools=TOOLS, add_generation_prompt=True, image_prompts=prompts
    )
    assert list(trace.token_in_ids) == _oracle_ids(
        tokenizer, messages, tools=TOOLS, add_generation_prompt=True, image_prompts=["<|media_pad|>"] * 2
    )
    token_in_ids, images = build_multimodal_completions_prompt_token_ids(
        _conversation(messages, tools=TOOLS), trace.model_input, tokenizer, renderer=vision_renderer
    )
    assert token_in_ids == list(trace.token_in_ids)
    assert len(images) == 2

    model_input, weights = vision_renderer.build_supervised_example(_conversation(messages, tools=TOOLS))
    pad_id = vision_renderer.image_placeholder_token_id
    expanded = _expanded_ids(model_input, pad_id)
    assert len(expanded) == len(weights) == model_input.length
    pad_indexes = [i for i, token in enumerate(expanded) if token == pad_id]
    assert len(pad_indexes) == sum(item.expected_tokens for item in trace.processed_media)
    assert all(float(weights[i]) == 0.0 for i in pad_indexes)
    assert any(float(w) == 1.0 for w in weights)
    assert sum(isinstance(chunk, sdk.types.ImageChunk) for chunk in model_input.chunks) == 2
    observation = _expanded_ids(vision_renderer.build_generation_prompt(_conversation(messages[:-1], tools=TOOLS)), pad_id)
    assert expanded[: len(observation)] == observation


def test_developer_role_is_rejected_by_the_k31_encoder(tokenizer, renderer) -> None:
    """Pins the matrix's ``unsupported_scenarios``: K3's encoder accepts ``developer``, K3.1's does not."""
    messages = [{"role": "developer", "content": "Be terse."}, {"role": "user", "content": "hi"}]
    with pytest.raises(ValueError, match="developer"):
        renderer.build_generation_prompt(messages)
    with pytest.raises(ValueError, match="developer"):
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


def _staged_subset(tmp_path, names: tuple[str, ...]):
    """A directory holding exactly ``names`` from the pinned release, like a trainer stage dir."""
    import shutil

    staged = tmp_path / "tokenizer-opaque"
    staged.mkdir()
    if _LOCAL_RELEASE_PATH:
        for name in names:
            shutil.copy(os.path.join(_LOCAL_RELEASE_PATH, name), staged / name)
    else:
        from huggingface_hub import hf_hub_download

        for name in names:
            shutil.copy(hf_hub_download(_KIMI_K31_HF_MODEL, name, revision=_KIMI_K31_HF_REVISION), staged / name)
    return staged


# The ``kimi-k3p1-pro`` upload: tokenizer, encoder and processor config, but not
# the processor module ``preprocessor_config.json`` maps to.
_PRO_ARTIFACT_FILES = (
    "config.json",
    "encoding_k3.py",
    "generation_config.json",
    "preprocessor_config.json",
    "tiktoken.model",
    "tokenization_kimi.py",
    "tokenizer_config.json",
)


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (1, 1),
        (28, 28),
        (35, 49),
        (640, 480),
        (1023, 1),
        (1, 1023),
        (1920, 1080),
        (4096, 4096),
        (9000, 100),
        (100, 9000),
        (7168, 7168),
        (12000, 9000),
        (333, 777),
    ],
)
def test_image_token_counter_matches_release_processor(image_processor, width, height) -> None:
    """The in-repo counter reproduces the checkpoint processor's count, prompt and grid."""
    from training.renderer.kimi_k31 import KimiK31ImageTokenCounter

    counter = KimiK31ImageTokenCounter(image_processor.media_proc_cfg)
    image = Image.new("RGBA" if (width + height) % 2 else "RGB", (width, height), (10, 20, 30, 128) if (width + height) % 2 else (10, 20, 30))
    item = {"type": "image", "image": image}
    expected = int(image_processor.media_tokens_calculator({"type": "image", "image": image.copy()}))
    assert counter.media_tokens_calculator(item) == expected
    assert counter.make_image_prompt(width, height) == image_processor.make_image_prompt(width, height)
    grid = tuple(int(v) for v in image_processor.preprocess([{"type": "image", "image": image.copy()}], return_tensors="pt")["grid_thws"][0].tolist())
    assert counter.grid_thw(width, height) == grid


def test_counter_backed_renderer_matches_release_processor_renderer(native_tokenizer, vision_renderer, image_processor) -> None:
    from training.renderer.kimi_k31 import KimiK31ImageTokenCounter

    counted = KimiK31Renderer(native_tokenizer, image_processor=KimiK31ImageTokenCounter(image_processor.media_proc_cfg))
    encoded = base64.b64encode(_encoded_image("RGB", "PNG")).decode()
    messages = [
        {"role": "user", "content": [{"type": "image", "image": Image.new("RGBA", (300, 200), (1, 2, 3, 100))}, {"type": "text", "text": "Describe."}]},
        {"role": "assistant", "content": "", "tool_calls": [_call("weather-1", "get_weather", {"city": "Paris"})]},
        {"role": "tool", "tool_call_id": "weather-1", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}]},
        {"role": "assistant", "reasoning_content": "Color check.", "content": "A red rectangle."},
    ]
    conversation = _conversation(messages, tools=TOOLS)
    left = counted.render_trace(conversation)
    right = vision_renderer.render_trace(conversation)
    assert left.materialized_one_pad_ids == right.materialized_one_pad_ids
    assert left.expanded_token_ids == right.expanded_token_ids
    assert [m.expected_tokens for m in left.processed_media] == [m.expected_tokens for m in right.processed_media]
    assert [c.model_dump() for c in left.model_input.chunks] == [c.model_dump() for c in right.model_input.chunks]
    left_input, left_weights = counted.build_supervised_example(conversation)
    right_input, right_weights = vision_renderer.build_supervised_example(conversation)
    assert left_weights.tolist() == right_weights.tolist()
    assert [c.model_dump() for c in left_input.chunks] == [c.model_dump() for c in right_input.chunks]


def test_staged_pro_artifact_renders_images_without_processor_module(tokenizer, image_processor, tmp_path) -> None:
    """Managed workers build the counter from preprocessor_config.json; the missing
    ``kimi_k3_vision_processing.py`` no longer matters."""
    from training.renderer.kimi_k31 import KimiK31ImageTokenCounter

    staged = _staged_subset(tmp_path, _PRO_ARTIFACT_FILES)
    staged_tokenizer = load_tokenizer(str(staged), None, None)
    # The staged pinned data resolves to the code-free native bundle; the
    # checkpoint's remote-code TikTokenTokenizer is never imported.
    assert type(staged_tokenizer).__name__ == "TokenizersBackend"
    assert not type(staged_tokenizer).__module__.startswith("transformers_modules.")
    for built in (
        build_renderer(staged_tokenizer, str(staged), "kimi_k31"),
        build_renderer_from_resolved_name(staged_tokenizer, str(staged), "kimi_k31"),
    ):
        assert isinstance(built.image_processor, KimiK31ImageTokenCounter)
        assert built.image_processor.media_proc_cfg == image_processor.media_proc_cfg
    built = build_renderer(staged_tokenizer, str(staged), "kimi_k31")
    assert isinstance(built, KimiK31Renderer)
    messages = [{"role": "user", "content": [{"type": "image", "image": Image.new("RGB", (640, 480), "red")}, {"type": "text", "text": "hello"}]}]
    trace = built.render_trace(messages)
    assert trace.processed_media[0].expected_tokens == int(image_processor.media_tokens_calculator({"type": "image", "image": Image.new("RGB", (640, 480), "red")}))
    assert list(trace.materialized_one_pad_ids) == _oracle_ids(
        tokenizer, messages, add_generation_prompt=True, image_prompts=[trace.processed_media[0].image_prompt]
    )
    assert "HF_TRUST_REMOTE_CODE" not in os.environ


def test_pickle_round_trip_keeps_the_counter_on_the_pro_artifact(tokenizer, image_processor, tmp_path) -> None:
    """Pickle restore goes through the same processor route as construction."""
    import pickle

    from training.renderer.kimi_k31 import KimiK31ImageTokenCounter

    staged = _staged_subset(tmp_path, _PRO_ARTIFACT_FILES)
    staged_tokenizer = load_tokenizer(str(staged), None, None)
    built = build_renderer(staged_tokenizer, str(staged), "kimi_k31")
    assert isinstance(built.image_processor, KimiK31ImageTokenCounter)
    restored = pickle.loads(pickle.dumps(built))
    assert isinstance(restored, KimiK31Renderer)
    assert isinstance(restored.image_processor, KimiK31ImageTokenCounter)
    assert restored.image_processor.media_proc_cfg == built.image_processor.media_proc_cfg
    messages = [{"role": "user", "content": [{"type": "image", "image": Image.new("RGB", (640, 480), "red")}, {"type": "text", "text": "hello"}]}]
    assert restored.render_trace(messages).expanded_token_ids == built.render_trace(messages).expanded_token_ids
    assert "HF_TRUST_REMOTE_CODE" not in os.environ


def test_staged_artifact_without_processor_config_starts_text_only(tokenizer, tmp_path) -> None:
    staged = _staged_subset(tmp_path, tuple(n for n in _PRO_ARTIFACT_FILES if n != "preprocessor_config.json"))
    staged_tokenizer = load_tokenizer(str(staged), None, None)
    built = build_renderer(staged_tokenizer, str(staged), "kimi_k31")
    assert built.image_processor is None
    messages = [{"role": "user", "content": "hello"}]
    assert _ids(built.build_generation_prompt(messages)) == _oracle_ids(tokenizer, messages, add_generation_prompt=True)
    with pytest.raises(ValueError, match="image processor"):
        built.build_generation_prompt([{"role": "user", "content": [{"type": "image", "image": Image.new("RGB", (28, 28), "red")}]}])


def test_rejects_images_without_processor(renderer) -> None:
    with pytest.raises(ValueError, match="image processor"):
        renderer.build_generation_prompt([{"role": "user", "content": [{"type": "image", "image": Image.new("RGB", (28, 28), "red")}]}])


# ---------------------------------------------------------------------------
# Fuzz
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 42, 123, 456, 999])
def test_deterministic_random_conversations_match_oracle(seed, tokenizer, renderer) -> None:
    rng = random.Random(seed)
    values = ["plain", " leading", "line\nbreak", "你好", "emoji 🚀", "\tindent", "<|sep|> literal", "&quot;&amp;", ""]
    messages: list[dict[str, Any]] = [{"role": "system", "content": f"seed={seed}"}]
    for turn in range(rng.randint(1, 5)):
        messages.append({"role": "user", "content": f"{turn}:{rng.choice(values)}"})
        assistant: dict[str, Any] = {"role": "assistant", "content": rng.choice(values)}
        if rng.random() < 0.7:
            assistant["reasoning_content"] = f"reason-{seed}-{turn}{rng.choice(values)}"
        if rng.random() < 0.5:
            calls = [
                _call(
                    f"call-{turn}-{k}",
                    rng.choice(["get_weather", "calculate"]),
                    {"city": rng.choice(values), "precision": rng.choice([1, 2.5, "x", None, True, [1], {"q": 1}])},
                )
                for k in range(rng.randint(1, 3))
            ]
            assistant["tool_calls"] = calls
            messages.append(assistant)
            order = list(range(len(calls)))
            rng.shuffle(order)
            for k in order:
                messages.append({"role": "tool", "tool_call_id": calls[k]["id"], "content": rng.choice(values)})
            messages.append({"role": "assistant", "content": rng.choice(values)})
        else:
            messages.append(assistant)

    supervised, weights = _supervised(renderer, messages, tools=TOOLS, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
    assert supervised == _oracle_ids(tokenizer, messages, tools=TOOLS, add_generation_prompt=False)
    assert len(weights) == len(supervised)
    generation_messages = messages + [{"role": "user", "content": f"next-{rng.choice(values)}"}]
    assert _ids(renderer.build_generation_prompt(_conversation(generation_messages, tools=TOOLS))) == _oracle_ids(
        tokenizer, generation_messages, tools=TOOLS, add_generation_prompt=True
    )
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        _, action, _ = _last_action(renderer, messages[: index + 1], tools=TOOLS)
        parsed, termination = renderer.parse_response(action)
        assert termination == ParseTermination.STOP_SEQUENCE, (seed, index)
        _, rebuilt, _ = _last_action(renderer, messages[:index] + [_parsed_to_message(parsed)], tools=TOOLS)
        assert rebuilt == action, (seed, index)
