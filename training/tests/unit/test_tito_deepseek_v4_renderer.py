from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fireworks.training.sdk import TITOChatRequest, TITOError, TITOIncrementalRenderer
from training.examples.harbor_rl.tito.sidecar import build_sidecar_bundle
from training.renderer.tito import build_sidecar_tito_renderer, load_sidecar_tokenizer
from training.renderer.tito.deepseek_v4 import (
    DEEPSEEK_V4_RENDERER_NAME,
    DEEPSEEK_V4_REPOSITORY,
    DEEPSEEK_V4_REVISION,
    DEEPSEEK_V4_PRO_RENDERER_NAME,
    DEEPSEEK_V4_PRO_REPOSITORY,
    DEEPSEEK_V4_PRO_REVISION,
    DeepSeekV4TITORenderer,
)
from training.renderer.tito.deepseek_v41 import (
    DEEPSEEK_V41_RENDERER_NAME,
    DEEPSEEK_V41_REPOSITORY,
    DEEPSEEK_V41_REVISION,
)

TOOL = {
    "type": "function",
    "function": {
        "name": "run",
        "description": "Run a command",
        "parameters": {"type": "object", "properties": {"z": {"type": "string"}, "a": {"type": "integer"}}},
    },
}


def _tool_history(arguments: Any, content: str | None = None, reasoning: str = "reason") -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "Run"},
        {
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "run", "arguments": arguments}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]


@pytest.mark.parametrize("tools", [None, []])
def test_null_and_empty_tools_match_omitted_tools(native: tuple[Any, DeepSeekV4TITORenderer], tools: Any) -> None:
    _, renderer = native
    payload = {"messages": [{"role": "user", "content": "Hello"}]}
    expected = renderer.render_conversation_tokens(TITOChatRequest.from_openai(payload))
    payload["tools"] = tools
    request = TITOChatRequest.from_openai(payload)
    assert renderer.render_conversation_tokens(request) == expected
    assert request.wire_value() == payload


@pytest.mark.parametrize("part", [{"type": "text"}, {"type": "text", "text": None}, {"type": "text", "text": 42}])
def test_rejects_non_string_text_blocks(native: tuple[Any, DeepSeekV4TITORenderer], part: dict[str, Any]) -> None:
    _, renderer = native
    request = TITOChatRequest.from_openai({"messages": [{"role": "user", "content": [part]}]})
    with pytest.raises(TITOError, match="supports text only") as error:
        renderer.render_conversation_tokens(request)
    assert error.value.status == 400


def test_text_blocks_match_joined_native_text(native: tuple[Any, DeepSeekV4TITORenderer]) -> None:
    _, renderer = native
    expected = renderer.render_conversation_tokens(
        TITOChatRequest.from_openai({"messages": [{"role": "user", "content": "Hello world"}]})
    )
    parts = [{"type": "text", "text": text} for text in ("Hello ", "", "world")]
    payload = {"messages": [{"role": "user", "content": parts}]}
    request = TITOChatRequest.from_openai(payload)
    assert renderer.render_conversation_tokens(request) == expected
    assert request.wire_value() == payload


@pytest.fixture(
    scope="module",
    params=[
        (DEEPSEEK_V4_REPOSITORY, DEEPSEEK_V4_REVISION, DEEPSEEK_V4_RENDERER_NAME),
        (DEEPSEEK_V4_PRO_REPOSITORY, DEEPSEEK_V4_PRO_REVISION, DEEPSEEK_V4_PRO_RENDERER_NAME),
        (DEEPSEEK_V41_REPOSITORY, DEEPSEEK_V41_REVISION, DEEPSEEK_V41_RENDERER_NAME),
    ],
)
def native(request: pytest.FixtureRequest) -> tuple[Any, DeepSeekV4TITORenderer]:
    transformers = pytest.importorskip("transformers")
    repository, revision, name = request.param
    if name == DEEPSEEK_V41_RENDERER_NAME:
        # lazy: tokenizer loading requires the optional transformers dependency.
        from training.utils.tokenizers import load_tokenizer

        tokenizer = load_tokenizer(repository, tokenizer_revision=revision, trust_remote_code=False)
    else:
        tokenizer = transformers.TokenizersBackend.from_pretrained(
            repository, revision=revision, trust_remote_code=False
        )
    renderer = build_sidecar_tito_renderer(tokenizer, name)
    assert renderer.renderer_id == name
    assert renderer.certification_id.startswith(repository.rsplit("/", 1)[-1].lower())
    return tokenizer, renderer


@pytest.mark.parametrize("system", [False, True])
@pytest.mark.parametrize("tools", [False, True])
@pytest.mark.parametrize("content", [None, "", "None"])
def test_full_history_matches_official_encoder(
    native: tuple[Any, DeepSeekV4TITORenderer], system: bool, tools: bool, content: str | None
) -> None:
    tokenizer, renderer = native
    messages = (
        ([{"role": "system", "content": "System"}] if system else [])
        + _tool_history('{"z":"hello","a":2}', content, "ordered reasoning")
        + [{"role": "user", "content": "Continue"}]
    )
    payload = {"messages": messages, "tools": [TOOL] if tools else []}
    expected = copy.deepcopy(messages)
    expected[2 if system else 1]["content"] = content or ""
    if tools:
        if not system:
            expected.insert(0, {"role": "system", "content": ""})
        expected[0]["tools"] = [TOOL]
    text = renderer._encoder.encode_messages(
        expected,
        thinking_mode="thinking",
        drop_thinking=False,
        reasoning_effort=renderer.reasoning_effort,
        add_default_bos_token=True,
    )
    request = TITOChatRequest.from_openai(payload)
    assert tuple(renderer.render_conversation_tokens(request)) == tuple(
        tokenizer.encode(text, add_special_tokens=False)
    )
    assert request.wire_value() == payload
    assert "ordered reasoning" in text
    assert not isinstance(renderer, TITOIncrementalRenderer)


@pytest.mark.parametrize("with_eos", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_parse_parallel_tools_preserves_argument_order_and_ids(
    native: tuple[Any, DeepSeekV4TITORenderer], with_eos: bool, parallel: bool, finish_reason: str
) -> None:
    tokenizer, renderer = native
    assistant = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "reason",
        "tool_calls": [{"type": "function", "function": {"name": "run", "arguments": '{"z":"hello","a":2}'}}],
    }
    if parallel:
        assistant["tool_calls"] *= 2
    # Render a correctly formatted sampled turn with the independent official
    # encoder, removing only the generation-prefix-owned <think>.
    text = renderer._encoder.render_message(
        1, [{"role": "user", "content": "run"}, assistant], "thinking", drop_thinking=False
    )
    text = text.removeprefix("<think>")
    if not with_eos:
        text = text.removesuffix(renderer._encoder.eos_token)
    request = TITOChatRequest.from_openai({"messages": [{"role": "user", "content": "run"}], "tools": [TOOL]})
    ids = tokenizer.encode(text, add_special_tokens=False)
    parsed = renderer.parse_assistant(request, ids, "ignored", finish_reason)
    calls = parsed.response_message["tool_calls"]
    assert len(calls) == (2 if parallel else 1)
    assert len({call["id"] for call in calls}) == len(calls)
    assert calls == renderer.parse_assistant(request, ids, "", "stop").response_message["tool_calls"]
    assert all(list(json.loads(call["function"]["arguments"])) == ["z", "a"] for call in calls)
    assert parsed.message["reasoning_content"] == "reason"


@pytest.mark.parametrize(
    "text,finish",
    [
        ("missing close", "stop"),
        ("reason</think>\n\n<\uff5cDSML\uff5ctool_calls>", "stop"),
        ("reason</think>\n\n<\uff5cDSML\uff5ctool_calls>", "length"),
        ("reason</think>Explain the \uff5cDSML\uff5c marker", "stop"),
        ("reason</think>Explain the \uff5cDSML\uff5c marker", "length"),
        ("reason</think>answer<\uff5cbegin\u2581of\u2581sentence\uff5c>", "stop"),
    ],
)
def test_malformed_protocol_never_falls_back_to_text(
    native: tuple[Any, DeepSeekV4TITORenderer], text: str, finish: str
) -> None:
    tokenizer, renderer = native
    request = TITOChatRequest.from_openai({"messages": [{"role": "user", "content": "run"}]})
    ids = tokenizer.encode(text, add_special_tokens=False)
    with pytest.raises(ValueError):
        renderer.parse_assistant(request, ids, text, finish)


def test_length_stop_inside_reasoning_is_not_visible_content(native: tuple[Any, DeepSeekV4TITORenderer]) -> None:
    tokenizer, renderer = native
    request = TITOChatRequest.from_openai({"messages": [{"role": "user", "content": "run"}]})
    parsed = renderer.parse_assistant(request, tokenizer.encode("unfinished", add_special_tokens=False), "", "length")
    assert parsed.message == {"role": "assistant", "content": "", "reasoning_content": "unfinished"}


def test_partial_native_tool_block_cannot_execute_at_length_limit(native: tuple[Any, DeepSeekV4TITORenderer]) -> None:
    tokenizer, renderer = native
    assistant = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "reason",
        "tool_calls": [{"type": "function", "function": {"name": "run", "arguments": '{"z":"hello","a":2}'}}],
    }
    text = (
        renderer._encoder.render_message(
            1, [{"role": "user", "content": "run"}, assistant], "thinking", drop_thinking=False
        )
        .removeprefix("<think>")
        .removesuffix(renderer._encoder.eos_token)
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    closing_marker = text[text.rfind("</") :].rstrip()
    request = TITOChatRequest.from_openai({"messages": [{"role": "user", "content": "run"}], "tools": [TOOL]})
    for end in range(1, len(ids)):
        prefix = ids[:end]
        try:
            parsed = renderer.parse_assistant(request, prefix, "", "length")
        except ValueError:
            continue
        if closing_marker not in tokenizer.decode(prefix):
            assert not parsed.message.get("tool_calls")


@pytest.mark.parametrize("echo_indices", [False, True])
def test_admitted_empty_refusal_marker_preserves_wire_and_prompt(
    native: tuple[Any, DeepSeekV4TITORenderer],
    echo_indices: bool,
) -> None:
    _, renderer = native
    messages = _tool_history('{"z":"hello","a":2}')
    expected = renderer.render_conversation_tokens(
        TITOChatRequest.from_openai({"messages": messages, "tools": [TOOL]})
    )
    messages[1]["provider_specific_fields"] = {"refusal": None}
    if echo_indices:
        messages[1]["tool_calls"][0]["index"] = 0
    payload = {"messages": messages, "tools": [TOOL]}
    request = TITOChatRequest.from_openai(payload)
    assert "provider_specific_fields" not in request.messages[1]
    assert "index" not in request.messages[1]["tool_calls"][0]
    assert renderer.render_conversation_tokens(request) == expected
    assert request.wire_value() == payload
    assert request.wire_value()["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"z":"hello","a":2}'


@pytest.mark.parametrize("metadata", [None, {}, {"refusal": "declined"}, {"refusal": None, "other": True}])
def test_does_not_silently_drop_other_provider_metadata(
    native: tuple[Any, DeepSeekV4TITORenderer],
    metadata: Any,
) -> None:
    _, renderer = native
    messages = _tool_history('{"z":"hello","a":2}')
    messages[1]["provider_specific_fields"] = metadata
    request = TITOChatRequest.from_openai({"messages": messages})
    with pytest.raises(TITOError, match="unsupported DeepSeek V4 message fields"):
        renderer.render_conversation_tokens(request)


@pytest.mark.parametrize("role", ["assistant", "system", "developer", "latest_reminder"])
def test_rejects_requests_without_a_reviewed_generation_boundary(
    native: tuple[Any, DeepSeekV4TITORenderer], role: str
) -> None:
    _, renderer = native
    with pytest.raises(TITOError):
        renderer.render_conversation_tokens(
            TITOChatRequest.from_openai({"messages": [{"role": role, "content": "prefill"}]})
        )


@pytest.mark.parametrize("arguments", ["broken", "[]", "null", 42])
def test_invalid_historical_arguments_do_not_use_encoder_fallback(
    native: tuple[Any, DeepSeekV4TITORenderer], arguments: Any
) -> None:
    _, renderer = native
    request = TITOChatRequest.from_openai({"messages": _tool_history(arguments)})
    with pytest.raises(TITOError):
        renderer.render_conversation_tokens(request)


def test_parallel_results_and_object_arguments_match_native_encoder(
    native: tuple[Any, DeepSeekV4TITORenderer],
) -> None:
    tokenizer, renderer = native
    messages = [
        {"role": "user", "content": "Run"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "reason",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "run", "arguments": {"z": "hello", "a": number}},
                }
                for call_id, number in (("c1", 1), ("c2", 2))
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "two"},
        {"role": "tool", "tool_call_id": "c1", "content": "one"},
    ]
    payload = {"messages": messages, "tools": [TOOL]}
    expected = copy.deepcopy(messages)
    expected[1]["content"] = ""
    for call in expected[1]["tool_calls"]:
        call["function"]["arguments"] = json.dumps(call["function"]["arguments"], ensure_ascii=False)
    expected.insert(0, {"role": "system", "content": "", "tools": [TOOL]})
    text = renderer._encoder.encode_messages(
        expected, thinking_mode="thinking", drop_thinking=False, reasoning_effort=renderer.reasoning_effort
    )
    request = TITOChatRequest.from_openai(payload)
    assert tuple(renderer.render_conversation_tokens(request)) == tuple(
        tokenizer.encode(text, add_special_tokens=False)
    )
    assert request.wire_value() == payload


@pytest.mark.parametrize("extra", [{"reasoning_effort": "max"}, {"drop_thinking": True}])
def test_rejects_dynamic_template_overrides(native: tuple[Any, DeepSeekV4TITORenderer], extra: dict[str, Any]) -> None:
    _, renderer = native
    with pytest.raises(TITOError):
        renderer.render_conversation_tokens(
            TITOChatRequest.from_openai({"messages": [{"role": "user", "content": "run"}], **extra})
        )


def test_native_bundle_loads_offline_and_rejects_missing_or_modified_encoder(
    native: tuple[Any, DeepSeekV4TITORenderer], tmp_path: Path
) -> None:
    tokenizer, renderer = native
    name = renderer.renderer_id
    setup = SimpleNamespace(
        tokenizer=tokenizer,
        tokenizer_id=tokenizer.name_or_path,
        model="policy",
    )
    bundle = build_sidecar_bundle(setup, tmp_path / "bundles", renderer_name=name)
    assert build_sidecar_bundle(setup, tmp_path / "bundles", renderer_name=name).digest == bundle.digest
    with zipfile.ZipFile(bundle.path) as archive:
        archive.extractall(tmp_path / "runtime")
    root = tmp_path / "runtime/tokenizer"
    loaded = load_sidecar_tokenizer(root, renderer_name=name)
    build_sidecar_tito_renderer(loaded, name)
    if not tokenizer.chat_template:
        with pytest.raises(ValueError, match="no bundled chat template"):
            load_sidecar_tokenizer(root)
    encoder = root / renderer.encoder_file
    original = encoder.read_bytes()
    encoder.write_bytes(original + b"\n# changed\n")
    with pytest.raises(ValueError, match="does not match certification"):
        load_sidecar_tokenizer(root, renderer_name=name)
    encoder.unlink()
    with pytest.raises(ValueError, match="missing a certified renderer artifact"):
        load_sidecar_tokenizer(root, renderer_name=name)


@pytest.fixture(scope="module")
def v41() -> DeepSeekV4TITORenderer:
    pytest.importorskip("transformers")
    # lazy: tokenizer loading requires the optional transformers dependency.
    from training.utils.tokenizers import load_tokenizer

    tokenizer = load_tokenizer(
        DEEPSEEK_V41_REPOSITORY, tokenizer_revision=DEEPSEEK_V41_REVISION, trust_remote_code=False
    )
    return build_sidecar_tito_renderer(tokenizer, DEEPSEEK_V41_RENDERER_NAME)


@pytest.mark.parametrize(
    "value", ["<image>x</image>", "<\uff5cdeepseek_image\uff5c>", "namespace", "function_namespace", "qualified"]
)
def test_v41_rejects_images_and_namespaces(v41: DeepSeekV4TITORenderer, value: str) -> None:
    payload = {"messages": [{"role": "user", "content": "run"}], "tools": [copy.deepcopy(TOOL)]}
    if value == "namespace":
        payload["tools"][0]["namespace"] = "shell"
    elif value == "function_namespace":
        payload["tools"][0]["function"]["namespace"] = "shell"
    elif value == "qualified":
        payload["tools"][0]["function"]["name"] = "shell::run"
    else:
        payload["messages"][0]["content"] = value
    with pytest.raises(TITOError) as error:
        v41.render_conversation_tokens(TITOChatRequest.from_openai(payload))
    assert error.value.status == 400
