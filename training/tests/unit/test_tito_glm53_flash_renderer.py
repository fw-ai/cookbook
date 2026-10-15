from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from fireworks.training.sdk import TITOChatRequest, TITOError
from training.renderer.tito import build_sidecar_tito_renderer
from training.renderer.tito.glm53 import GLM53TITORenderer
from training.renderer.tito.glm53_flash import (
    GLM53_FLASH_RENDERER_NAME,
    GLM53_FLASH_REPOSITORY,
    GLM53_FLASH_REVISION,
    GLM53FlashTITORenderer,
)

TOOL = {
    "type": "function",
    "function": {
        "name": "run",
        "description": "Run a command",
        "parameters": {"type": "object", "properties": {"z": {"type": "string"}, "a": {"type": "integer"}}},
    },
}


def _history(content: Any = None, arguments: str = '{"z":"hello","a":2}') -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "Run"},
        {
            "role": "assistant",
            "content": content,
            "reasoning_content": "ordered reasoning",
            "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": "run", "arguments": arguments}}
                for cid in ("c1", "c2")
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "one"},
        {"role": "tool", "tool_call_id": "c2", "content": "two"},
        {"role": "user", "content": "Continue"},
    ]


@pytest.fixture(scope="module")
def flash() -> tuple[Any, GLM53FlashTITORenderer]:
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.TokenizersBackend.from_pretrained(
        GLM53_FLASH_REPOSITORY, revision=GLM53_FLASH_REVISION, trust_remote_code=False
    )
    renderer = build_sidecar_tito_renderer(tokenizer, GLM53_FLASH_RENDERER_NAME)
    assert renderer.renderer_id == GLM53_FLASH_RENDERER_NAME
    assert not isinstance(renderer, GLM53TITORenderer)
    with pytest.raises(ValueError, match="does not match"):
        build_sidecar_tito_renderer(tokenizer, "glm53_preserve_thinking")
    return tokenizer, renderer


@pytest.mark.parametrize("content", [None, "", "None"])
@pytest.mark.parametrize("reverse", [False, True])
def test_history_matches_hf_template(flash: tuple[Any, GLM53FlashTITORenderer], content: Any, reverse: bool) -> None:
    tokenizer, renderer = flash
    messages = _history(content)
    if reverse:
        messages[2:4] = messages[2:4][::-1]
    payload = {"messages": messages, "tools": [TOOL]}
    expected = copy.deepcopy(messages)
    expected[1]["content"] = "" if content is None else content
    for call in expected[1]["tool_calls"]:
        call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    ids = tokenizer.apply_chat_template(
        expected,
        tools=[TOOL],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        clear_thinking=False,
        reasoning_effort="max",
    )
    request = TITOChatRequest.from_openai(payload)
    assert tuple(renderer.render_conversation_tokens(request)) == tuple(ids)
    assert request.wire_value() == payload
    text = tokenizer.decode(ids, skip_special_tokens=False)
    assert "ordered reasoning" in text
    assert text.index("<tool_response>one") < text.index("<tool_response>two")
    assert renderer.prepare_incremental_prompt() is None


@pytest.mark.parametrize("stop", ["", "<|user|>", "<|observation|>"])
def test_parallel_tool_parse(flash: tuple[Any, GLM53FlashTITORenderer], stop: str) -> None:
    tokenizer, renderer = flash
    text = "reason</think><tool_call>run<arg_key>z</arg_key><arg_value>hello</arg_value><arg_key>a</arg_key><arg_value>2</arg_value></tool_call>"
    text += "<tool_call>run<arg_key>z</arg_key><arg_value>world</arg_value></tool_call>" + stop
    request = TITOChatRequest.from_openai({"messages": [{"role": "user", "content": "Run"}], "tools": [TOOL]})
    ids = tokenizer.encode(text, add_special_tokens=False)
    parsed = renderer.parse_assistant(request, ids, "ignored", "stop")
    calls = parsed.response_message["tool_calls"]
    assert len(calls) == len({call["id"] for call in calls}) == 2
    assert parsed.message["reasoning_content"] == "reason"
    assert list(json.loads(calls[0]["function"]["arguments"])) == ["z", "a"]
    assert parsed == renderer.parse_assistant(request, ids, "", "stop")


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}],
        [{"type": "text", "text": None}],
        42,
    ],
)
def test_rejects_nontext_payloads(flash: tuple[Any, GLM53FlashTITORenderer], content: Any) -> None:
    _, renderer = flash
    with pytest.raises(TITOError, match="supports text only"):
        renderer.render_conversation_tokens(
            TITOChatRequest.from_openai({"messages": [{"role": "user", "content": content}]})
        )


@pytest.mark.parametrize("arguments", ["[]", "null", "42", '"text"'])
def test_rejects_nonobject_historical_arguments(flash: tuple[Any, GLM53FlashTITORenderer], arguments: str) -> None:
    _, renderer = flash
    request = TITOChatRequest.from_openai({"messages": _history(arguments=arguments)})
    with pytest.raises(TITOError, match="must be JSON objects") as error:
        renderer.render_conversation_tokens(request)
    assert error.value.status == 400
