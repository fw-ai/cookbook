"""Null assistant content regressions against pinned GLM templates."""

from __future__ import annotations

import json
from typing import Any

import pytest

from fireworks.training.sdk import TITOChatRequest
from training.renderer.tito import GLM52TITORenderer, GLM53TITORenderer, TITORendererCertification
from training.renderer.tito.shared import _tokenizer_fingerprint


@pytest.fixture(
    scope="module",
    params=[
        ("zai-org/GLM-5.2", "b4734de4facf877f85769a911abafc5283eab3d9", "glm_moe_dsa_preserve_thinking"),
        ("zai-org/GLM-5.3", "935644c05e76fc198714f4cca449fd8b970ff6d7", "glm53_preserve_thinking"),
    ],
)
def glm(request: pytest.FixtureRequest) -> tuple[Any, GLM52TITORenderer]:
    transformers = pytest.importorskip("transformers")
    model, revision, name = request.param
    tokenizer = transformers.TokenizersBackend.from_pretrained(model, revision=revision)
    renderer_type = GLM52TITORenderer if name == "glm_moe_dsa_preserve_thinking" else GLM53TITORenderer
    # Exercise the pinned template primitive, not production registry admission.
    certification = TITORendererCertification(
        certification_id=f"test:{model}@{revision}",
        renderer_names=frozenset({name}),
        tokenizer_fingerprint=_tokenizer_fingerprint(tokenizer),
        renderer_factory=renderer_type,
    )
    return tokenizer, renderer_type(tokenizer, certification=certification)


@pytest.mark.parametrize("content", [None, "", "None"])
@pytest.mark.parametrize("arguments", ['{"z":1,"a":{"y":2,"b":3}}', '{"a":{"b":3,"y":2},"z":1}'])
def test_glm_tool_history_normalizes_null_without_reordering_arguments(
    glm: tuple[Any, GLM52TITORenderer], content: str | None, arguments: str
) -> None:
    tokenizer, renderer = glm
    payload = {
        "messages": [
            {"role": "user", "content": "Use run"},
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {"id": "call-1", "type": "function", "function": {"name": "run", "arguments": arguments}},
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
        ]
    }
    request = TITOChatRequest.from_openai(payload)
    expected_messages = [
        payload["messages"][0],
        {
            **payload["messages"][1],
            "content": "" if content is None else content,
            "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": "run", "arguments": json.loads(arguments)}},
            ],
        },
        payload["messages"][2],
    ]
    expected = tokenizer.apply_chat_template(
        expected_messages,
        tokenize=False,
        add_generation_prompt=True,
        clear_thinking=False,
        reasoning_effort="max",
    )
    actual = renderer.render_conversation_tokens(request)
    assert tuple(actual) == tuple(tokenizer.encode(expected, add_special_tokens=False))
    assert request.wire_value() == payload
    text = tokenizer.decode(list(actual))
    assert ("None" in text) == (content == "None")
    ordered_keys = list(json.loads(arguments))
    assert text.index(f"<arg_key>{ordered_keys[0]}</arg_key>") < text.index(f"<arg_key>{ordered_keys[1]}</arg_key>")
