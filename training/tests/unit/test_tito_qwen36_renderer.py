"""Pinned real-tokenizer parity for Qwen3.6 preserved history."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from fireworks.training.sdk import TITOChatRequest

from training.renderer.tito import build_sidecar_tito_renderer
from training.renderer.tito.qwen36 import (
    QWEN36_RENDERER_NAME,
    QWEN36_REPOSITORY,
    QWEN36_REVISION,
    QWEN36_TOKENIZER_FINGERPRINT,
    Qwen36TITORenderer,
)
from training.renderer.tito.shared import TITORendererCertification, _tokenizer_fingerprint
from training.utils.tokenizers import load_tokenizer

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_value",
            "description": "Record an integer",
            "parameters": {"type": "object", "properties": {"number": {"type": "integer"}}, "required": ["number"]},
        },
    }
]
CALLS = [{"id": "call_1", "type": "function", "function": {"name": "record_value", "arguments": '{"number":11}'}}]


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    return load_tokenizer(QWEN36_REPOSITORY, QWEN36_REVISION, False)


def candidate_renderer(tokenizer: Any) -> Qwen36TITORenderer:
    certificate = TITORendererCertification(
        certification_id="qwen3.6-candidate",
        renderer_names=frozenset({QWEN36_RENDERER_NAME}),
        tokenizer_fingerprint=QWEN36_TOKENIZER_FINGERPRINT,
        renderer_factory=lambda tokenizer, certification: Qwen36TITORenderer(tokenizer, certification=certification),
    )
    return Qwen36TITORenderer(tokenizer, certification=certificate)


def test_actual_tokenizer_fingerprint(tokenizer: Any) -> None:
    assert _tokenizer_fingerprint(tokenizer) == QWEN36_TOKENIZER_FINGERPRINT


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "Reply OK"}],
        [{"role": "system", "content": "Be concise"}, {"role": "user", "content": "Compute 11"}],
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello", "reasoning_content": "old reasoning"},
            {"role": "user", "content": "Continue"},
        ],
        [
            {"role": "user", "content": "Record 11"},
            {"role": "assistant", "content": None, "reasoning_content": "Use the tool", "tool_calls": CALLS},
            {"role": "tool", "tool_call_id": "call_1", "content": "saved"},
        ],
        [
            {"role": "user", "content": "Record twice"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Two calls",
                "tool_calls": CALLS + [{**CALLS[0], "id": "call_2"}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "first"},
            {"role": "tool", "tool_call_id": "call_2", "content": "second"},
            {"role": "user", "content": "Continue"},
        ],
    ],
)
def test_render_matches_pinned_preserved_template(tokenizer: Any, messages: list[dict]) -> None:
    request = TITOChatRequest.from_openai({"model": "policy", "messages": messages, "tools": TOOLS})
    renderer = candidate_renderer(tokenizer)
    canonical = copy.deepcopy(messages)
    for message in canonical:
        for call in message.get("tool_calls", []):
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                call["function"]["arguments"] = json.loads(arguments)
    expected = tokenizer.apply_chat_template(
        canonical,
        tools=TOOLS,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=True,
        preserve_thinking=True,
    )
    assert list(renderer.render_conversation_tokens(request)) == expected
    if any(message.get("reasoning_content") == "old reasoning" for message in messages):
        assert "old reasoning" in tokenizer.decode(expected)
        interleaved = tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            preserve_thinking=False,
        )
        assert expected != interleaved


@pytest.mark.parametrize("parallel", [False, True])
def test_real_token_tool_parse_and_stop(tokenizer: Any, parallel: bool) -> None:
    renderer = candidate_renderer(tokenizer)
    request = TITOChatRequest.from_openai(
        {"model": "policy", "messages": [{"role": "user", "content": "Record 11"}], "tools": TOOLS}
    )
    call = "<tool_call>\n<function=record_value>\n<parameter=number>\n11\n</parameter>\n</function>\n</tool_call>"
    text = "Use the tool.\n</think>\n\n" + call + ("\n" + call if parallel else "") + "<|im_end|>"
    parsed = renderer.parse_assistant(request, tokenizer.encode(text, add_special_tokens=False), text, "stop")
    assert parsed.message["reasoning_content"].strip() == "Use the tool."
    assert len(parsed.message["tool_calls"]) == (2 if parallel else 1)
    for item in parsed.message["tool_calls"]:
        assert item["function"]["name"] == "record_value"
        assert json.loads(item["function"]["arguments"]) == {"number": 11}
    assert parsed.message["content"] in (None, "")


def test_registry_matches_pinned_pair(tokenizer: Any) -> None:
    renderer = build_sidecar_tito_renderer(tokenizer, QWEN36_RENDERER_NAME)
    assert isinstance(renderer, Qwen36TITORenderer)
    assert renderer.certification_id == "qwen3.6-35b-a3b-preserved@995ad96e-candidate"
