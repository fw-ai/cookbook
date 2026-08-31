from __future__ import annotations

import json

import pytest

from fireworks.training.sdk import TITOChatRequest
from training.tito import renderer as renderer_runtime
from training.tito.renderer import (
    GLM52TITORenderer,
    TITORendererCertification,
    build_sidecar_tito_renderer,
    get_tito_renderer_certification,
)


class _Tokenizer:
    special_tokens_map: dict[str, str] = {}
    chat_template = "test-template"
    _pieces = {
        1: "<|user|>",
        2: "<|observation|>",
        3: "<think>",
        10: "reason",
        11: "</think>",
        12: "\n",
        20: "<tool_call>echo",
        21: "<arg_key>message</arg_key>",
        22: "<arg_value>green</arg_value>",
        23: "</tool_call>",
        30: "answer",
    }

    def __init__(self) -> None:
        self.render_calls: list[tuple[list[dict], dict]] = []

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        special = {
            "<|user|>": [1],
            "<|observation|>": [2],
            "<think>": [3],
        }
        if text in special:
            return special[text]
        tokens: list[int] = []
        index = 0
        while index < len(text):
            for marker, marker_tokens in special.items():
                if text.startswith(marker, index):
                    tokens.extend(marker_tokens)
                    index += len(marker)
                    break
            else:
                tokens.append(1000 + ord(text[index]))
                index += 1
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return "".join(
            chr(token - 1000) if token >= 1000 else self._pieces[token]
            for token in tokens
        )

    def apply_chat_template(self, messages: list[dict], **kwargs):
        self.render_calls.append((messages, kwargs))
        if kwargs["tokenize"]:
            return [90, 91]
        output: list[str] = []
        for message in messages:
            role = message["role"]
            marker = {
                "system": "<|system|>",
                "user": "<|user|>",
                "assistant": "<|assistant|>",
                "tool": "<|observation|>",
            }[role]
            output.extend((marker, str(message.get("content") or "")))
            for call in message.get("tool_calls") or ():
                function = call.get("function") or {}
                output.append(f"<tool_call>{function.get('name', '')}</tool_call>")
        if kwargs["add_generation_prompt"]:
            output.append("<|assistant|><think>")
        return "".join(output)


def _renderer() -> tuple[GLM52TITORenderer, _Tokenizer]:
    tokenizer = _Tokenizer()
    renderer = GLM52TITORenderer(
        tokenizer,
        certification=TITORendererCertification(
            certification_id="test",
            renderer_names=frozenset({"glm_moe_dsa_preserve_thinking"}),
            tokenizer_fingerprint="test",
            renderer_factory=lambda tokenizer, certification: GLM52TITORenderer(
                tokenizer,
                certification=certification,
            ),
        ),
    )
    return renderer, tokenizer


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                },
                "description": "Echo one message.",
                "name": "echo",
            },
        }
    ]


def test_full_history_render_delegates_once_to_the_chat_template() -> None:
    renderer, tokenizer = _renderer()
    payload = {
        "model": "policy",
        "messages": [
            {"role": "user", "content": "call echo"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": '{"message":"green"}',
                        },
                    }
                ],
            },
        ],
        "tools": _tools(),
    }
    request = TITOChatRequest.from_openai(
        payload,
        wire_request_body=json.dumps(payload),
    )

    assert tuple(renderer.render_conversation_tokens(request)) == (90, 91)
    assert len(tokenizer.render_calls) == 1
    messages, kwargs = tokenizer.render_calls[0]
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"message": "green"}
    assert list(kwargs["tools"][0]["function"]) == [
        "name",
        "description",
        "parameters",
    ]
    assert kwargs == {
        "tools": kwargs["tools"],
        "tokenize": True,
        "add_generation_prompt": True,
        "clear_thinking": False,
        "reasoning_effort": "max",
    }


def test_renderer_owns_string_stop_conversion() -> None:
    renderer, _tokenizer = _renderer()
    request = TITOChatRequest(
        messages=({"role": "user", "content": "q"},),
    )
    assert tuple(renderer.stop_sequences(request)) == (
        "<|user|>",
        "<|observation|>",
    )


def test_incremental_render_uses_anchor_and_deduplicates_role_boundary() -> None:
    renderer, tokenizer = _renderer()
    payload = {
        "model": "policy",
        "messages": [
            {"role": "user", "content": "call echo"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": '{"message":"green"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "green", "tool_call_id": "call-1"},
        ],
        "tools": _tools(),
    }
    request = TITOChatRequest.from_openai(payload)
    checkpoint = (40, 41, 2)

    prepared = renderer.prepare_incremental_prompt(
        request,
        request.messages[:2],
        request.messages[2:],
        checkpoint,
    )

    assert prepared is not None
    assert prepared.prompt_ids[: len(checkpoint)] == checkpoint
    assert prepared.prompt_ids[len(checkpoint)] != 2
    assert prepared.junction_kind == "deduplicate_role_boundary"
    assert prepared.checkpoint_trim_tokens == 1
    assert prepared.contract_id == "test:incremental-v1"
    assert [call[1]["tokenize"] for call in tokenizer.render_calls] == [False, False]


def test_incremental_render_replaces_ambiguous_sampled_role_boundary() -> None:
    renderer, _ = _renderer()
    request = TITOChatRequest(
        messages=(
            {"role": "user", "content": "call echo"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "echo", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "green"},
        ),
        tools=tuple(_tools()),
    )

    prepared = renderer.prepare_incremental_prompt(
        request,
        request.messages[:2],
        request.messages[2:],
        (40, 41, 1),
    )

    assert prepared is not None
    assert prepared.prompt_ids[:3] == (40, 41, 2)
    assert prepared.checkpoint_trim_tokens == 1
    assert prepared.junction_kind == "replace_role_boundary"


def test_parser_preserves_reasoning_and_valid_tool_call() -> None:
    renderer, _ = _renderer()
    request = TITOChatRequest(
        messages=({"role": "user", "content": "call echo"},),
        tools=tuple(_tools()),
    )

    parsed = renderer.parse_assistant(
        request,
        [10, 11, 12, 20, 21, 22, 23, 1],
        "",
        "stop",
    )

    assert parsed.output_kind == "tool_calls"
    assert parsed.message["reasoning_content"] == "reason"
    call = parsed.message["tool_calls"][0]
    assert call["id"].startswith("call_")
    assert call["function"] == {
        "name": "echo",
        "arguments": '{"message":"green"}',
    }


def test_parser_rejects_malformed_tool_markup_instead_of_repairing_it() -> None:
    renderer, _ = _renderer()
    request = TITOChatRequest(
        messages=({"role": "user", "content": "call echo"},),
        tools=tuple(_tools()),
    )

    with pytest.raises(ValueError, match="tool-call boundary"):
        renderer.parse_assistant(request, [10, 11, 12, 20, 1], "", "stop")


def test_length_stop_inside_reasoning_is_structured_but_not_visible_text() -> None:
    renderer, _ = _renderer()
    parsed = renderer.parse_assistant(
        TITOChatRequest(messages=({"role": "user", "content": "think"},)),
        [10],
        "",
        "length",
    )

    assert parsed.output_kind == "reasoning"
    assert parsed.message == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "reason",
    }


def test_uncertified_renderer_fails_closed() -> None:
    with pytest.raises(ValueError, match="no production TITO certification"):
        get_tito_renderer_certification("glm_moe_dsa_interleaved", _Tokenizer())


def test_certified_renderer_dispatches_through_its_registered_factory(
    monkeypatch,
) -> None:
    sentinel = object()
    calls = []

    def factory(tokenizer, certification):
        calls.append((tokenizer, certification))
        return sentinel

    certification = TITORendererCertification(
        certification_id="test-factory",
        renderer_names=frozenset({"test-renderer"}),
        tokenizer_fingerprint="test-fingerprint",
        renderer_factory=factory,
    )
    monkeypatch.setitem(
        renderer_runtime._TITO_CERTIFICATION_BY_RENDERER,
        "test-renderer",
        certification,
    )
    monkeypatch.setattr(
        renderer_runtime,
        "_tokenizer_fingerprint",
        lambda _tokenizer: "test-fingerprint",
    )
    tokenizer = _Tokenizer()

    assert build_sidecar_tito_renderer(tokenizer, "test-renderer") is sentinel
    assert calls == [(tokenizer, certification)]
