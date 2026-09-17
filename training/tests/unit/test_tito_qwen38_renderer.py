"""Tests for the pinned Qwen3.8-27B TITO renderer certification."""

from __future__ import annotations

import json

import pytest

from fireworks.training.sdk import TITOChatRequest, TITOError
from training.renderer.tito import (
    Qwen38TITORenderer,
    TITORendererCertification,
    build_sidecar_tito_renderer,
    get_tito_renderer_certification,
)

_QWEN38_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

# Captured from a live Qwen3.8-27B RFT deployment on 2026-09-16 via
# /v1/completions with the template-rendered prompt: reasoning continues the
# prompt-opened <think>, closes with </think>, then emits one function= tool
# call, stopping at <|im_end|> (finish_reason=stop).
_PROBE_TOOL_COMPLETION = (
    "The user is asking to use a tool to calculate 2+2. This is a simple "
    "request. Let's use the execute_python tool to calculate 2+2.\n"
    "</think>\n\n"
    "<tool_call>\n<function=execute_python>\n<parameter=code>\nprint(2 + 2)\n"
    "</parameter>\n</function>\n</tool_call>"
)


class _QwenTokenizer:
    special_tokens_map: dict[str, str] = {}
    chat_template = "test-template"
    _pieces = {
        1: "<|im_end|>",
        2: "<|im_start|>",
        3: "<|endoftext|>",
        4: "<think>",
        5: "</think>",
    }
    _specials = {piece: token for token, piece in _pieces.items()}

    def __init__(self) -> None:
        self.render_calls: list[tuple[list[dict], dict]] = []

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        tokens: list[int] = []
        index = 0
        while index < len(text):
            for marker, marker_token in self._specials.items():
                if text.startswith(marker, index):
                    tokens.append(marker_token)
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
        return "".join(str(message.get("content") or "") for message in messages)


def _certification() -> TITORendererCertification:
    return TITORendererCertification(
        certification_id="test-qwen38",
        renderer_names=frozenset({"qwen3_8"}),
        tokenizer_fingerprint="test",
        renderer_factory=lambda tokenizer, certification: Qwen38TITORenderer(
            tokenizer,
            certification=certification,
        ),
    )


def _renderer() -> tuple[Qwen38TITORenderer, _QwenTokenizer]:
    tokenizer = _QwenTokenizer()
    return (
        Qwen38TITORenderer(tokenizer, certification=_certification()),
        tokenizer,
    )


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "execute_python",
                "description": "Run python code and return stdout",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }
    ]


def _request(**overrides) -> TITOChatRequest:
    payload = {
        "model": "policy",
        "messages": [{"role": "user", "content": "compute 2+2"}],
        "tools": _tools(),
    }
    payload.update(overrides)
    return TITOChatRequest.from_openai(
        payload,
        wire_request_body=json.dumps(payload),
    )


def test_render_delegates_with_pinned_defaults_only() -> None:
    renderer, tokenizer = _renderer()
    request = _request()

    assert tuple(renderer.render_conversation_tokens(request)) == (90, 91)
    assert len(tokenizer.render_calls) == 1
    _messages, kwargs = tokenizer.render_calls[0]
    # No dynamic template fields: enable_thinking / preserve_thinking /
    # reasoning_effort stay at the certified template defaults.
    assert kwargs == {
        "tools": kwargs["tools"],
        "tokenize": True,
        "add_generation_prompt": True,
    }


def test_render_rejects_dynamic_template_fields() -> None:
    renderer, _tokenizer = _renderer()
    request = _request(reasoning_effort="low")
    with pytest.raises(TITOError, match="unsupported per-request fields"):
        renderer.render_conversation_tokens(request)


def test_stop_sequences_are_im_end_and_endoftext() -> None:
    renderer, _tokenizer = _renderer()
    assert tuple(renderer.stop_sequences(_request())) == (
        "<|im_end|>",
        "<|endoftext|>",
    )


def test_parse_probe_tool_call_completion() -> None:
    """The live-deployment probe completion parses into one tool call."""
    renderer, tokenizer = _renderer()
    ids = tokenizer.encode(_PROBE_TOOL_COMPLETION) + [
        tokenizer._specials["<|im_end|>"]
    ]
    parsed = renderer.parse_assistant(_request(), ids, "", "stop")

    assert parsed.output_kind == "tool_calls"
    message = parsed.message
    assert message["reasoning_content"].startswith("The user is asking")
    assert message["content"] == ""
    assert len(message["tool_calls"]) == 1
    function = message["tool_calls"][0]["function"]
    assert function["name"] == "execute_python"
    assert json.loads(function["arguments"]) == {"code": "print(2 + 2)"}
    assert message["tool_calls"][0]["id"].startswith("call_")


def test_parse_completion_without_stop_token_retries_cleanly() -> None:
    renderer, tokenizer = _renderer()
    ids = tokenizer.encode(_PROBE_TOOL_COMPLETION)
    parsed = renderer.parse_assistant(_request(), ids, "", "stop")
    assert parsed.output_kind == "tool_calls"


def test_parse_reasoning_only_length_stop() -> None:
    renderer, tokenizer = _renderer()
    ids = tokenizer.encode("still reasoning, never closed")
    parsed = renderer.parse_assistant(_request(), ids, "", "length")

    assert parsed.output_kind == "reasoning"
    assert parsed.message["content"] == ""
    assert "still reasoning" in parsed.message["reasoning_content"]


def test_parse_text_answer_after_reasoning() -> None:
    renderer, tokenizer = _renderer()
    text = "working it out\n</think>\n\nThe answer is 4."
    ids = tokenizer.encode(text) + [tokenizer._specials["<|im_end|>"]]
    parsed = renderer.parse_assistant(_request(), ids, "", "stop")

    # A turn carrying reasoning_content classifies as "reasoning" even with
    # visible content (engine contract, same as the GLM renderer).
    assert parsed.output_kind == "reasoning"
    assert parsed.message["content"] == "The answer is 4."


def test_parse_rejects_reasoning_open_without_close_on_clean_stop() -> None:
    renderer, tokenizer = _renderer()
    text = "never closed reasoning"
    ids = tokenizer.encode(text) + [tokenizer._specials["<|im_end|>"]]
    with pytest.raises(ValueError, match="no closing boundary"):
        renderer.parse_assistant(_request(), ids, "", "stop")


def test_parse_rejects_tool_name_absent_from_request() -> None:
    renderer, tokenizer = _renderer()
    text = (
        "thinking\n</think>\n\n<tool_call>\n<function=unknown_tool>\n"
        "</function>\n</tool_call>"
    )
    ids = tokenizer.encode(text) + [tokenizer._specials["<|im_end|>"]]
    with pytest.raises(ValueError, match="absent from the request"):
        renderer.parse_assistant(_request(), ids, "", "stop")


def test_parse_rejects_truncated_tool_call_on_length_stop() -> None:
    renderer, tokenizer = _renderer()
    text = "thinking\n</think>\n\n<tool_call>\n<function=execute_python>\n"
    ids = tokenizer.encode(text)
    with pytest.raises(ValueError, match="unparsed Qwen3.8 tool-call boundary"):
        renderer.parse_assistant(_request(), ids, "", "length")


def test_incremental_prompt_is_not_certified() -> None:
    renderer, _tokenizer = _renderer()
    assert (
        renderer.prepare_incremental_prompt(
            _request(),
            stored_messages=[{"role": "user", "content": "q"}],
            appended_messages=[{"role": "assistant", "content": "a"}],
            exact_checkpoint_ids=[1, 2],
        )
        is None
    )


def test_certification_registry_resolves_qwen3_8(monkeypatch) -> None:
    monkeypatch.setattr(
        "training.renderer.tito.shared._tokenizer_fingerprint",
        lambda _tokenizer: (
            "90e8dd75a5fa5c8009f981975336de7177bb5ab04d41940e49c9a6e2d37325c7"
        ),
    )
    renderer = build_sidecar_tito_renderer(_QwenTokenizer(), "qwen3_8")
    assert isinstance(renderer, Qwen38TITORenderer)
    assert renderer.certification_id == "qwen3.8-27b-preserved@1d4bf0f2-v1"


def test_certification_rejects_tokenizer_fingerprint_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        "training.renderer.tito.shared._tokenizer_fingerprint",
        lambda _tokenizer: "wrong-fingerprint",
    )
    with pytest.raises(ValueError, match="does not match TITO certification"):
        build_sidecar_tito_renderer(_QwenTokenizer(), "qwen3_8")


def test_unknown_renderer_still_fails_closed() -> None:
    with pytest.raises(ValueError, match="no production TITO certification"):
        get_tito_renderer_certification("qwen3_7", _QwenTokenizer())


@pytest.mark.timeout(180)
def test_pinned_tokenizer_fingerprint_matches_certification() -> None:
    """The certification pins the Qwen/Qwen3.8-27B tokenizer contract."""
    pytest.importorskip("transformers")
    try:
        from transformers import AutoTokenizer

        from training.renderer.tito.shared import _tokenizer_fingerprint

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B", revision=_QWEN38_REVISION)
        fingerprint = _tokenizer_fingerprint(tokenizer)
    except (OSError, ValueError, RuntimeError) as exc:
        pytest.skip(f"tokenizer unavailable: {exc}")

    certification = get_tito_renderer_certification("qwen3_8", tokenizer)
    assert fingerprint == certification.tokenizer_fingerprint


@pytest.mark.timeout(180)
def test_render_matches_hf_chat_template_token_for_token() -> None:
    """render_conversation_tokens must equal the authoritative HF render."""
    pytest.importorskip("transformers")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B", revision=_QWEN38_REVISION)
    except (OSError, ValueError, RuntimeError) as exc:
        pytest.skip(f"tokenizer unavailable: {exc}")

    renderer = build_sidecar_tito_renderer(tokenizer, "qwen3_8")
    tools = _tools()
    messages = [
        {"role": "system", "content": "You are a data analysis agent."},
        {"role": "user", "content": "What is the average fee?"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should query the payments table.",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {
                        "name": "execute_python",
                        "arguments": '{"code": "print(1)"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "1"},
        {"role": "user", "content": "and the median?"},
    ]
    payload = {"model": "policy", "messages": messages, "tools": tools}
    request = TITOChatRequest.from_openai(
        payload,
        wire_request_body=json.dumps(payload),
    )

    from training.renderer.tito.shared import (
        _normalize_template_messages,
        _normalize_template_tools,
    )

    # The certified contract normalizes historical string tool arguments to a
    # mapping and fixes the tool envelope order before rendering; the HF
    # ground truth gets the same normalized inputs.
    expected = tokenizer.apply_chat_template(
        _normalize_template_messages(messages),
        tools=_normalize_template_tools(tools),
        tokenize=True,
        add_generation_prompt=True,
    )
    from collections.abc import Mapping as _Mapping

    if isinstance(expected, _Mapping):
        expected = expected["input_ids"]
    assert tuple(int(token) for token in expected) == tuple(
        renderer.render_conversation_tokens(request)
    )
