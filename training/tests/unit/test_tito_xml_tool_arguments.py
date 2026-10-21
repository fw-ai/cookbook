from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fireworks.training.sdk import TITOChatRequest, TITOParseError
from training.renderer.tito import (
    GLM53FlashTITORenderer,
    GLM53TITORenderer,
    Qwen35TITORenderer,
    Qwen36TITORenderer,
    Qwen38TITORenderer,
    TITORendererCertification,
    build_sidecar_tito_renderer,
)
from training.tests.unit.test_tito_qwen38_renderer import _QwenTokenizer
from training.tests.unit.test_tito_renderer import _Tokenizer


@pytest.fixture(
    params=[Qwen35TITORenderer, Qwen36TITORenderer, Qwen38TITORenderer, GLM53TITORenderer, GLM53FlashTITORenderer]
)
def renderer(request: pytest.FixtureRequest) -> Any:
    renderer_class = request.param
    tokenizer = (
        _QwenTokenizer()
        if issubclass(renderer_class, Qwen38TITORenderer)
        else _Tokenizer()
    )
    certification = TITORendererCertification(
        certification_id="test",
        renderer_names=frozenset({"test"}),
        tokenizer_fingerprint="test",
        renderer_factory=renderer_class,
    )
    return renderer_class(tokenizer, certification=certification)


def _request(schema: dict[str, Any] | bool) -> TITOChatRequest:
    return TITOChatRequest.from_openai(
        {
            "messages": [{"role": "user", "content": "Write the exact text."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "parameters": {"type": "object", "properties": {"content": schema}},
                    },
                }
            ],
        }
    )


def _completion(renderer: Any, value: str) -> str:
    if isinstance(renderer, Qwen38TITORenderer):
        return f"reason</think><tool_call><function=write_file><parameter=content>{value}</parameter></function></tool_call>"
    return f"reason</think><tool_call>write_file<arg_key>content</arg_key><arg_value>{value}</arg_value></tool_call>"


def test_renderer_rejection_exposes_a_stable_reason(renderer: Any) -> None:
    text = _completion(renderer, "hello") + "<tool_cal"
    with pytest.raises(TITOParseError) as failure:
        renderer.parse_assistant(
            _request({"type": "string"}),
            renderer.tokenizer.encode(text),
            text,
            "length",
        )
    assert failure.value.reason_code == "length_tool_trailing_content"


@pytest.mark.parametrize(
    "value",
    [
        "42",
        "true",
        "null",
        '"quoted"',
        "[1,2]",
        '{"key":1}',
        " 42 ",
        "a < b",
        "<div>x</div>",
        "<parameter=unfinished",
        "literal <parameter=other> text",
        "literal <arg_key>other</arg_key><arg_value> text",
    ],
)
def test_string_arguments_preserve_exact_text(renderer: Any, value: str) -> None:
    text = _completion(renderer, value)
    parsed = renderer.parse_assistant(_request({"type": "string"}), renderer.tokenizer.encode(text), text, "stop")
    arguments = json.loads(parsed.message["tool_calls"][0]["function"]["arguments"])
    assert arguments == {"content": value}


@pytest.mark.parametrize(
    "schema, value, expected",
    [
        ({"type": "integer"}, "42", 42),
        ({"type": "boolean"}, "true", True),
        ({"type": "object"}, '{"key":1}', {"key": 1}),
        (True, "42", 42),
    ],
)
def test_nonstring_arguments_keep_json_types(
    renderer: Any, schema: dict[str, Any] | bool, value: str, expected: Any
) -> None:
    text = _completion(renderer, value)
    parsed = renderer.parse_assistant(_request(schema), renderer.tokenizer.encode(text), text, "stop")
    arguments = json.loads(parsed.message["tool_calls"][0]["function"]["arguments"])
    assert arguments == {"content": expected}


@pytest.mark.parametrize("suffix", ["", "\n", "\t \n"])
def test_complete_tool_call_at_length_limit(renderer: Any, suffix: str) -> None:
    text = _completion(renderer, "hello") + suffix
    parsed = renderer.parse_assistant(_request({"type": "string"}), renderer.tokenizer.encode(text), text, "length")
    assert parsed.output_kind == "tool_calls"


@pytest.mark.parametrize(
    "suffix",
    ["<tool_call>"[:end] for end in range(1, len("<tool_call>"))] + ["explanation", "a < b", "<div>x</div>"],
)
def test_length_stop_with_unconsumed_tail_is_rejected(renderer: Any, suffix: str) -> None:
    text = _completion(renderer, "hello") + suffix
    request = _request({"type": "string"})
    tokens = renderer.tokenizer.encode(text)
    stopped = renderer.parse_assistant(request, tokens, text, "stop")
    assert stopped.message["content"] == suffix
    assert len(stopped.message["tool_calls"]) == 1
    with pytest.raises(TITOParseError, match="trailing"):
        renderer.parse_assistant(request, tokens, text, "length")


def test_length_stop_preserves_literal_markers_before_and_inside_calls(renderer: Any) -> None:
    prefix = "literal < and <tool_cal and <div>x</div>"
    text = _completion(renderer, "<tool_cal").replace("</think>", "</think>" + prefix, 1)
    parsed = renderer.parse_assistant(_request({"type": "string"}), renderer.tokenizer.encode(text), text, "length")
    assert parsed.message["content"] == prefix
    assert json.loads(parsed.message["tool_calls"][0]["function"]["arguments"]) == {"content": "<tool_cal"}


def test_incomplete_tool_call_at_length_limit_is_rejected(renderer: Any) -> None:
    text = _completion(renderer, "hello").removesuffix("</tool_call>")
    with pytest.raises(TITOParseError):
        renderer.parse_assistant(_request({"type": "string"}), renderer.tokenizer.encode(text), text, "length")


def test_length_stop_with_complete_and_incomplete_calls_is_rejected(renderer: Any) -> None:
    text = _completion(renderer, "hello")
    text += (
        _completion(renderer, "world")
        .split("</think>", 1)[1]
        .removesuffix("</tool_call>")
    )
    with pytest.raises(TITOParseError):
        renderer.parse_assistant(_request({"type": "string"}), renderer.tokenizer.encode(text), text, "length")


def test_live_qwen35_completion_tokens_at_length_limit() -> None:
    transformers = pytest.importorskip("transformers")
    huggingface_hub = pytest.importorskip("huggingface_hub")
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/tito_qwen35_complete_length.json").read_text())
    tokenizer_path = huggingface_hub.try_to_load_from_cache(
        fixture["tokenizer_repository"], "tokenizer.json", revision=fixture["tokenizer_revision"]
    )
    if not isinstance(tokenizer_path, str):
        pytest.skip("pinned Qwen3.5 tokenizer is not cached")
    try:
        tokenizer = transformers.TokenizersBackend.from_pretrained(
            fixture["tokenizer_repository"],
            revision=fixture["tokenizer_revision"],
            local_files_only=True,
            trust_remote_code=False,
        )
    except OSError:
        pytest.skip("pinned Qwen3.5 tokenizer is not cached")
    renderer = build_sidecar_tito_renderer(tokenizer, "qwen3_5")
    parsed = renderer.parse_assistant(
        TITOChatRequest.from_openai(fixture["request"]),
        fixture["completion_token_ids"],
        fixture["completion_text"],
        fixture["finish_reason"],
    )
    assert fixture["finish_reason"] == "length"
    assert parsed.output_kind == "tool_calls"
    assert json.loads(parsed.message["tool_calls"][0]["function"]["arguments"]) == {
        "path": "value.txt",
        "content": "42",
    }
