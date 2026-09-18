"""Tests for the pinned Muse Glimmer 30B TITO renderer certification."""

from __future__ import annotations

import json

import pytest

from fireworks.training.sdk import TITOChatRequest, TITOError
from training.renderer.tito import (
    MuseGlimmerTITORenderer,
    TITORendererCertification,
    build_sidecar_tito_renderer,
    get_tito_renderer_certification,
)

_MUSE_GLIMMER_REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"

# Captured from a live Muse Glimmer 30B RFT probe deployment on
# 2026-09-18 via /inference/v1/completions with the renderer-rendered token
# prompt: a ``to=self`` reasoning channel, then a ``to=weather.get`` tool
# channel carrying one ATEM call, stopping at ``<|eot|>`` (finish_reason=stop;
# the server strips the stop token from the returned text, so the parser's
# append-stop retry path produces the clean parse).
_PROBE_TOOL_COMPLETION = (
    " to=self<|message|>What is the weather in Tokyo today? Use the weather "
    "tool.\n\nWe need to use weather.get with city = Tokyo. Probably "
    '"Tokyo". Use tool.\n\nWe must follow guidelines: call function.'
    "<|eom|><|start|>assistant to=weather.get<|message|>"
    '<atem:function_calls>\n<atem:invoke name="weather.get">\n'
    '<atem:parameter name="city">Tokyo</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls>"
)

# Same structure with narrated text on the tool channel: the model family's
# registered divergence. The parser drops it, mirroring the upstream
# template's own re-render of tool-call turns.
_NARRATED_TOOL_COMPLETION = (
    " to=self<|message|>The user wants the weather for Tokyo.<|eom|>"
    "<|start|>assistant to=weather.get<|message|>"
    "Let me look that up.\n"
    '<atem:function_calls>\n<atem:invoke name="weather.get">\n'
    '<atem:parameter name="city">Tokyo</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls>"
)


class _MuseTokenizer:
    special_tokens_map: dict[str, str] = {}
    chat_template = "test-template"
    _pieces = {
        1: "<|eot|>",
        2: "<|eom|>",
        3: "<|start|>",
        4: "<|message|>",
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
        certification_id="test-muse-glimmer",
        renderer_names=frozenset({"muse_glimmer"}),
        tokenizer_fingerprint="test",
        renderer_factory=lambda tokenizer, certification: MuseGlimmerTITORenderer(
            tokenizer,
            certification=certification,
        ),
    )


def _renderer() -> tuple[MuseGlimmerTITORenderer, _MuseTokenizer]:
    tokenizer = _MuseTokenizer()
    return (
        MuseGlimmerTITORenderer(tokenizer, certification=_certification()),
        tokenizer,
    )


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "weather.get",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


def _request(**overrides) -> TITOChatRequest:
    payload = {
        "model": "policy",
        "messages": [{"role": "user", "content": "weather in Tokyo?"}],
        "tools": _tools(),
    }
    payload.update(overrides)
    return TITOChatRequest.from_openai(
        payload,
        wire_request_body=json.dumps(payload),
    )


def _ids(tokenizer: _MuseTokenizer, text: str) -> list[int]:
    return tokenizer.encode(text)


def test_render_delegates_with_pinned_defaults_only() -> None:
    renderer, tokenizer = _renderer()
    request = _request()

    assert tuple(renderer.render_conversation_tokens(request)) == (90, 91)
    assert len(tokenizer.render_calls) == 1
    _messages, kwargs = tokenizer.render_calls[0]
    # No dynamic template fields: reasoning_strength / knowledge_cutoff stay
    # at the certified template defaults.
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


def test_stop_sequences_are_eot_only() -> None:
    renderer, _tokenizer = _renderer()
    # ``<|eom|>`` continues the same sampled response; only ``<|eot|>`` is a
    # stop, matching the offline renderer's stop contract.
    assert tuple(renderer.stop_sequences(_request())) == ("<|eot|>",)


def test_parse_probe_tool_call_completion() -> None:
    """A reasoning-then-tool-call completion parses into one tool call."""
    renderer, tokenizer = _renderer()
    ids = _ids(tokenizer, _PROBE_TOOL_COMPLETION + "<|eot|>")
    parsed = renderer.parse_assistant(_request(), ids, "", "stop")

    assert parsed.output_kind == "tool_calls"
    message = parsed.message
    assert message["reasoning_content"].startswith("What is the weather")
    assert message["content"] == ""
    assert len(message["tool_calls"]) == 1
    function = message["tool_calls"][0]["function"]
    assert function["name"] == "weather.get"
    assert json.loads(function["arguments"]) == {"city": "Tokyo"}
    assert message["tool_calls"][0]["id"].startswith("call_")


def test_parse_drops_narration_on_tool_channel() -> None:
    """Narrated text on a tool channel is dropped, mirroring the template."""
    renderer, tokenizer = _renderer()
    ids = _ids(tokenizer, _NARRATED_TOOL_COMPLETION + "<|eot|>")
    parsed = renderer.parse_assistant(_request(), ids, "", "stop")

    assert "Let me look that up" not in parsed.message["content"]
    assert parsed.message["content"] == ""


def test_parse_completion_without_stop_token_retries_cleanly() -> None:
    renderer, tokenizer = _renderer()
    ids = _ids(tokenizer, _PROBE_TOOL_COMPLETION)
    parsed = renderer.parse_assistant(_request(), ids, "", "stop")
    assert parsed.output_kind == "tool_calls"


def test_parse_reasoning_only_length_stop() -> None:
    renderer, tokenizer = _renderer()
    ids = _ids(tokenizer, " to=self<|message|>still reasoning, never closed")
    parsed = renderer.parse_assistant(_request(), ids, "", "length")

    assert parsed.output_kind == "reasoning"
    assert parsed.message["content"] == ""
    assert "still reasoning" in parsed.message["reasoning_content"]


def test_parse_text_answer_after_reasoning() -> None:
    renderer, tokenizer = _renderer()
    text = (
        " to=self<|message|>working it out<|eom|>"
        "<|start|>assistant to=user<|message|>The answer is 4."
    )
    ids = _ids(tokenizer, text + "<|eot|>")
    parsed = renderer.parse_assistant(_request(), ids, "", "stop")

    # A turn carrying reasoning_content classifies as "reasoning" even with
    # visible content (engine contract, same as the GLM/Qwen renderers).
    assert parsed.output_kind == "reasoning"
    assert parsed.message["content"] == "The answer is 4."


def test_parse_rejects_segment_without_message_boundary() -> None:
    renderer, tokenizer = _renderer()
    ids = _ids(tokenizer, "no channel opened here<|eot|>")
    with pytest.raises(ValueError, match="no message boundary"):
        renderer.parse_assistant(_request(), ids, "", "stop")


def test_parse_rejects_segment_missing_recipient() -> None:
    renderer, tokenizer = _renderer()
    ids = _ids(tokenizer, "user<|message|>hi<|eot|>")
    with pytest.raises(ValueError, match="missing its recipient"):
        renderer.parse_assistant(_request(), ids, "", "stop")


def test_parse_rejects_unknown_recipient() -> None:
    renderer, tokenizer = _renderer()
    ids = _ids(tokenizer, " to=cuisine.chat<|message|>hi<|eot|>")
    with pytest.raises(ValueError, match="unknown Muse Glimmer recipient"):
        renderer.parse_assistant(_request(), ids, "", "stop")


def test_parse_rejects_tool_name_absent_from_request() -> None:
    renderer, tokenizer = _renderer()
    text = (
        " to=self<|message|>need a tool<|eom|>"
        "<|start|>assistant to=unknown.tool<|message|><atem:function_calls>\n"
        '<atem:invoke name="unknown.tool">\n'
        '<atem:parameter name="city">Tokyo</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    ids = _ids(tokenizer, text + "<|eot|>")
    with pytest.raises(ValueError, match="absent from the request"):
        renderer.parse_assistant(_request(), ids, "", "stop")


def test_parse_rejects_truncated_tool_call_on_length_stop() -> None:
    renderer, tokenizer = _renderer()
    text = (
        " to=self<|message|>need the tool<|eom|>"
        "<|start|>assistant to=weather.get<|message|><atem:function_calls>\n"
        '<atem:invoke name="weather.get">\n'
        '<atem:parameter name="city">Tok'
    )
    ids = _ids(tokenizer, text)
    with pytest.raises(ValueError, match="unknown Muse Glimmer recipient"):
        renderer.parse_assistant(_request(), ids, "", "length")


def test_parse_rejects_unparsed_atem_residue() -> None:
    renderer, tokenizer = _renderer()
    text = (
        " to=weather.get<|message|><atem:function_calls>\n"
        '<atem:invoke name="weather.get">\n'
        "junk between parameters\n"
        '<atem:parameter name="city">Tokyo</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    ids = _ids(tokenizer, text + "<|eot|>")
    with pytest.raises(ValueError, match="unparsed Muse Glimmer tool-call content"):
        renderer.parse_assistant(_request(), ids, "", "stop")


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


def test_certification_registry_resolves_muse_glimmer(monkeypatch) -> None:
    monkeypatch.setattr(
        "training.renderer.tito.shared._tokenizer_fingerprint",
        lambda _tokenizer: (
            "2fec80a849b8cb52120e297e001ce675c0ee1b0de067aed449d7f1e756b04e3a"
        ),
    )
    renderer = build_sidecar_tito_renderer(_MuseTokenizer(), "muse_glimmer")
    assert isinstance(renderer, MuseGlimmerTITORenderer)
    assert renderer.certification_id == "muse-glimmer-30b-preserved@a4e59da5-v1"


def test_certification_rejects_tokenizer_fingerprint_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        "training.renderer.tito.shared._tokenizer_fingerprint",
        lambda _tokenizer: "wrong-fingerprint",
    )
    with pytest.raises(ValueError, match="does not match TITO certification"):
        build_sidecar_tito_renderer(_MuseTokenizer(), "muse_glimmer")


def test_unknown_renderer_still_fails_closed() -> None:
    with pytest.raises(ValueError, match="no production TITO certification"):
        get_tito_renderer_certification("muse_glimmer_vision", _MuseTokenizer())


@pytest.mark.timeout(180)
def test_pinned_tokenizer_fingerprint_matches_certification() -> None:
    """The certification pins the meta-models/Muse-Glimmer-30B tokenizer contract.

    Load through ``load_tokenizer`` with the pinned revision so the backend
    hash matches the sidecar's bundled artifact, exactly as production loads
    it.
    """
    pytest.importorskip("transformers")
    try:
        from training.renderer.tito.shared import _tokenizer_fingerprint
        from training.utils.tokenizers import load_tokenizer

        tokenizer = load_tokenizer(
            "meta-models/Muse-Glimmer-30B", _MUSE_GLIMMER_REVISION, False
        )
        fingerprint = _tokenizer_fingerprint(tokenizer)
    except (OSError, ValueError, RuntimeError) as exc:
        pytest.skip(f"tokenizer unavailable: {exc}")

    certification = get_tito_renderer_certification("muse_glimmer", tokenizer)
    assert fingerprint == certification.tokenizer_fingerprint


@pytest.mark.timeout(180)
def test_render_matches_hf_chat_template_token_for_token() -> None:
    """render_conversation_tokens must equal the authoritative HF render."""
    pytest.importorskip("transformers")
    try:
        from training.utils.tokenizers import load_tokenizer

        tokenizer = load_tokenizer(
            "meta-models/Muse-Glimmer-30B", _MUSE_GLIMMER_REVISION, False
        )
    except (OSError, ValueError, RuntimeError) as exc:
        pytest.skip(f"tokenizer unavailable: {exc}")

    renderer = build_sidecar_tito_renderer(tokenizer, "muse_glimmer")
    tools = _tools()
    messages = [
        {"role": "system", "content": "You are a weather analysis agent."},
        {"role": "user", "content": "What is the weather in Tokyo?"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should look up the weather for Tokyo.",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {
                        "name": "weather.get",
                        "arguments": '{"city": "Tokyo"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny, 24C"},
        {"role": "user", "content": "and tomorrow?"},
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
