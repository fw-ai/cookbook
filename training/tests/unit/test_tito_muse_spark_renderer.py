"""Tests for the pinned Muse Spark 1.2 open TITO renderer certification."""

from __future__ import annotations

import json

import pytest

from fireworks.training.sdk import TITOChatRequest, TITOError
from training.renderer.tito import MuseSparkTITORenderer, build_sidecar_tito_renderer
from training.renderer.tito import shared as tito_shared
from training.renderer.tito.muse_spark import (
    MUSE_SPARK_TOKENIZER_FINGERPRINT,
    muse_spark_certifications,
)

# A reasoning channel, then a tool channel carrying one ATEM call. The server
# strips the stop token, so the parser's append-stop retry produces the clean
# parse.
_TOOL_COMPLETION = (
    " to=self<|message|>The user wants Tokyo weather. Call the tool."
    "<|eom|><|start|>assistant to=weather.get<|message|>"
    '<atem:function_calls>\n<atem:invoke name="weather.get">\n'
    '<atem:parameter name="city">Tokyo</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls>"
)

# Commentary is user-visible but does not end the turn.
_COMMENTARY_COMPLETION = (
    " to=self<|message|>Plan it.<|eom|>" "<|start|>assistant to=commentary<|message|>Working on it."
)

_TERMINAL_COMPLETION = " to=self<|message|>Simple.<|eom|><|start|>assistant to=user<|message|>Done.<|eot|>"


class _SparkTokenizer:
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
        return "".join(chr(token - 1000) if token >= 1000 else self._pieces[token] for token in tokens)

    def apply_chat_template(self, messages: list[dict], **kwargs):
        self.render_calls.append((messages, kwargs))
        if kwargs["tokenize"]:
            return [90, 91]
        return "".join(str(message.get("content") or "") for message in messages)


def _certification(effort: str = "xhigh"):
    for certification in muse_spark_certifications():
        if certification.certification_id.endswith(f"-{effort}@v2"):
            return certification
    raise AssertionError(f"no certification for {effort!r}")


def _renderer(effort: str = "xhigh") -> tuple[MuseSparkTITORenderer, _SparkTokenizer]:
    tokenizer = _SparkTokenizer()
    return (
        MuseSparkTITORenderer(
            tokenizer,
            certification=_certification(effort),
            reasoning_effort=effort,
        ),
        tokenizer,
    )


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "weather.get",
                "description": "Look up weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]


def _request(**kwargs) -> TITOChatRequest:
    kwargs.setdefault("messages", ({"role": "user", "content": "weather?"},))
    return TITOChatRequest(**kwargs)


def _encode(tokenizer: _SparkTokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def test_render_pins_reasoning_effort_and_suppresses_current_date() -> None:
    renderer, tokenizer = _renderer("high")

    assert renderer.render_conversation_tokens(_request()) == (90, 91)
    _, kwargs = tokenizer.render_calls[-1]
    assert kwargs["reasoning_effort"] == "high"
    # The template's own default calls strftime_now; a defined-but-falsy value
    # keeps a trajectory's system block stable across a UTC midnight rollover.
    assert kwargs["current_date"] is None
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["tokenize"] is True


def test_per_request_template_options_are_refused() -> None:
    renderer, _ = _renderer()
    request = _request(sampling_fields={"reasoning_effort": "low"})

    with pytest.raises(TITOError, match="fixed by the certified"):
        renderer.render_conversation_tokens(request)


def test_media_content_is_refused() -> None:
    renderer, _ = _renderer()
    request = _request(
        messages=(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                ],
            },
        )
    )

    with pytest.raises(TITOError, match="text-only"):
        renderer.render_conversation_tokens(request)


def test_reasoning_and_tool_call_parse() -> None:
    renderer, tokenizer = _renderer()
    request = _request(tools=tuple(_tools()))

    parsed = renderer.parse_assistant(
        request,
        _encode(tokenizer, _TOOL_COMPLETION),
        "",
        "stop",
    )

    assert parsed.output_kind == "tool_calls"
    assert parsed.message["reasoning_content"] == ("The user wants Tokyo weather. Call the tool.")
    assert parsed.message["content"] == ""
    (call,) = parsed.message["tool_calls"]
    assert call["function"]["name"] == "weather.get"
    assert json.loads(call["function"]["arguments"]) == {"city": "Tokyo"}
    assert call["id"].startswith("call_")


def test_commentary_recipient_is_preserved_for_faithful_rerender() -> None:
    renderer, tokenizer = _renderer()

    parsed = renderer.parse_assistant(
        _request(),
        _encode(tokenizer, _COMMENTARY_COMPLETION),
        "",
        "stop",
    )

    assert parsed.message["content"] == "Working on it."
    # Without this the template would re-render the turn as to=user/<|eot|>,
    # changing tokens the trajectory already committed to.
    assert parsed.message["recipient"] == "commentary"
    assert "tool_calls" not in parsed.message


def test_terminal_user_channel_has_no_recipient_override() -> None:
    renderer, tokenizer = _renderer()

    parsed = renderer.parse_assistant(
        _request(),
        _encode(tokenizer, _TERMINAL_COMPLETION),
        "",
        "stop",
    )

    assert parsed.output_kind == "reasoning"
    assert parsed.message["content"] == "Done."
    assert parsed.message["reasoning_content"] == "Simple."
    assert "recipient" not in parsed.message


def test_commentary_followed_by_user_fails_closed() -> None:
    """One assistant message cannot re-render two visible channels."""
    renderer, tokenizer = _renderer()
    completion = (
        " to=commentary<|message|>Looking it up.<|eom|>" "<|start|>assistant to=user<|message|>Here you go.<|eot|>"
    )

    # Collapsing these into one content string would re-render as a single
    # to=user block and silently break the stored checkpoint's token lineage.
    with pytest.raises(ValueError, match="multiple visible segments"):
        renderer.parse_assistant(
            _request(),
            _encode(tokenizer, completion),
            "",
            "stop",
        )


def test_repeated_commentary_segments_fail_closed() -> None:
    renderer, tokenizer = _renderer()
    completion = " to=commentary<|message|>One.<|eom|>" "<|start|>assistant to=commentary<|message|>Two."

    with pytest.raises(ValueError, match="multiple visible segments"):
        renderer.parse_assistant(
            _request(),
            _encode(tokenizer, completion),
            "",
            "stop",
        )


def test_reasoning_after_a_visible_segment_fails_closed() -> None:
    """The template always emits reasoning first, so this cannot round-trip."""
    renderer, tokenizer = _renderer()
    completion = (
        " to=commentary<|message|>Thinking out loud.<|eom|>"
        "<|start|>assistant to=self<|message|>Actually reconsider."
    )

    with pytest.raises(ValueError, match="reasoning after a visible segment"):
        renderer.parse_assistant(
            _request(),
            _encode(tokenizer, completion),
            "",
            "stop",
        )


def test_user_body_alongside_tool_calls_fails_closed() -> None:
    """A tool-call turn re-renders its body on the commentary channel."""
    renderer, tokenizer = _renderer()
    completion = (
        " to=user<|message|>Checking.<|eom|>"
        "<|start|>assistant to=weather.get<|message|>"
        '<atem:function_calls>\n<atem:invoke name="weather.get">\n'
        '<atem:parameter name="city">Tokyo</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )

    with pytest.raises(ValueError, match="non-commentary body with tool calls"):
        renderer.parse_assistant(
            _request(tools=tuple(_tools())),
            _encode(tokenizer, completion),
            "",
            "stop",
        )


def test_commentary_after_tool_calls_fails_closed() -> None:
    renderer, tokenizer = _renderer()
    completion = (
        " to=weather.get<|message|>"
        '<atem:function_calls>\n<atem:invoke name="weather.get">\n'
        '<atem:parameter name="city">Tokyo</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eom|>"
        "<|start|>assistant to=commentary<|message|>On it."
    )

    with pytest.raises(ValueError, match="commentary body after tool calls"):
        renderer.parse_assistant(
            _request(tools=tuple(_tools())),
            _encode(tokenizer, completion),
            "",
            "stop",
        )


def test_commentary_body_before_tool_calls_is_accepted() -> None:
    """The one body+tools shape the template does round-trip."""
    renderer, tokenizer = _renderer()
    completion = (
        " to=commentary<|message|>Let me check.<|eom|>"
        "<|start|>assistant to=weather.get<|message|>"
        '<atem:function_calls>\n<atem:invoke name="weather.get">\n'
        '<atem:parameter name="city">Tokyo</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )

    parsed = renderer.parse_assistant(
        _request(tools=tuple(_tools())),
        _encode(tokenizer, completion),
        "",
        "stop",
    )

    assert parsed.output_kind == "tool_calls"
    assert parsed.message["content"] == "Let me check."
    assert "recipient" not in parsed.message


@pytest.mark.parametrize("reserved", ["self", "commentary", "user"])
def test_reserved_tool_names_are_refused(reserved: str) -> None:
    """A tool on a reserved recipient would parse back as text, not a call."""
    renderer, _ = _renderer()
    request = _request(
        tools=({"type": "function", "function": {"name": reserved}},),
    )

    with pytest.raises(TITOError, match="reserves the"):
        renderer.render_conversation_tokens(request)


def test_unknown_recipient_fails_closed() -> None:
    renderer, tokenizer = _renderer()
    completion = " to=mystery<|message|>hello"

    with pytest.raises(ValueError, match="unknown Muse Spark recipient"):
        renderer.parse_assistant(
            _request(),
            _encode(tokenizer, completion),
            "",
            "stop",
        )


def test_tool_call_absent_from_request_fails_closed() -> None:
    renderer, tokenizer = _renderer()

    with pytest.raises(ValueError, match="absent from the request"):
        renderer.parse_assistant(
            _request(),
            _encode(tokenizer, _TOOL_COMPLETION),
            "",
            "stop",
        )


def test_truncated_tool_call_is_rejected() -> None:
    renderer, tokenizer = _renderer()
    request = _request(tools=tuple(_tools()))

    with pytest.raises(ValueError, match="not protocol-safe"):
        renderer.parse_assistant(
            request,
            _encode(tokenizer, _TOOL_COMPLETION),
            "",
            "length",
        )


def test_incremental_prompt_is_not_certified() -> None:
    renderer, _ = _renderer()

    assert renderer.prepare_incremental_prompt(_request(), (), (), ()) is None


def test_contract_id_separates_reasoning_effort() -> None:
    low, _ = _renderer("low")
    high, _ = _renderer("high")

    assert low.render_contract_id(_request()) != high.render_contract_id(_request())


def test_stop_sequence_is_the_turn_terminator() -> None:
    renderer, _ = _renderer()

    assert renderer.stop_sequences(_request()) == ("<|eot|>",)


@pytest.mark.parametrize(
    ("renderer_name", "effort"),
    [
        ("muse_spark", "xhigh"),
        ("muse_spark_low", "low"),
        ("muse_spark_medium", "medium"),
        ("muse_spark_high", "high"),
        ("muse_spark_xhigh", "xhigh"),
    ],
)
def test_builtin_registry_resolves_each_effort(monkeypatch, renderer_name: str, effort: str) -> None:
    monkeypatch.setattr(
        tito_shared,
        "_tokenizer_fingerprint",
        lambda _tokenizer: MUSE_SPARK_TOKENIZER_FINGERPRINT,
    )

    renderer = build_sidecar_tito_renderer(_SparkTokenizer(), renderer_name)

    assert isinstance(renderer, MuseSparkTITORenderer)
    assert renderer.reasoning_effort == effort


def test_wrong_tokenizer_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        tito_shared,
        "_tokenizer_fingerprint",
        lambda _tokenizer: "f" * 64,
    )

    with pytest.raises(ValueError, match="does not match TITO certification"):
        build_sidecar_tito_renderer(_SparkTokenizer(), "muse_spark")
