"""Unit tests for ``frozen_lake_schema`` tool-call loaders."""

from __future__ import annotations

import pytest

from training.examples.rl.frozen_lake.frozen_lake_schema import (
    FrozenLakeToolCallParseError,
    _load_anthropic_xml_function_call_payload_with_text_span,
    _load_kimi_native_tool_call_payload_with_text_span,
    _load_xml_tool_call_payload_with_text_span,
    parse_first_frozen_lake_tool_call,
    parse_first_frozen_lake_tool_call_with_content,
)


# ---------------------------------------------------------------------------
# Anthropic-XML loader
# ---------------------------------------------------------------------------


ANTHROPIC_XML_BASIC = (
    "<tool_call>\n"
    "<function=lake_move>\n"
    "<parameter=action>RIGHT</parameter>\n"
    "</function>\n"
    "</tool_call>"
)

ANTHROPIC_XML_WITH_PROSE = (
    "I will move RIGHT to make progress toward the goal.\n\n"
    + ANTHROPIC_XML_BASIC
    + "\n\nThis avoids the hole at (1,1)."
)


def test_anthropic_xml_loader_basic():
    """Single-parameter Anthropic-style payload parses to the expected dict."""
    payload, start, end = _load_anthropic_xml_function_call_payload_with_text_span(
        ANTHROPIC_XML_BASIC,
    )
    assert payload["tool_calls"][0]["name"] == "lake_move"
    assert payload["tool_calls"][0]["arguments"] == {"action": "RIGHT"}
    # The returned span must cover exactly the <tool_call>...</tool_call> block.
    assert ANTHROPIC_XML_BASIC[start:end].startswith("<tool_call>")
    assert ANTHROPIC_XML_BASIC[start:end].endswith("</tool_call>")


def test_anthropic_xml_loader_strips_residual_prose():
    """End-to-end parse keeps prose around the tool call as residual content."""
    parsed, residual = parse_first_frozen_lake_tool_call_with_content(
        ANTHROPIC_XML_WITH_PROSE,
    )
    assert parsed.name == "lake_move"
    assert parsed.arguments == {"action": "RIGHT"}
    assert residual.startswith("I will move RIGHT")
    assert residual.endswith("This avoids the hole at (1,1).")
    assert "<tool_call>" not in residual


def test_anthropic_xml_loader_decodes_json_parameter_values():
    """Parameter values that look like JSON round-trip with the right type.

    The model occasionally wraps an action in quotes (``"RIGHT"``) or
    nests nested-object args (``{"key": "value"}``); the loader should
    decode them via ``json.loads`` rather than treating them as opaque
    strings.
    """
    payload, _, _ = _load_anthropic_xml_function_call_payload_with_text_span(
        "<tool_call>\n"
        "<function=lake_move>\n"
        '<parameter=action>"DOWN"</parameter>\n'
        "<parameter=count>3</parameter>\n"
        "<parameter=enabled>true</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    args = payload["tool_calls"][0]["arguments"]
    assert args == {"action": "DOWN", "count": 3, "enabled": True}


def test_anthropic_xml_loader_ignores_non_matching_text():
    """Payloads without a `<tool_call><function=...>` block raise."""
    with pytest.raises(ValueError, match="No Anthropic-style"):
        _load_anthropic_xml_function_call_payload_with_text_span(
            "<tool_call>{\"tool_calls\": [{\"name\": \"lake_move\"}]}</tool_call>",
        )


# ---------------------------------------------------------------------------
# Loader chain priority
# ---------------------------------------------------------------------------


def test_anthropic_xml_takes_priority_over_xml_json_loader():
    """The chain tries Anthropic-XML before the JSON-in-tool_call loader.

    Both loaders inspect ``<tool_call>...</tool_call>``; if Anthropic-XML
    were ordered after the JSON one, the JSON loader would error out
    with ``Invalid JSON inside <tool_call>: '<function=...'`` and the
    Anthropic payload would never be tried.
    """
    parsed = parse_first_frozen_lake_tool_call(ANTHROPIC_XML_BASIC)
    assert parsed.name == "lake_move"
    assert parsed.arguments == {"action": "RIGHT"}


def test_existing_json_in_xml_loader_still_works():
    """The JSON-inside-`<tool_call>` variant must still parse after we add
    the Anthropic loader (regression guard against accidentally swallowing
    the JSON shape with the new regex).
    """
    payload, _, _ = _load_xml_tool_call_payload_with_text_span(
        '<tool_call>{"name": "lake_move", "arguments": {"action": "UP"}}</tool_call>'
    )
    assert payload["tool_calls"][0]["name"] == "lake_move"
    assert payload["tool_calls"][0]["arguments"] == {"action": "UP"}


def test_existing_kimi_loader_still_works():
    """Smoke-check the Kimi-native loader hasn't regressed."""
    payload, _, _ = _load_kimi_native_tool_call_payload_with_text_span(
        "<|tool_call_begin|>functions.lake_move:0<|tool_call_argument_begin|>"
        '{"action": "LEFT"}<|tool_call_end|>'
    )
    assert payload["tool_calls"][0]["name"] == "lake_move"
    assert payload["tool_calls"][0]["arguments"] == {"action": "LEFT"}


# ---------------------------------------------------------------------------
# Structured failure
# ---------------------------------------------------------------------------


def test_parser_raises_structured_error_when_no_loader_matches():
    """When every loader fails, the chain raises ``FrozenLakeToolCallParseError``
    with per-loader errors instead of bubbling up the misleading last-error
    message (e.g. ``"No Kimi native tool_call block found"``) that the
    previous implementation surfaced verbatim — that wording confused
    operators about which parser actually rejected the model output.
    """
    with pytest.raises(FrozenLakeToolCallParseError) as excinfo:
        parse_first_frozen_lake_tool_call("This response has no tool call at all.")
    err = excinfo.value
    loader_names = {name for name, _ in err.errors}
    assert loader_names == {
        "_load_json_object_with_text_span",
        "_load_anthropic_xml_function_call_payload_with_text_span",
        "_load_xml_tool_call_payload_with_text_span",
        "_load_kimi_native_tool_call_payload_with_text_span",
    }
    assert "No tool-call loader matched" in str(err)
