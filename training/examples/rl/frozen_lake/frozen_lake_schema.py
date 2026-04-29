"""Tool schema and parsing helpers for FrozenLake tool-call rollouts."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Tuple

from eval_protocol.integrations.fireworks_v1_completions_client import ParsedToolCall

TOOL_NAME_LAKE_MOVE = "lake_move"
FROZEN_LAKE_ACTIONS: Tuple[str, ...] = ("LEFT", "DOWN", "RIGHT", "UP")


FROZEN_LAKE_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME_LAKE_MOVE,
            "description": "Move in FrozenLake by one step using action LEFT, DOWN, RIGHT, or UP.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": list(FROZEN_LAKE_ACTIONS),
                        "description": "Movement direction in the grid world.",
                    }
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        },
    }
]


def _load_json_object_with_text_span(text: str) -> Tuple[Dict[str, Any], int, int]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty model output; expected JSON tool call payload")

    decoder = json.JSONDecoder()
    leading_ws = len(text) - len(text.lstrip())
    trailing_non_ws = len(text.rstrip())
    try:
        json.loads(stripped)
        start = leading_ws
        end = trailing_non_ws
    except json.JSONDecodeError:
        start_in_stripped = stripped.find("{")
        if start_in_stripped < 0:
            raise ValueError(f"Model output is not valid JSON: {text!r}") from None
        try:
            _, consumed = decoder.raw_decode(stripped[start_in_stripped:])
            start = leading_ws + start_in_stripped
            end = start + consumed
        except json.JSONDecodeError:
            raise ValueError(f"Model output is not valid JSON: {text!r}") from None

    value = json.loads(text[start:end])
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object for tool call payload, got: {type(value).__name__}")
    return value, start, end


def _parse_tool_call_from_payload(payload: Dict[str, Any]) -> ParsedToolCall:
    tool_calls = payload.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        raise ValueError("Model output must include non-empty 'tool_calls'")

    first = tool_calls[0]
    if not isinstance(first, dict):
        raise ValueError("Each tool call must be an object")

    name = str(first.get("name") or "").strip()
    if name != TOOL_NAME_LAKE_MOVE:
        raise ValueError(f"Unsupported tool '{name}', expected '{TOOL_NAME_LAKE_MOVE}'")

    arguments = first.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("Tool call 'arguments' must be an object")

    action = str(arguments.get("action") or "").strip().upper()
    if action not in FROZEN_LAKE_ACTIONS:
        raise ValueError(f"Invalid action '{action}'. Expected one of {FROZEN_LAKE_ACTIONS}")

    normalized_args = {"action": action}
    tool_call_id = str(first.get("id") or f"toolcall_{uuid.uuid4().hex[:12]}")
    return ParsedToolCall(
        tool_call_id=tool_call_id,
        name=TOOL_NAME_LAKE_MOVE,
        arguments=normalized_args,
    )


def _load_xml_tool_call_payload_with_text_span(text: str) -> Tuple[Dict[str, Any], int, int]:
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No XML tool_call block found")
    payload_text = match.group(1)
    try:
        raw_tool_call = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON inside <tool_call>: {payload_text!r}") from exc
    if not isinstance(raw_tool_call, dict):
        raise ValueError("XML <tool_call> payload must be a JSON object")
    payload = {"tool_calls": [raw_tool_call]}
    start, end = match.span()
    return payload, start, end


# Anthropic / Qwen 3.5 style: ``<tool_call>`` wraps an inner
# ``<function=NAME>...</function>`` element whose children are
# ``<parameter=KEY>VALUE</parameter>`` blocks. The cookbook's
# ``_load_xml_tool_call_payload_with_text_span`` (above) only matches the
# ``<tool_call>{...JSON...}</tool_call>`` variant, so we need a dedicated
# loader for the ``<function=...>`` shape.
#
# Example payload that this loader accepts:
#
#   <tool_call>
#   <function=lake_move>
#   <parameter=action>RIGHT</parameter>
#   </function>
#   </tool_call>
#
# The Qwen 3.5 chat template advertises this exact shape in the system
# prompt (see ``apply_chat_template(..., tools=...)`` output), and the
# model emits matching tool calls during rollouts.
_ANTHROPIC_XML_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*"
    r"<function=(?P<name>[^>\s]+)>(?P<body>.*?)</function>\s*"
    r"</tool_call>",
    flags=re.DOTALL,
)
_ANTHROPIC_XML_PARAMETER_RE = re.compile(
    r"<parameter=(?P<key>[^>\s]+)>(?P<value>.*?)</parameter>",
    flags=re.DOTALL,
)


def _coerce_anthropic_parameter_value(raw: str) -> Any:
    """Decode an Anthropic-XML <parameter=...>...</parameter> body.

    The body is plain text — the model writes the value verbatim between
    the opening and closing tags. We try ``json.loads`` first so primitives
    that look like JSON (numbers, booleans, ``null``, JSON strings/objects/
    arrays) round-trip with the right type, and fall back to the stripped
    raw string for everything else.
    """
    stripped = raw.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return stripped


def _load_anthropic_xml_function_call_payload_with_text_span(
    text: str,
) -> Tuple[Dict[str, Any], int, int]:
    match = _ANTHROPIC_XML_TOOL_CALL_RE.search(text)
    if not match:
        raise ValueError("No Anthropic-style <function=...> tool_call block found")

    name = (match.group("name") or "").strip()
    if not name:
        raise ValueError("<function=...> tag is missing a function name")
    body = match.group("body") or ""

    arguments: Dict[str, Any] = {}
    for param in _ANTHROPIC_XML_PARAMETER_RE.finditer(body):
        key = (param.group("key") or "").strip()
        if not key:
            continue
        arguments[key] = _coerce_anthropic_parameter_value(param.group("value") or "")

    payload = {
        "tool_calls": [
            {
                "id": f"toolcall_{uuid.uuid4().hex[:12]}",
                "name": name,
                "arguments": arguments,
            }
        ]
    }
    start, end = match.span()
    return payload, start, end


def _load_kimi_native_tool_call_payload_with_text_span(text: str) -> Tuple[Dict[str, Any], int, int]:
    match = re.search(
        r"(?:<\|tool_calls_section_begin\|>)?\s*"
        r"<\|tool_call_begin\|>\s*(.*?)\s*"
        r"<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*"
        r"<\|tool_call_end\|>\s*"
        r"(?:<\|tool_calls_section_end\|>)?",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("No Kimi native tool_call block found")

    raw_identifier = str(match.group(1) or "").strip()
    arguments_text = match.group(2)
    try:
        arguments = json.loads(arguments_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON inside Kimi tool call: {arguments_text!r}") from exc
    if not isinstance(arguments, dict):
        raise ValueError("Kimi tool call arguments must be a JSON object")

    tool_name = TOOL_NAME_LAKE_MOVE
    identifier = raw_identifier
    if raw_identifier.startswith("functions."):
        qualified_name = raw_identifier.split(":", 1)[0]
        tool_name = qualified_name.split(".", 1)[1] or TOOL_NAME_LAKE_MOVE
    payload = {
        "tool_calls": [
            {
                "id": identifier or f"toolcall_{uuid.uuid4().hex[:12]}",
                "name": tool_name,
                "arguments": arguments,
            }
        ]
    }
    start, end = match.span()
    return payload, start, end


# Order matters: each loader is tried in turn and the first one that
# matches wins. Anthropic-XML must come BEFORE
# `_load_xml_tool_call_payload_with_text_span` because both inspect
# `<tool_call>...</tool_call>` blocks, and the JSON-body variant would
# otherwise throw a less-actionable "Invalid JSON inside <tool_call>" on
# Anthropic-style payloads.
_TOOL_CALL_LOADERS = (
    _load_json_object_with_text_span,
    _load_anthropic_xml_function_call_payload_with_text_span,
    _load_xml_tool_call_payload_with_text_span,
    _load_kimi_native_tool_call_payload_with_text_span,
)


class FrozenLakeToolCallParseError(ValueError):
    """Raised when none of the tool-call loaders match the model output.

    Carries the per-loader errors so callers can log a single, structured
    summary instead of the misleading last-error-in-the-chain message
    (e.g. ``"No Kimi native tool_call block found"``) that the previous
    loop would surface verbatim — confusing operators about which parser
    actually produced the failure.
    """

    def __init__(self, errors: Tuple[Tuple[str, str], ...], output_text: str):
        self.errors = errors
        self.output_text = output_text
        names = ", ".join(name for name, _ in errors)
        sample = output_text.strip().splitlines()[:1]
        sample_str = sample[0][:120] if sample else ""
        super().__init__(
            f"No tool-call loader matched the model output (tried: {names}). "
            f"First line of output: {sample_str!r}"
        )


def parse_first_frozen_lake_tool_call(output_text: str) -> ParsedToolCall:
    """Parse the first tool call from strict tool-call JSON output."""
    errors: List[Tuple[str, str]] = []
    for loader in _TOOL_CALL_LOADERS:
        try:
            payload, _, _ = loader(output_text)
            # _parse_tool_call_from_payload itself raises on payloads that
            # don't carry a non-empty `tool_calls` array. Keep that call
            # inside the try so a stray JSON object (e.g. raw `{"action":
            # "DOWN"}`) doesn't short-circuit the chain — subsequent
            # loaders may still recognize the actual tool-call shape.
            return _parse_tool_call_from_payload(payload)
        except Exception as exc:  # noqa: BLE001 — chain captures all loader errors
            errors.append((loader.__name__, str(exc)))
            continue
    raise FrozenLakeToolCallParseError(tuple(errors), output_text)


def parse_first_frozen_lake_tool_call_with_content(output_text: str) -> Tuple[ParsedToolCall, str]:
    """Parse the first tool call and return residual assistant text content."""
    errors: List[Tuple[str, str]] = []
    for loader in _TOOL_CALL_LOADERS:
        try:
            payload, json_start, json_end = loader(output_text)
            parsed_tool_call = _parse_tool_call_from_payload(payload)
            content = (output_text[:json_start] + output_text[json_end:]).strip()
            return parsed_tool_call, content
        except Exception as exc:  # noqa: BLE001
            errors.append((loader.__name__, str(exc)))
            continue
    raise FrozenLakeToolCallParseError(tuple(errors), output_text)


def normalize_parsed_tool_call(
    *, tool_call_id: str | None, name: Any, arguments: Any,
) -> ParsedToolCall:
    """Validate and normalize a parsed tool call for FrozenLake."""
    if str(name or "").strip() != TOOL_NAME_LAKE_MOVE:
        raise ValueError(f"Unsupported tool '{name}', expected '{TOOL_NAME_LAKE_MOVE}'")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("Tool call arguments must be a JSON object")

    action = str(arguments.get("action") or "").strip().upper()
    if action not in FROZEN_LAKE_ACTIONS:
        raise ValueError(f"Invalid action '{action}', expected one of {FROZEN_LAKE_ACTIONS}")

    normalized_id = str(tool_call_id or f"toolcall_{uuid.uuid4().hex[:12]}")
    return ParsedToolCall(
        tool_call_id=normalized_id,
        name=TOOL_NAME_LAKE_MOVE,
        arguments={"action": action},
    )


def parse_tool_call_with_fallback(
    completion_text: str, *, allow_plaintext: bool = False,
) -> ParsedToolCall:
    """Try structured parsing, optionally fall back to regex action matching."""
    try:
        return parse_first_frozen_lake_tool_call(completion_text)
    except Exception:
        if not allow_plaintext:
            raise

    action: str | None = None
    word_match = re.search(r"\b(LEFT|DOWN|RIGHT|UP)\b", completion_text, flags=re.IGNORECASE)
    if word_match:
        action = word_match.group(1).upper()

    if action is None:
        letter_match = re.search(
            r"(?:\"action\"|action)\s*[:=]\s*\"?\s*([LRUD])\s*\"?",
            completion_text,
            flags=re.IGNORECASE,
        ) or re.search(r"\b([LRUD])\b", completion_text, flags=re.IGNORECASE)
        if letter_match:
            alias = letter_match.group(1).upper()
            alias_map = {"L": "LEFT", "D": "DOWN", "R": "RIGHT", "U": "UP"}
            action = alias_map.get(alias)

    if action is None:
        raise ValueError(f"Model output is not valid JSON and no action fallback found: {completion_text!r}")

    return ParsedToolCall(
        tool_call_id=f"toolcall_{uuid.uuid4().hex[:12]}",
        name=TOOL_NAME_LAKE_MOVE,
        arguments={"action": action},
    )
