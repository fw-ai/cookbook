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


def parse_first_frozen_lake_tool_call(output_text: str) -> ParsedToolCall:
    """Parse the first tool call from strict tool-call JSON output."""
    loaders = (
        _load_json_object_with_text_span,
        _load_xml_tool_call_payload_with_text_span,
        _load_kimi_native_tool_call_payload_with_text_span,
    )
    last_error: Exception | None = None
    for loader in loaders:
        try:
            payload, _, _ = loader(output_text)
            return _parse_tool_call_from_payload(payload)
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise ValueError("Failed to parse tool call payload")


def parse_first_frozen_lake_tool_call_with_content(output_text: str) -> Tuple[ParsedToolCall, str]:
    """Parse the first tool call and return residual assistant text content."""
    loaders = (
        _load_json_object_with_text_span,
        _load_xml_tool_call_payload_with_text_span,
        _load_kimi_native_tool_call_payload_with_text_span,
    )
    last_error: Exception | None = None
    for loader in loaders:
        try:
            payload, json_start, json_end = loader(output_text)
            parsed_tool_call = _parse_tool_call_from_payload(payload)
            content = (output_text[:json_start] + output_text[json_end:]).strip()
            return parsed_tool_call, content
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise ValueError("Failed to parse tool call payload")


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
