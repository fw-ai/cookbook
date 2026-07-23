"""Private metadata for renderer-specific reasoning-field precedence.

``normalize_messages`` converts API-shaped top-level reasoning fields into
Tinker's structured ``thinking`` content.  Most model families use Jinja
truthiness when both ``reasoning`` and ``reasoning_content`` are present, but
Kimi uses field presence.  Keep the original string values on private keys so
the Kimi adapters can apply their vendor-specific rule without changing the
generic normalized representation seen by Qwen, Gemma, and legacy renderers.
"""

from __future__ import annotations

from typing import Any, Mapping


ORIGINAL_REASONING = "_fireworks_original_reasoning"
ORIGINAL_REASONING_CONTENT = "_fireworks_original_reasoning_content"


def original_reasoning(message: Mapping[str, Any]) -> tuple[bool, str]:
    """Return whether a normalized/raw ``reasoning`` string was supplied."""

    if ORIGINAL_REASONING in message:
        value = message[ORIGINAL_REASONING]
        return isinstance(value, str), value if isinstance(value, str) else ""
    if "reasoning" in message:
        value = message["reasoning"]
        return isinstance(value, str), value if isinstance(value, str) else ""
    return False, ""


def original_reasoning_content(message: Mapping[str, Any]) -> tuple[bool, str]:
    """Return whether a normalized/raw ``reasoning_content`` was supplied."""

    if ORIGINAL_REASONING_CONTENT in message:
        value = message[ORIGINAL_REASONING_CONTENT]
        return isinstance(value, str), value if isinstance(value, str) else ""
    if "reasoning_content" in message:
        value = message["reasoning_content"]
        return isinstance(value, str), value if isinstance(value, str) else ""
    return False, ""
