"""Split a sampled completion into reasoning and answer with the task's renderer."""

from __future__ import annotations

from typing import Any


def split_reply(renderer: Any, tokens: list[int]) -> tuple[str, str]:
    """``(reasoning, answer)`` from completion tokens."""
    message, _ = renderer.parse_response(tokens)
    content = message["content"]
    if isinstance(content, str):
        if "</think>" in content:
            reasoning, answer = content.split("</think>", 1)
            return reasoning.replace("<think>", "").strip(), answer.strip()
        return "", content.strip()
    reasoning = "\n".join(p["thinking"] for p in content if p["type"] == "thinking")
    answer = "".join(p.get("text", "") for p in content if p["type"] == "text")
    return reasoning.strip(), answer.strip()
