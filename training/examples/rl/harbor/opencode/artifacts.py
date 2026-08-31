"""OpenCode-specific inspection of retained Harbor agent events."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def tool_timeout_count(trial_path: Path) -> int:
    timed_out_calls: set[str] = set()
    for path in sorted((trial_path / "agent").glob("*.txt")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, Mapping) or event.get("type") != "tool_use":
                continue
            part = event.get("part")
            if not isinstance(part, Mapping) or part.get("tool") != "bash":
                continue
            state = part.get("state")
            if not isinstance(state, Mapping):
                continue
            encoded = json.dumps(state, sort_keys=True)
            if any(
                marker in encoded
                for marker in ("timed out", "exceeded timeout", "exceeding timeout")
            ):
                timed_out_calls.add(str(part.get("callID") or len(timed_out_calls)))
    return len(timed_out_calls)


__all__ = ["tool_timeout_count"]
