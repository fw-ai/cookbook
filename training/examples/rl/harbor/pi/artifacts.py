"""Pi-specific inspection of retained Harbor agent events."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def tool_timeout_count(trial_path: Path) -> int:
    timed_out_calls: set[str] = set()
    pending_bash_calls: set[str] = set()
    for path in sorted((trial_path / "agent").glob("*.txt")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, Mapping):
                continue
            if event.get("type") == "tool_execution_start":
                if event.get("toolName") == "bash":
                    pending_bash_calls.add(str(event.get("toolCallId") or ""))
                continue
            if event.get("type") != "tool_execution_end":
                continue
            call_id = str(event.get("toolCallId") or "")
            pending_bash_calls.discard(call_id)
            result = event.get("result")
            encoded = json.dumps(result, sort_keys=True) if result is not None else ""
            if event.get("toolName") == "bash" and "timed out after" in encoded:
                timed_out_calls.add(call_id or str(len(timed_out_calls)))

    # Pi can be killed before its terminal tool event is flushed. Attribute the
    # timeout only when the pending bash call, exit 137, and Harbor matcher agree.
    result_path = trial_path / "result.json"
    if pending_bash_calls and result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            result = {}
        exception = result.get("exception_info") or {}
        message = str(exception.get("exception_message") or "")
        trial_log_path = trial_path / "trial.log"
        try:
            trial_log = trial_log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            trial_log = ""
        if "exit 137" in message and "pattern: 'Request timed out'" in trial_log:
            timed_out_calls.update(pending_bash_calls)
    return len(timed_out_calls)


__all__ = ["tool_timeout_count"]
