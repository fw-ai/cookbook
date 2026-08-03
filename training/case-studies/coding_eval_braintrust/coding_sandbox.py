"""Small helpers for function-completion coding evals (HumanEval-style)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from typing import Any


_FENCE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python(text: str) -> str:
    """Pull Python from a model response; strip markdown fences when present."""
    text = text or ""
    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1)
    # Drop leading/trailing blank lines but preserve body indentation.
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def build_program(prompt: str, completion: str) -> str:
    """Merge a HumanEval-style prompt prefix with the model completion."""
    completion = extract_python(completion)
    if completion.startswith(prompt):
        return completion
    return prompt + completion


def run_python_tests(program: str, test_code: str, *, timeout: float = 5.0) -> tuple[bool, str | None]:
    """Execute `program` plus `test_code` in a subprocess. Returns (passed, error)."""
    script = f"{program.rstrip()}\n\n{test_code.strip()}\n"
    fd, path = tempfile.mkstemp(suffix=".py", text=True)
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0:
            return True, None
        err = (proc.stderr or proc.stdout or "").strip()
        return False, err[:2000] if err else "non-zero exit"
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def passes_tests(
    output: str,
    expected: dict[str, Any],
    *,
    input: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Braintrust scorer: run bundled unit tests against generated code."""
    row = input or kwargs.get("input") or {}
    prompt = row.get("prompt", "")
    program = build_program(prompt, output)
    ok, err = run_python_tests(program, expected["test"])
    return {"name": "passes_tests", "score": 1.0 if ok else 0.0, "metadata": {"error": err}}
