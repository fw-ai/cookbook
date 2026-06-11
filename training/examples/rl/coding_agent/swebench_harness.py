"""SWE-Bench/SWE-Gym grading for the coding-agent sandbox.

Adapted from NVIDIA-NeMo/ProRL-Agent-Server
``polar/trajectory/evaluator/swebench_harness.py`` and its patch helpers.
The Polar version is runtime-framework generic; this file keeps only the
pieces needed by the cookbook ``Sandbox`` protocol.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import shlex
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from training.examples.rl.coding_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

APPLY_PATCH_PASS = "__CODING_AGENT_APPLY_PATCH_PASS__"
APPLY_PATCH_FAIL = "__CODING_AGENT_APPLY_PATCH_FAIL__"
DEFAULT_EXCLUDE_PATTERNS = (
    "__pycache__/**",
    "**/__pycache__/**",
    "*.pyc",
    "**/*.pyc",
    "*.pyo",
    "**/*.pyo",
    ".pytest_cache/**",
    "**/.pytest_cache/**",
    "trajectories/**",
    "**/trajectories/**",
    "node_modules/**",
    "**/node_modules/**",
    ".cache/**",
    "**/.cache/**",
    ".venv/**",
    "**/.venv/**",
    ".claude/**",
    "**/.claude/**",
    ".codex/**",
    "**/.codex/**",
    ".gemini/**",
    "**/.gemini/**",
    ".opencode/**",
    "**/.opencode/**",
    ".config/opencode/**",
    ".pi/**",
    "**/.pi/**",
    ".qwen/**",
    "**/.qwen/**",
    "$HOME/**",
    "**/$HOME/**",
)
APPLY_PATCH_TIMEOUT_SEC = 120
APPLY_OUTPUT_TAIL_CHARS = 8000
PATCH_HEAD_CHARS = 4000


def normalize_harness_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Normalize an instance for SWE-Gym/SWE-bench harness calls."""
    normalized = dict(instance)
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = normalized.get(key)
        if isinstance(value, str):
            try:
                normalized[key] = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                pass
    instance_id = str(normalized["instance_id"]).lower()
    normalized["instance_id"] = instance_id
    if "version" not in normalized and "base_commit" in normalized:
        normalized["version"] = normalized["base_commit"]
    return normalized


def filter_patch(
    patch: str,
    *,
    exclude_patterns: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Drop noisy generated files from a unified git diff."""
    if not patch.strip():
        return patch
    patterns = tuple(dict.fromkeys([*DEFAULT_EXCLUDE_PATTERNS, *(exclude_patterns or [])]))
    kept_sections: list[str] = []
    current_lines: list[str] = []
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current_lines and not _exclude_patch_section(current_lines, patterns):
                kept_sections.extend(current_lines)
            current_lines = [line]
        elif current_lines:
            current_lines.append(line)
        else:
            kept_sections.append(line)
    if current_lines and not _exclude_patch_section(current_lines, patterns):
        kept_sections.extend(current_lines)
    return "".join(kept_sections)


async def evaluate_swebench_patch(
    *,
    sandbox: "Sandbox",
    workdir: str,
    diff_text: str,
    instance: dict[str, Any],
    timeout_sec: int,
    exclude_patterns: list[str] | None = None,
) -> tuple[float, bool, bool, dict[str, Any]]:
    """Apply ``diff_text`` in ``sandbox`` and grade with SWE-Gym/SWE-bench."""
    filtered_diff = filter_patch(diff_text, exclude_patterns=exclude_patterns)
    if not filtered_diff.strip():
        return 0.0, False, True, {"empty_generation": True, "resolved": False}

    normalized = normalize_harness_instance(instance)
    test_spec, get_eval_report = _load_harness(normalized)
    eval_script = str(test_spec.eval_script)

    applied, apply_output = await _apply_patch(sandbox, workdir, filtered_diff)
    if not applied:
        return (
            0.0,
            False,
            False,
            {
                "empty_generation": False,
                "resolved": False,
                "failed_apply_patch": True,
                "apply_patch_output": apply_output[-APPLY_OUTPUT_TAIL_CHARS:],
                "patch_head": filtered_diff[:PATCH_HEAD_CHARS],
            },
        )

    await sandbox.write_file("/tmp/cagent_eval.sh", eval_script, user="agent")
    ec, stdout, stderr = await sandbox.exec(
        f"cd {shlex.quote(workdir)} && /bin/bash /tmp/cagent_eval.sh",
        user="agent",
        timeout=timeout_sec,
        check=False,
    )
    combined = (stdout or "") + (stderr or "")
    report = _grade_harness_run(
        get_eval_report,
        test_spec=test_spec,
        prediction={"model_patch": filtered_diff, "instance_id": normalized["instance_id"]},
        test_output=combined,
    )
    instance_report = report[normalized["instance_id"]]
    resolved = bool(instance_report.get("resolved", False))
    metadata = {
        "empty_generation": False,
        "resolved": resolved,
        "failed_apply_patch": False,
        "error_eval": False,
        "test_timeout": ec == -1,
        "exit_code": ec,
        "grading_report": instance_report,
    }
    return (1.0 if resolved else 0.0), resolved, True, metadata


async def _apply_patch(sandbox: "Sandbox", workdir: str, patch: str) -> tuple[bool, str]:
    await sandbox.write_file("/tmp/cagent_swebench_patch.diff", patch, user="agent")
    patch_path = shlex.quote("/tmp/cagent_swebench_patch.diff")
    quoted_workdir = shlex.quote(workdir)
    command = (
        f"cd {quoted_workdir} && "
        f"(git apply -v {patch_path} && echo '{APPLY_PATCH_PASS}' || "
        f"(echo 'Failed to apply patch with git apply, trying with patch command...' && "
        f"(patch --batch --fuzz=5 -p1 -i {patch_path} && "
        f"echo '{APPLY_PATCH_PASS}' || echo '{APPLY_PATCH_FAIL}')))"
    )
    ec, stdout, stderr = await sandbox.exec(
        command,
        user="agent",
        timeout=APPLY_PATCH_TIMEOUT_SEC,
        check=False,
    )
    output = (stdout or "") + (stderr or "")
    return ec == 0 and APPLY_PATCH_PASS in output and APPLY_PATCH_FAIL not in output, output


def _load_harness(instance: dict[str, Any]) -> tuple[Any, Any]:
    try:
        from swegym.harness.grading import get_eval_report
        from swegym.harness.test_spec import make_test_spec
    except ModuleNotFoundError:
        from swebench.harness.grading import get_eval_report
        try:
            from swebench.harness.test_spec.test_spec import make_test_spec
        except ModuleNotFoundError:
            from swebench.harness.test_spec import make_test_spec
    return make_test_spec(instance), get_eval_report


def _grade_harness_run(
    get_eval_report: Any,
    *,
    test_spec: Any,
    prediction: dict[str, Any],
    test_output: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cagent-swebench-") as tmp:
        instance_id = str(prediction["instance_id"])
        log_dir = Path(tmp) / instance_id
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "test_output.txt"
        log_path.write_text(test_output)
        try:
            return get_eval_report(
                test_spec=test_spec,
                prediction=prediction,
                log_path=str(log_path),
                include_tests_status=True,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            return get_eval_report(
                test_spec=test_spec,
                prediction=prediction,
                test_log_path=str(log_path),
                include_tests_status=True,
            )


def _exclude_patch_section(lines: list[str], patterns: tuple[str, ...]) -> bool:
    for line in lines:
        if line.startswith("diff --git "):
            parts = line.strip().split()
            if len(parts) >= 4:
                path = parts[3][2:] if parts[3].startswith("b/") else parts[3]
                return _matches_exclude(path, patterns)
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path != "/dev/null":
                normalized = path[2:] if path.startswith("b/") else path
                return _matches_exclude(normalized, patterns)
    return False


def _matches_exclude(path: str, patterns: tuple[str, ...]) -> bool:
    normalized = path.strip()
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in patterns)
