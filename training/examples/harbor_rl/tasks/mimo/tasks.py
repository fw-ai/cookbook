"""Write one MiMo-V2.6-RL-oss row as a Harbor task directory.

The agent image is the published task image, unchanged. Hidden tests stay in
``tests/`` and Harbor mounts that directory at ``/tests`` only at verify time.
``docker_image`` must already be a pullable reference (see the dataset's
``image-mapping.jsonl``); this module does not talk to a registry.

Supported ``dataset_type`` values:

- ``opensource-code``: ``tests/test.sh`` resets the paths the hidden patch
  touches, applies it, and runs ``test_command``.
- ``terminal_bench``: the row ships its verifier inline (``tests_files``), and
  its ``test.sh`` already writes ``/logs/verifier/reward.txt``.
- ``arvo``: the grading server gives in-episode feedback; at verify time a
  separate verifier re-runs the collected PoC against the pristine image and
  matches the crash signature itself (``/root/last_result.json`` is ignored).
- ``webdev``: the verifier screenshots the built site and asks the configured
  vision model. Render or judge failure writes no reward.
- ``general_agent``: a Compose task with a main container and an MCP sidecar.
  Pass the task bundle directory as ``bundle_dir``.

Music is not a Harbor task. Score a completion with ``mimo.music.score_music``.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import urlsplit

_ASSETS = Path(__file__).resolve().parent / "assets"
_COMMON = ("instance_id", "docker_image", "cwd", "problem_statement")
_TYPES = {
    "opensource-code": ("mimo-code", ("test_patch", "test_command")),
    "terminal_bench": ("mimo-terminal-bench", ("tests_files",)),
    "arvo": ("mimo-cyber", ()),
    "webdev": ("mimo-webdev", ()),
    "general_agent": ("mimo-general", ()),
}


@dataclass
class _TaskSpec:
    """Per-split task.toml extras composed by ``write_harbor_task``."""

    verifier_lines: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    tail: list[str] = field(default_factory=list)


def _judge_host(judge_base_url: str | None) -> str:
    """Host the verifier's judge calls go to; must be allowlisted for them."""
    url = judge_base_url or "https://api.fireworks.ai"
    host = urlsplit(url).hostname
    if not host:
        raise ValueError(f"judge base URL has no host: {url!r}")
    return host


def unwrap_instance(row: Mapping[str, Any]) -> dict[str, Any]:
    """Accept either a flat mimoagent row or a parquet row with instance_json."""
    if row.get("instance_id") and row.get("docker_image"):
        return dict(row)
    extra = row.get("extra_info") or {}
    raw = extra.get("instance_json") if isinstance(extra, Mapping) else None
    if isinstance(raw, str):
        instance = json.loads(raw)
    elif isinstance(raw, Mapping):
        instance = dict(raw)
    else:
        raise ValueError(
            "row has no flat instance fields and no extra_info.instance_json"
        )
    if not isinstance(instance, dict):
        raise ValueError("instance_json must be an object")
    return instance


def write_harbor_task(
    instance: Mapping[str, Any],
    destination: Path,
    *,
    bundle_dir: Path | None = None,
    judge_base_url: str | None = None,
) -> Path:
    """Create ``destination/<instance_id>`` and return that path.

    ``judge_base_url`` is the endpoint the verifier's judge calls go to
    (webdev vision, general_agent rubric judge). Its host is written into the
    verifier's allowlist; default is ``https://api.fireworks.ai``, which is
    where a Fireworks serverless judge resolves anyway. Pass the same URL you
    set in ``WEBDEV_EVAL_JUDGE_BASE_URL`` / ``GA_JUDGE_URL`` at trial time.
    """
    dataset_type = instance.get("dataset_type") or "opensource-code"
    if dataset_type not in _TYPES:
        raise ValueError(
            f"{instance.get('instance_id')}: dataset_type {dataset_type!r} is not supported"
        )
    prefix, required = _TYPES[dataset_type]
    missing = [field for field in (*_COMMON, *required) if not instance.get(field)]
    if missing:
        raise ValueError(f"{instance.get('instance_id')}: missing {missing}")

    instance_id = str(instance["instance_id"])
    root = destination / instance_id
    if root.exists():
        raise FileExistsError(root)
    (root / "environment").mkdir(parents=True)
    (root / "tests").mkdir()

    (root / "instruction.md").write_text(
        str(instance["problem_statement"]).rstrip() + "\n", encoding="utf-8"
    )
    (root / "environment" / "Dockerfile").write_text(
        f"FROM {instance['docker_image']}\nWORKDIR {instance['cwd']}\n",
        encoding="utf-8",
    )
    spec = _TaskSpec()
    if dataset_type == "terminal_bench":
        _write_inline_tests(root / "tests", instance)
    elif dataset_type == "opensource-code":
        _write_patch_tests(root / "tests", instance)
    elif dataset_type == "arvo":
        spec = _write_arvo(root, instance)
    elif dataset_type == "webdev":
        spec = _write_webdev(root, instance, judge_host=_judge_host(judge_base_url))
    elif dataset_type == "general_agent":
        if bundle_dir is None:
            raise ValueError(f"{instance_id}: general_agent requires bundle_dir")
        spec = _write_general_agent(
            root, instance, Path(bundle_dir), judge_host=_judge_host(judge_base_url)
        )
    (root / "tests" / "test.sh").chmod(0o755)

    agent_timeout = float(instance.get("agent_timeout_sec") or 4800)
    verifier_timeout = float(instance.get("verifier_timeout_sec") or 1800)
    (root / "task.toml").write_text(
        "\n".join(
            [
                'schema_version = "1.1"',
                *(
                    [
                        "artifacts = ["
                        + ", ".join(json.dumps(path) for path in spec.artifacts)
                        + "]"
                    ]
                    if spec.artifacts
                    else []
                ),
                "",
                "[task]",
                f'name = "{prefix}/{instance_id}"',
                f'keywords = ["mimo", "{dataset_type}"]',
                "",
                "[metadata]",
                'source = "XiaomiMiMo/MiMo-V2.6-RL-oss"',
                f'dataset_type = "{dataset_type}"',
                f'instance_id = "{instance_id}"',
                f'health = "{instance.get("health") or "unlabeled"}"',
                "",
                "[agent]",
                f"timeout_sec = {agent_timeout}",
                "",
                "[verifier]",
                f"timeout_sec = {verifier_timeout}",
                *spec.verifier_lines,
                "",
                "[environment]",
                f"cpus = {int(instance.get('cpus') or 4)}",
                f"memory_mb = {int(instance.get('memory_mb') or 8192)}",
                f"storage_mb = {int(instance.get('storage_mb') or 20480)}",
                "gpus = 0",
                "build_timeout_sec = 1800.0",
                'network_mode = "allowlist"',
                "allowed_hosts = []",
                "",
                *spec.tail,
            ]
        ),
        encoding="utf-8",
    )
    return root


def _write_patch_tests(tests: Path, instance: Mapping[str, Any]) -> None:
    (tests / "hidden.patch").write_text(str(instance["test_patch"]), encoding="utf-8")
    (tests / "test.sh").write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                f"cd {instance['cwd']}",
                "while IFS= read -r path; do",
                '  git checkout -q HEAD -- "$path" 2>/dev/null || rm -rf -- "$path"',
                "done < <(grep -E '^diff --git a/' /tests/hidden.patch | sed -E 's|^diff --git a/([^ ]+) .*|\\1|')",
                "git apply --verbose /tests/hidden.patch",
                str(instance["test_command"]),
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_inline_tests(tests: Path, instance: Mapping[str, Any]) -> None:
    raw = instance["tests_files"]
    if isinstance(raw, Mapping):
        files = dict(raw)
    else:
        files = json.loads(str(raw))
    if "test.sh" not in files:
        raise ValueError(f"{instance['instance_id']}: tests_files has no test.sh")
    for name, encoded in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{instance['instance_id']}: unsafe test path {name!r}")
        path = tests.joinpath(*relative.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(encoded))


def _parse_arvo_description(description: str) -> dict[str, str]:
    func = re.search(r"in function `([^`]+)`", description)
    file_match = re.search(r"in file `([^`]+)`", description)
    sanitizer = re.match(r"(\S+):\s+(\S+)", description)
    if not func:
        raise ValueError(f"cannot parse function from description: {description!r}")
    return {
        "function": func.group(1),
        "file": file_match.group(1) if file_match else "",
        "sanitizer": sanitizer.group(1) if sanitizer else "",
        "error_type": sanitizer.group(2) if sanitizer else "",
        "max_submits": 0,
    }


def _write_arvo(root: Path, instance: Mapping[str, Any]) -> _TaskSpec:
    """ARVO task: agent submits PoCs to the in-sandbox server; the verifier
    re-runs the last PoC against the pristine binary in a *separate*
    environment and matches the crash signature itself.

    The expected signature also ships in the image (the grading server needs
    it for in-episode feedback, and it comes from the problem statement), but
    scoring never trusts anything the agent container wrote: ``/root/last_
    result.json`` is ignored, and the verifier replays the collected PoC
    against a fresh copy of the image so a tampered ``/root/binary`` or
    ``run.sh`` cannot fake a crash either.
    """
    description = str(instance.get("description") or instance["problem_statement"])
    expected = _parse_arvo_description(description)
    environment = root / "environment"
    shutil.copy(_ASSETS / "arvo_server.py", environment / "arvo_server.py")
    (environment / "expected_func.json").write_text(
        json.dumps(expected), encoding="utf-8"
    )
    (environment / "Dockerfile").write_text(
        "\n".join(
            [
                f"FROM {instance['docker_image']}",
                "COPY arvo_server.py /root/server.py",
                "COPY expected_func.json /root/expected_func.json",
                f"WORKDIR {instance['cwd']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    # The verifier-only copy of the expected signature; /tests is mounted only
    # at verify time, so the agent never sees the authoritative scorer input.
    (root / "tests" / "expected_func.json").write_text(
        json.dumps(expected), encoding="utf-8"
    )
    shutil.copy(_ASSETS / "arvo_match.py", root / "tests" / "arvo_match.py")
    (root / "tests" / "test.sh").write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                "mkdir -p /logs/verifier",
                "python3 /tests/arvo_match.py",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return _TaskSpec(
        verifier_lines=[
            'environment_mode = "separate"',
            'network_mode = "no-network"',
        ],
        artifacts=["/logs/artifacts/poc"],
        tail=[
            "[[verifier.collect]]",
            # Snapshot the server's last submitted PoC after the agent phase
            # ends; a missing file means no submission, i.e. reward 0.
            'command = "mkdir -p /logs/artifacts && cp /root/last_poc /logs/artifacts/poc 2>/dev/null || true"',
            "timeout_sec = 60.0",
            "",
            "[environment.healthcheck]",
            'command = "code=$(curl -s -o /dev/null -w \'%{http_code}\' http://127.0.0.1:8666/ || true); if [ \\"$code\\" != 000 ]; then exit 0; fi; setsid python3 /root/server.py >/tmp/arvo-grader.log 2>&1 </dev/null & sleep 0.5; curl -s -o /dev/null http://127.0.0.1:8666/; exit 0"',
            "interval_sec = 2.0",
            "retries = 15",
            "timeout_sec = 10.0",
        ],
    )


def _write_webdev(
    root: Path, instance: Mapping[str, Any], *, judge_host: str
) -> _TaskSpec:
    webdev = _ASSETS / "webdev"
    tests = root / "tests"
    for name in ("shot.py", "eval_rubric.py", "grade_webdev.py"):
        shutil.copy(webdev / name, tests / name)
    (tests / "query.txt").write_text(
        str(instance["problem_statement"]), encoding="utf-8"
    )
    (tests / "test.sh").write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                f"export WEBDEV_CWD={instance['cwd']}",
                "export WEBDEV_GRADE_HTTP=1",
                "cd /tests",
                "python3 /tests/grade_webdev.py",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return _TaskSpec(
        # The verifier's vision call leaves the sandbox; allowlist exactly its
        # judge host instead of relying on the trial's inference-host merge.
        verifier_lines=[
            'network_mode = "allowlist"',
            f"allowed_hosts = [{json.dumps(judge_host)}]",
        ],
        tail=[
            "[environment.env]",
            'WEBDEV_EVAL_JUDGE_MODEL = "${WEBDEV_EVAL_JUDGE_MODEL:-accounts/fireworks/models/qwen3p8-max}"',
            'WEBDEV_EVAL_JUDGE_BASE_URL = "${WEBDEV_EVAL_JUDGE_BASE_URL:-https://api.fireworks.ai/inference/v1}"',
            'WEBDEV_EVAL_JUDGE_API_KEY = "${FIREWORKS_API_KEY}"',
            'WEBDEV_GRADE_HTTP = "1"',
        ],
    )


def _write_general_agent(
    root: Path, instance: Mapping[str, Any], bundle_dir: Path, *, judge_host: str
) -> _TaskSpec:
    if not bundle_dir.is_dir():
        raise ValueError(
            f"{instance['instance_id']}: bundle_dir {bundle_dir} is not a directory"
        )
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    environment = root / "environment"
    staged = environment / "bundle"
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(bundle_dir, staged)
    (environment / "Dockerfile").write_text(
        "\n".join(
            [
                f"FROM {instance['docker_image']}",
                "COPY bundle/workspace /work/workspace",
                "COPY bundle/system /work/system",
                "COPY bundle/tools /work/tools",
                "COPY bundle/sidecar_entrypoint.py /installed-agent/sidecar_entrypoint.py",
                "COPY bundle/mcp_http.py /installed-agent/mcp_http.py",
                "COPY bundle/mcp_bridge.py /work/_setup/mcp_bridge.py",
                "COPY bundle/verify.py /work/verify.py",
                "COPY bundle/verifier_meta.json /work/verifier_meta.json",
                "COPY bundle/_helpers.py /work/_helpers.py",
                "COPY bundle/run_verify.py /work/run_verify.py",
                # The bundle's sidecar_entrypoint.py hardcodes MiMo's internal
                # /opt/openai-agents-venv/bin/python, and its mcp_http.py needs
                # mcp 1.x (FastMCP). The published OSS image instead ships mcp
                # 2.x in the system python, so satisfy the path with a real
                # compat venv (system site-packages plus a pinned mcp 1.x).
                "RUN if [ ! -x /opt/openai-agents-venv/bin/python ]; then \\",
                "      python3 -m venv --system-site-packages /opt/openai-agents-venv && \\",
                "      /opt/openai-agents-venv/bin/pip install --no-cache-dir 'mcp==1.30.0'; \\",
                "    fi",
                f"WORKDIR {instance['cwd']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (environment / "docker-compose.yaml").write_text(
        "\n".join(
            [
                "services:",
                "  sidecar:",
                "    build:",
                "      context: .",
                "    network_mode: service:main",
                "    command:",
                '      - "sh"',
                '      - "-c"',
                '      - "python3 /installed-agent/sidecar_entrypoint.py --start-and-detach && exec sleep infinity"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "tests" / "test.sh").write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -uo pipefail",
                "# The agent phase shares this mount; a prewritten reward file",
                "# must never count. Only what the verifier writes from scratch",
                "# is admissible.",
                "rm -f /logs/verifier/reward.json /logs/verifier/reward.txt",
                "python3 /work/run_verify.py || true",
                "# The bundle's verifier writes non-numeric markers (e.g.",
                "# reward_error=judge_crashed) on infra failure. Harbor expects",
                "# numeric reward fields, so sanitize: keep only numeric values,",
                "# and drop the file entirely when there is no numeric reward —",
                "# the trial is then masked, never mis-scored or error-parsed.",
                "python3 - <<'PY'",
                "import json",
                "from pathlib import Path",
                "path = Path('/logs/verifier/reward.json')",
                "if path.exists():",
                "    data = json.loads(path.read_text())",
                "    numeric = {k: v for k, v in data.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}",
                "    if 'reward' in numeric:",
                "        path.write_text(json.dumps(numeric))",
                "    else:",
                "        path.unlink()",
                "PY",
                "",
            ]
        ),
        encoding="utf-8",
    )
    mcp_lines = ["", "[[environment.mcp_servers]]"]
    for server in manifest.get("mcp_servers") or []:
        mcp_lines.extend(
            [
                f'name = "{server["name"]}"',
                'transport = "streamable-http"',
                f'url = "{server["url"]}"',
                "",
                "[[environment.mcp_servers]]",
            ]
        )
    if mcp_lines[-1] == "[[environment.mcp_servers]]":
        mcp_lines.pop()
    return _TaskSpec(
        # run_verify.py calls the rubric judge from inside the sandbox;
        # allowlist exactly the judge host for the verifier phase.
        verifier_lines=[
            'network_mode = "allowlist"',
            f"allowed_hosts = [{json.dumps(judge_host)}]",
        ],
        tail=[
            "[environment.env]",
            'GA_JUDGE_API = "${GA_JUDGE_API:-chat}"',
            'GA_JUDGE_URL = "${GA_JUDGE_URL}"',
            'GA_JUDGE_MODEL = "${GA_JUDGE_MODEL}"',
            'GA_JUDGE_KEY = "${FIREWORKS_API_KEY}"',
            "",
            "[environment.healthcheck]",
            # The MCP endpoint answers a plain GET with 406 by design, so
            # probe the LISTEN state instead of making a request.
            "command = \"python3 -c \\\"import sys; sys.exit(0 if any(line.split()[1].endswith(':%04X' % 39101) and line.split()[3] == '0A' for line in open('/proc/net/tcp').readlines()[1:]) else 1)\\\"\"",
            "interval_sec = 2.0",
            "retries = 30",
            "timeout_sec = 5.0",
            "start_period_sec = 20.0",
            *mcp_lines,
        ],
    )
