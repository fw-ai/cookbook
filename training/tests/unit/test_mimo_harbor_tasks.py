from __future__ import annotations

import base64
import json
import os
import shutil
from pathlib import Path

import pytest

from training.examples.rl.harbor.datasets.mimo import unwrap_instance, write_harbor_task


def _row() -> dict:
    return {
        "instance_id": "format-code-task-000001",
        "dataset_type": "opensource-code",
        "docker_image": "example.invalid/mimo:task-000001",
        "cwd": "/testbed",
        "problem_statement": "Fix the widget.",
        "test_patch": "diff --git a/mimo_test_command.sh b/mimo_test_command.sh\n",
        "test_command": "bash /testbed/mimo_test_command.sh",
        "verifier_timeout_sec": 90,
        "health": "ok",
    }


def _terminal_bench_row(files: dict[str, bytes]) -> dict:
    return {
        "instance_id": "candidate-0036",
        "dataset_type": "terminal_bench",
        "docker_image": "example.invalid/mimo:general-agent-env-1",
        "cwd": "/app",
        "problem_statement": "Repair the pipeline.",
        "tests_files": json.dumps(
            {name: base64.b64encode(body).decode() for name, body in files.items()}
        ),
        "cpus": 1,
        "memory_mb": 2048,
        "storage_mb": 10240,
        "agent_timeout_sec": 900.0,
        "verifier_timeout_sec": 120.0,
    }


def test_write_harbor_task_keeps_hidden_tests_out_of_the_image(tmp_path: Path) -> None:
    root = write_harbor_task(_row(), tmp_path)

    dockerfile = (root / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile == "FROM example.invalid/mimo:task-000001\nWORKDIR /testbed\n"
    assert "hidden.patch" not in dockerfile
    assert (root / "instruction.md").read_text(encoding="utf-8") == "Fix the widget.\n"
    script = (root / "tests" / "test.sh").read_text(encoding="utf-8")
    assert script.startswith("#!/bin/bash\nset -euo pipefail\n")
    assert "git checkout -q HEAD" in script
    assert "git apply --verbose /tests/hidden.patch" in script
    assert script.endswith("bash /testbed/mimo_test_command.sh\n")
    toml = (root / "task.toml").read_text(encoding="utf-8")
    assert 'health = "ok"' in toml
    assert 'network_mode = "allowlist"' in toml
    assert "allowed_hosts = []" in toml


def test_terminal_bench_row_ships_its_inline_verifier(tmp_path: Path) -> None:
    files = {
        "test.sh": b"#!/bin/sh\nprintf '1\\n' > /logs/verifier/reward.txt\n",
        "fixtures/cases.json": b"[]",
    }
    root = write_harbor_task(_terminal_bench_row(files), tmp_path)

    assert (root / "tests" / "test.sh").read_bytes() == files["test.sh"]
    assert os.access(root / "tests" / "test.sh", os.X_OK)
    assert (root / "tests" / "fixtures" / "cases.json").read_bytes() == b"[]"
    toml = (root / "task.toml").read_text(encoding="utf-8")
    assert 'name = "mimo-terminal-bench/candidate-0036"' in toml
    assert "cpus = 1\nmemory_mb = 2048\nstorage_mb = 10240" in toml
    assert "[verifier]\ntimeout_sec = 120.0" in toml


def test_terminal_bench_rejects_paths_outside_tests(tmp_path: Path) -> None:
    row = _terminal_bench_row({"test.sh": b"", "../escape.sh": b""})
    with pytest.raises(ValueError, match="unsafe test path"):
        write_harbor_task(row, tmp_path)


def test_terminal_bench_accepts_a_mapping_tests_files(tmp_path: Path) -> None:
    files = {"test.sh": base64.b64encode(b"#!/bin/sh\nexit 0\n").decode()}
    row = _terminal_bench_row({"test.sh": b"#!/bin/sh\nexit 0\n"})
    row["tests_files"] = files  # already-parsed mapping, not a JSON string
    root = write_harbor_task(row, tmp_path)
    assert (root / "tests" / "test.sh").read_bytes() == b"#!/bin/sh\nexit 0\n"


def test_unwraps_parquet_instance_json() -> None:
    flat = _row()
    wrapped = {"extra_info": {"instance_json": json.dumps(flat)}}
    assert unwrap_instance(wrapped)["instance_id"] == flat["instance_id"]


def test_rejects_unsupported_datasets(tmp_path: Path) -> None:
    row = _row()
    row["dataset_type"] = "not-a-split"
    with pytest.raises(ValueError, match="is not supported"):
        write_harbor_task(row, tmp_path)


def test_arvo_task_starts_the_grader_and_scores_its_result(tmp_path: Path) -> None:
    row = _row()
    row.update(
        instance_id="arvo_35858",
        dataset_type="arvo",
        docker_image="example.invalid/mimo:arvo-35858",
        cwd="/home/agent",
        problem_statement="heap-buffer-overflow: READ in function `parse` in file `src/a.c`",
    )
    root = write_harbor_task(row, tmp_path)
    assert (root / "environment" / "arvo_server.py").is_file()
    expected = json.loads((root / "environment" / "expected_func.json").read_text())
    assert expected["function"] == "parse"
    assert expected["sanitizer"] == "heap-buffer-overflow"
    # The verifier re-runs the collected PoC itself and matches against the
    # verify-time-only expected signature; the agent-writable verdict file
    # /root/last_result.json is never read.
    assert (root / "tests" / "arvo_match.py").is_file()
    assert json.loads((root / "tests" / "expected_func.json").read_text()) == expected
    script = (root / "tests" / "test.sh").read_text()
    assert script.endswith("python3 /tests/arvo_match.py\n")
    assert "last_result.json" not in script
    toml = (root / "task.toml").read_text()
    assert "8666" in toml
    assert 'artifacts = ["/logs/artifacts/poc"]' in toml
    assert '[verifier]\ntimeout_sec = 90.0\nenvironment_mode = "separate"\nnetwork_mode = "no-network"' in toml
    assert "[[verifier.collect]]" in toml
    assert "/root/last_poc" in toml


def test_webdev_task_carries_the_eval_grader(tmp_path: Path) -> None:
    row = _row()
    row.update(
        instance_id="web-1",
        dataset_type="webdev",
        cwd="/workspace",
        problem_statement="Build a gallery.",
    )
    root = write_harbor_task(row, tmp_path)
    assert (root / "tests" / "grade_webdev.py").is_file()
    assert (root / "tests" / "shot.py").is_file()
    assert (root / "tests" / "query.txt").read_text() == "Build a gallery."
    toml = (root / "task.toml").read_text()
    assert "qwen3p8-max" in toml
    assert "FIREWORKS_API_KEY" in toml
    # The verifier's vision call is allowlisted to exactly the judge host.
    assert '[verifier]\ntimeout_sec = 90.0\nnetwork_mode = "allowlist"\nallowed_hosts = ["api.fireworks.ai"]' in toml


def test_webdev_verifier_allowlist_follows_the_judge_url(tmp_path: Path) -> None:
    row = _row()
    row.update(
        instance_id="web-2",
        dataset_type="webdev",
        cwd="/workspace",
        problem_statement="Build a gallery.",
    )
    root = write_harbor_task(
        row, tmp_path, judge_base_url="https://judge.example.test/v1"
    )
    toml = (root / "task.toml").read_text()
    assert 'allowed_hosts = ["judge.example.test"]' in toml


def test_general_agent_task_is_compose_with_mcp(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    for name in ("workspace", "system", "tools"):
        (bundle / name).mkdir(parents=True)
        (bundle / name / "keep.txt").write_text("x")
    for name in (
        "sidecar_entrypoint.py",
        "mcp_http.py",
        "mcp_bridge.py",
        "verify.py",
        "verifier_meta.json",
        "_helpers.py",
        "run_verify.py",
    ):
        (bundle / name).write_text("pass")
    (bundle / "manifest.json").write_text(
        json.dumps(
            {"mcp_servers": [{"name": "ledger", "url": "http://127.0.0.1:39101/mcp"}]}
        )
    )
    row = _row()
    row.update(instance_id="s3k-1", dataset_type="general_agent", cwd="/work/workspace")
    root = write_harbor_task(row, tmp_path / "out", bundle_dir=bundle)
    compose = (root / "environment" / "docker-compose.yaml").read_text()
    assert "network_mode: service:main" in compose
    assert "sidecar_entrypoint.py" in compose
    # The OSS image lacks MiMo's internal agents venv and ships mcp 2.x while
    # the bundle needs mcp 1.x; the Dockerfile builds a compat venv for it.
    dockerfile = (root / "environment" / "Dockerfile").read_text()
    assert "/opt/openai-agents-venv" in dockerfile
    assert "mcp==1.30.0" in dockerfile
    toml = (root / "task.toml").read_text()
    assert 'name = "ledger"' in toml
    assert "GA_JUDGE_API" in toml
    # The rubric judge call from run_verify.py is allowlisted to its host.
    assert 'allowed_hosts = ["api.fireworks.ai"]' in toml
    script = (root / "tests" / "test.sh").read_text()
    assert "python3 /work/run_verify.py" in script
    # Non-numeric verifier markers (judge crash) must mask, not crash parsing.
    assert "path.unlink()" in script
    # A reward file prewritten during the agent phase must never count.
    assert "rm -f /logs/verifier/reward.json /logs/verifier/reward.txt" in script


def test_music_without_abc2midi_is_not_a_zero() -> None:
    from training.examples.rl.harbor.datasets.mimo import Abc2MidiMissing, score_music

    if shutil.which("abc2midi"):
        assert score_music("not abc at all") == 0.0
        return
    with pytest.raises(Abc2MidiMissing):
        score_music("X:1\nT:Hi\nK:C\nCDEF|\n")
