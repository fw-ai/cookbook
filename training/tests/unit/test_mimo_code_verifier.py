from __future__ import annotations

import subprocess
import json
import tomllib
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from training.examples.harbor_rl.tasks.mimo.assets import code_collect, code_verify
from training.examples.harbor_rl.tasks.mimo.assets.code_verify import verify
from training.examples.harbor_rl.tasks.mimo.assets.code_collect import collect
from training.examples.harbor_rl.tasks.mimo import write_harbor_task


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture
def task(tmp_path: Path) -> tuple[Path, dict, Path]:
    root = tmp_path / "workspace"
    root.mkdir()
    _git(root, "init")
    (root / "widget.py").write_text("value = 0\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "clean base")
    base = _git(root, "rev-parse", "HEAD").strip()
    (root / "hidden_check.py").write_text(
        "from widget import value\nassert value == 1\n"
    )
    _git(root, "add", "-N", "hidden_check.py")
    hidden = tmp_path / "hidden.patch"
    hidden.write_text(_git(root, "diff", "--binary", base))
    _git(root, "reset", "--hard", base)
    _git(root, "clean", "-fd")
    return (
        root,
        {
            "cwd": str(root),
            "base_commit": base,
            "test_command": "python3 hidden_check.py",
        },
        hidden,
    )


def _candidate(root: Path, tmp_path: Path, base: str) -> Path:
    _git(root, "add", "-A")
    patch = tmp_path / "candidate.patch"
    patch.write_text(_git(root, "diff", "--cached", "--binary", "--no-renames", base))
    _git(root, "reset", "--hard", base)
    _git(root, "clean", "-fd")
    return patch


def test_fixed_source_passes_in_fresh_workspace(task: tuple, tmp_path: Path) -> None:
    root, config, hidden = task
    (root / "widget.py").write_text("value = 1\n")
    candidate = _candidate(root, tmp_path, config["base_commit"])
    assert verify(config, candidate, hidden)


def test_trusted_test_command_preserves_image_git_identity(
    task: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config, hidden = task
    (root / "widget.py").write_text("value = 1\n")
    candidate = _candidate(root, tmp_path, config["base_commit"])
    image_config = tmp_path / "image.gitconfig"
    image_config.write_text("[user]\nname = Image User\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(image_config))
    config[
        "test_command"
    ] += ' && test "$(git config --global user.name)" = "Image User"'
    assert verify(config, candidate, hidden)


def test_generated_task_executes_its_bundled_capture_command(
    task: tuple, tmp_path: Path
) -> None:
    root, config, hidden = task
    task_dir = write_harbor_task(
        {
            "instance_id": "capture-command",
            "dataset_type": "opensource-code",
            "docker_image": "example.invalid/clean-task",
            "cwd": str(root),
            "problem_statement": "Fix the widget.",
            "clean_base_commit": config["base_commit"],
            "test_patch": hidden.read_text(),
            "test_command": config["test_command"],
        },
        tmp_path / "tasks",
        fresh_verifier=True,
    )
    (root / "widget.py").write_text("value = 1\n")
    candidate = tmp_path / "generated.patch"
    document = tomllib.loads((task_dir / "task.toml").read_text())
    command = document["verifier"]["collect"][0]["command"].replace(
        "/logs/artifacts/candidate.patch", str(candidate)
    )
    env = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent)
        + os.pathsep
        + os.environ.get("PATH", ""),
    }
    subprocess.run(["bash", "-c", command], check=True, env=env)
    assert "+value = 1" in candidate.read_text()


@pytest.mark.parametrize("configuration", ["filter", "fsmonitor", "environment"])
def test_capture_ignores_agent_git_commands_and_keeps_its_index(
    task: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configuration: str
) -> None:
    root, config, _ = task
    marker = tmp_path / "agent-git-command-ran"
    command = f"touch {marker}; cat"
    if configuration == "filter":
        (root / ".gitattributes").write_text("widget.py filter=agent\n")
        _git(root, "config", "filter.agent.clean", command)
    elif configuration == "fsmonitor":
        _git(root, "config", "core.fsmonitor", command)
    else:
        monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", command)
    (root / "widget.py").write_text("value = 1\n")
    (root / "new_source.py").write_text("new_source = True\n")
    original_index = (root / ".git/index").read_bytes()
    candidate = tmp_path / "candidate.patch"
    collect(root, config["base_commit"], candidate)
    assert "+value = 1" in candidate.read_text()
    assert "+new_source = True" in candidate.read_text()
    assert not marker.exists()
    assert (root / ".git/index").read_bytes() == original_index


def test_unmodified_source_fails(task: tuple, tmp_path: Path) -> None:
    _, config, hidden = task
    candidate = tmp_path / "candidate.patch"
    candidate.write_text("")
    assert not verify(config, candidate, hidden)


def test_candidate_cannot_replace_hidden_grader(task: tuple, tmp_path: Path) -> None:
    root, config, hidden = task
    (root / "hidden_check.py").write_text("raise SystemExit(0)\n")
    candidate = _candidate(root, tmp_path, config["base_commit"])
    assert not verify(config, candidate, hidden)


def test_candidate_test_edits_do_not_hide_a_real_fix(
    task: tuple, tmp_path: Path
) -> None:
    root, config, hidden = task
    (root / "widget.py").write_text("value = 1\n")
    (root / "hidden_check.py").write_text("raise SystemExit(0)\n")
    candidate = _candidate(root, tmp_path, config["base_commit"])
    assert verify(config, candidate, hidden)
    assert "assert value == 1" in (root / "hidden_check.py").read_text()


def test_invalid_patch_does_not_become_a_verifier_error(
    task: tuple, tmp_path: Path
) -> None:
    _, config, hidden = task
    candidate = tmp_path / "candidate.patch"
    candidate.write_text("not a patch")
    assert not verify(config, candidate, hidden)


def test_symlink_submission_is_rejected(task: tuple, tmp_path: Path) -> None:
    _, config, hidden = task
    candidate = tmp_path / "candidate.patch"
    candidate.symlink_to(hidden)
    assert not verify(config, candidate, hidden)


def test_unpinned_verifier_base_is_an_infrastructure_error(
    task: tuple, tmp_path: Path
) -> None:
    _, config, hidden = task
    candidate = tmp_path / "candidate.patch"
    candidate.write_text("")
    config["base_commit"] = "0" * 40
    with pytest.raises(RuntimeError, match="base does not match"):
        verify(config, candidate, hidden)


def test_patch_cannot_write_outside_workspace(task: tuple, tmp_path: Path) -> None:
    _, config, hidden = task
    outside = tmp_path / "outside.txt"
    outside.write_text("safe\n")
    candidate = tmp_path / "candidate.patch"
    candidate.write_text(
        "diff --git a/../outside.txt b/../outside.txt\n"
        "--- a/../outside.txt\n+++ b/../outside.txt\n@@ -1 +1 @@\n-safe\n+changed\n"
    )
    assert not verify(config, candidate, hidden)
    assert outside.read_text() == "safe\n"


def test_sha256_capture_preserves_source_changes(tmp_path: Path) -> None:
    root = tmp_path / "sha256"
    root.mkdir()
    _git(root, "init", "--object-format=sha256")
    (root / "widget.py").write_text("value = 0\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "clean base")
    base = _git(root, "rev-parse", "HEAD").strip()
    assert len(base) == 64
    (root / "widget.py").write_text("value = 1\n")
    candidate = tmp_path / "candidate.patch"
    collect(root, base, candidate)
    assert "+value = 1" in candidate.read_text()
    _git(root, "reset", "--hard", base)
    _git(root, "apply", "--check", str(candidate))
    _git(root, "apply", str(candidate))
    assert (root / "widget.py").read_text() == "value = 1\n"


def test_capture_preserves_bytes_despite_repository_attributes(
    task: tuple, tmp_path: Path
) -> None:
    root, config, _ = task
    (root / ".gitattributes").write_text("*.py text eol=crlf\n")
    edited = b"value = 1\r\n"
    (root / "widget.py").write_bytes(edited)
    candidate = tmp_path / "candidate.patch"
    collect(root, config["base_commit"], candidate)
    (root / ".gitattributes").unlink()
    _git(root, "reset", "--hard", config["base_commit"])
    _git(root, "apply", str(candidate))
    assert (root / "widget.py").read_bytes() == edited


def test_sha1_capture_accepts_git_without_object_format(
    task: tuple, tmp_path: Path, monkeypatch
) -> None:
    root, config, _ = task
    real_run = subprocess.run

    def older_git(args, **kwargs):
        if "init" in args and any(arg.startswith("--object-format") for arg in args):
            raise subprocess.CalledProcessError(129, args)
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", older_git)
    (root / "widget.py").write_text("value = 1\n")
    candidate = tmp_path / "candidate.patch"
    collect(root, config["base_commit"], candidate)
    assert "+value = 1" in candidate.read_text()


def test_sparse_checkout_requires_materialization(tmp_path: Path) -> None:
    root = tmp_path / "sparse"
    (root / "src").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "src/widget.py").write_text("value = 0\n")
    (root / "docs/manual.md").write_text("keep this file\n")
    _git(root, "init")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "base")
    base = _git(root, "rev-parse", "HEAD").strip()
    _git(root, "sparse-checkout", "set", "src")
    assert not (root / "docs/manual.md").exists()
    (root / "src/widget.py").write_text("value = 1\n")
    candidate = tmp_path / "candidate.patch"
    with pytest.raises(ValueError, match="materialize sparse checkouts"):
        collect(root, base, candidate)
    assert not candidate.exists()


@pytest.mark.parametrize(
    "attack", ["json", "symlink", "json-directory", "text-directory", "preexisting-directories"]
)
def test_candidate_cannot_override_final_reward_artifact(
    task: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack: str
) -> None:
    root, config, hidden = task
    reward = tmp_path / "verifier/reward.txt"
    reward.parent.mkdir()
    victim = tmp_path / "trusted-file"
    victim.write_text("untouched")
    if attack == "json":
        payload = f"Path({str(reward.with_suffix('.json'))!r}).write_text('{{\"reward\": 1}}')"
    elif attack == "json-directory":
        payload = (
            f"Path({str(reward.with_suffix('.json'))!r}).mkdir()\n"
            f"Path({str(reward)!r}).write_text('1\\n')"
        )
    elif attack == "text-directory":
        payload = f"Path({str(reward)!r}).mkdir()"
    elif attack == "preexisting-directories":
        reward.mkdir()
        reward.with_suffix(".json").mkdir()
        (reward / "outside").symlink_to(victim)
        payload = "pass"
    else:
        payload = f"Path({str(reward)!r}).symlink_to({str(victim)!r})"
    if attack in {"json-directory", "text-directory"}:
        directory = reward.with_suffix(".json") if attack == "json-directory" else reward
        payload += (
            f"\nPath({str(directory / 'nested')!r}).mkdir()"
            f"\nPath({str(directory / 'outside')!r}).symlink_to({str(victim)!r})"
        )
    (root / "widget.py").write_text(
        "from pathlib import Path\n" + payload + "\nvalue = 0\n"
    )
    candidate = _candidate(root, tmp_path, config["base_commit"])
    config_file = tmp_path / "verifier.json"
    config_file.write_text(json.dumps(config))
    monkeypatch.setattr(
        code_verify,
        "Path",
        lambda value: (
            reward if str(value) == "/logs/verifier/reward.txt" else Path(value)
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "code_verify",
            "--config",
            str(config_file),
            "--candidate",
            str(candidate),
            "--hidden-patch",
            str(hidden),
        ],
    )
    code_verify.main()
    assert not reward.with_suffix(".json").exists()
    assert not reward.is_symlink()
    assert reward.read_text() == "0\n"
    assert victim.read_text() == "untouched"


@pytest.mark.parametrize("path", [".github/workflows/ci.yml", ".gitlab-ci.yml"])
def test_ci_metadata_fixes_are_not_filtered(
    task: tuple, tmp_path: Path, path: str
) -> None:
    root, config, hidden = task
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("fixed\n")
    candidate = _candidate(root, tmp_path, config["base_commit"])
    config["test_command"] = f"test -f {path}"
    assert verify(config, candidate, hidden)
    assert target.read_text() == "fixed\n"


@pytest.mark.parametrize(
    "command,expected",
    [("false | true", False), ("false; true", False), ("false || true", True)],
)
def test_verifier_shell_preserves_failure_semantics(
    task: tuple, tmp_path: Path, command: str, expected: bool
) -> None:
    _, config, hidden = task
    candidate = tmp_path / "candidate.patch"
    candidate.write_text("")
    config["test_command"] = command
    assert verify(config, candidate, hidden) is expected


@pytest.mark.parametrize(
    "command,expected",
    [
        ("yes | head -n 1 >/dev/null", False),
        ('{ yes || test "$?" -eq 141; } | head -n 1 >/dev/null', True),
        ('{ false || test "$?" -eq 141; } | head -n 1 >/dev/null', False),
    ],
)
def test_trusted_pipeline_handles_expected_sigpipe_explicitly(
    task: tuple, tmp_path: Path, command: str, expected: bool
) -> None:
    _, config, hidden = task
    candidate = tmp_path / "candidate.patch"
    candidate.write_text("")
    config["test_command"] = command
    assert verify(config, candidate, hidden) is expected


@pytest.mark.parametrize("conversion", ["crlf", "ident", "filter"])
@pytest.mark.parametrize("edited", [False, True])
def test_converted_checkout_roundtrips_raw_candidate_bytes(
    task: tuple, tmp_path: Path, conversion: str, edited: bool
) -> None:
    root, config, hidden = task
    attributes = {"crlf": "text eol=crlf", "ident": "ident", "filter": "filter=expand"}
    (root / ".gitattributes").write_text(f"widget.py {attributes[conversion]}\n")
    (root / "widget.py").write_text("# $Id$ BASE\nvalue = 0\n")
    if conversion == "filter":
        _git(root, "config", "filter.expand.clean", "sed 's/EXPANDED/BASE/g'")
        _git(root, "config", "filter.expand.smudge", "sed 's/BASE/EXPANDED/g'")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "base with worktree conversion")
    config["base_commit"] = _git(root, "rev-parse", "HEAD").strip()
    (root / "widget.py").unlink()
    _git(root, "checkout", "--", "widget.py")
    submitted = (root / "widget.py").read_bytes()
    if conversion == "crlf":
        assert b"\r\n" in submitted
    elif conversion == "ident":
        assert b"$Id:" in submitted
    else:
        assert b"EXPANDED" in submitted
    if edited:
        submitted = submitted.replace(b"value = 0", b"value = 1")
        (root / "widget.py").write_bytes(submitted)
    candidate = tmp_path / "candidate.patch"
    collect(root, config["base_commit"], candidate)
    _git(root, "reset", "--hard", config["base_commit"])
    assert verify(config, candidate, hidden) is edited
    assert (root / "widget.py").read_bytes() == submitted


def test_legacy_git_cannot_run_system_fsmonitor(
    task: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config, _ = task
    marker = tmp_path / "fsmonitor-ran"
    hook = tmp_path / "fsmonitor"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
    hook.chmod(0o755)
    system_config = tmp_path / "system.gitconfig"
    system_config.write_text(f"[core]\nfsmonitor = {hook}\n")
    original_run = subprocess.run

    def legacy_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        # Emulate old Git loading system config despite GIT_CONFIG_SYSTEM=/dev/null.
        env = dict(kwargs["env"])
        env["GIT_CONFIG_SYSTEM"] = str(system_config)
        kwargs["env"] = env
        return original_run(args, **kwargs)

    monkeypatch.setattr(code_collect.subprocess, "run", legacy_run)
    (root / "widget.py").write_text("value = 1\n")
    candidate = tmp_path / "candidate.patch"
    collect(root, config["base_commit"], candidate)
    assert not marker.exists()
    assert "+value = 1" in candidate.read_text()


@pytest.mark.parametrize("fixture_path", ["fixtures/model.bin", "tests/model.bin"])
@pytest.mark.parametrize("raw_capture", [False, True])
def test_verifier_preserves_unedited_filtered_assets(
    task: tuple, tmp_path: Path, fixture_path: str, raw_capture: bool
) -> None:
    root, config, hidden = task
    fixture = root / fixture_path
    fixture.parent.mkdir()
    fixture.write_text("BASE\n")
    (root / ".gitattributes").write_text(f"{fixture_path} filter=expand\n")
    _git(root, "config", "filter.expand.clean", "sed 's/EXPANDED/BASE/g'")
    _git(root, "config", "filter.expand.smudge", "sed 's/BASE/EXPANDED/g'")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "base with a filtered fixture")
    config["base_commit"] = _git(root, "rev-parse", "HEAD").strip()
    fixture.unlink()
    _git(root, "checkout", "--", fixture_path)
    assert fixture.read_text() == "EXPANDED\n"
    (root / "widget.py").write_text("value = 1\n")
    if raw_capture:
        candidate = tmp_path / "candidate.patch"
        collect(root, config["base_commit"], candidate)
        _git(root, "reset", "--hard", config["base_commit"])
    else:
        candidate = _candidate(root, tmp_path, config["base_commit"])
    config["test_command"] += f' && test "$(cat {fixture_path})" = EXPANDED'
    assert verify(config, candidate, hidden)
    assert fixture.read_text() == "EXPANDED\n"
