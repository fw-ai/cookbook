"""Grade an untrusted source patch in a fresh, pinned task workspace."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

MAX_PATCH_BYTES = 16 * 1024 * 1024
_PROTECTED = (
    ".git",
    ".git/**",
    "tests/**",
    "test/**",
    "**/tests/**",
    "**/test/**",
    "*_test.go",
    "**/*_test.go",
    "test_*.py",
    "**/test_*.py",
    "**/conftest.py",
    "pytest.ini",
    "conftest.py",
)


def verify(config: dict, candidate: Path, hidden_patch: Path) -> bool:
    cwd = Path(config["cwd"])
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
    git = [
        "git",
        "-C",
        str(cwd),
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
    ]

    def run(args: list[str], *, input: bytes | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            args, cwd=cwd, env=env, input=input, capture_output=True, check=False
        )

    base = run([*git, "rev-parse", "HEAD"])
    if base.returncode or base.stdout.decode().strip() != config["base_commit"]:
        raise RuntimeError("fresh code verifier base does not match the task manifest")
    if run([*git, "diff", "--quiet", "HEAD"]).returncode:
        raise RuntimeError("fresh code verifier workspace has modified tracked files")
    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or candidate.stat().st_size > MAX_PATCH_BYTES
    ):
        return False
    # Hidden test paths are trusted metadata. Git supplies unquoted NUL-delimited
    # names; exclusions keep the candidate from replacing the grading tests.
    paths = run([*git, "apply", "--numstat", "-z", str(hidden_patch)])
    if paths.returncode:
        raise RuntimeError("hidden test patch is invalid for this task")
    if run([*git, "apply", "--check", str(hidden_patch)]).returncode:
        raise RuntimeError("hidden test patch does not apply to the clean task base")
    hidden_paths = [
        record.split(b"\t", 2)[-1].decode()
        for record in paths.stdout.split(b"\0")
        if record
    ]
    excludes = [f"--exclude={path}" for path in (*_PROTECTED, *hidden_paths)]
    if candidate.stat().st_size:
        # Capture emits --no-renames patches. Only normalize accepted paths;
        # unrelated and protected fixtures retain their image checkout contents.
        candidate_paths = run(
            [*git, "apply", "--numstat", "-z", *excludes, str(candidate)]
        )
        if candidate_paths.returncode:
            return False
        touched = {
            record.split(b"\t", 2)[-1]
            for record in candidate_paths.stdout.split(b"\0")
            if record
        }
        tracked = run([*git, "ls-files", "-z"])
        if tracked.returncode:
            raise RuntimeError("fresh code verifier cannot list tracked base files")
        raw_paths = sorted(touched.intersection(tracked.stdout.split(b"\0")))
        # Capture preserves raw bytes without agent content filters. Override
        # conversions for applying those bytes, never re-smudging candidate code.
        git_dir = run([*git, "rev-parse", "--absolute-git-dir"])
        if git_dir.returncode:
            raise RuntimeError("fresh code verifier cannot locate its Git metadata")
        attributes = Path(os.fsdecode(git_dir.stdout).strip()) / "info/attributes"
        attributes.parent.mkdir(exist_ok=True)
        attributes.write_text("* -text -filter -ident -working-tree-encoding -crlf\n")
        if raw_paths and run(
            [*git, "checkout-index", "--force", "--index", "-z", "--stdin"],
            input=b"\0".join(raw_paths) + b"\0",
        ).returncode:
            raise RuntimeError("fresh code verifier cannot materialize its raw base")
        if run(
            [*git, "apply", "--check", "--index", *excludes, str(candidate)]
        ).returncode:
            return False
        if run([*git, "apply", "--index", *excludes, str(candidate)]).returncode:
            return False
    if run([*git, "apply", str(hidden_patch)]).returncode:
        return False
    result = subprocess.run(
        ["bash", "-e", "-u", "-o", "pipefail", "-c", config["test_command"]],
        cwd=cwd,
        check=False,
    )
    return result.returncode == 0


def _remove_reward_artifact(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--hidden-patch", type=Path, required=True)
    args = parser.parse_args()
    reward = Path("/logs/verifier/reward.txt")
    _remove_reward_artifact(reward)
    _remove_reward_artifact(reward.with_suffix(".json"))
    passed = verify(
        json.loads(args.config.read_text()), args.candidate, args.hidden_patch
    )
    # Candidate code may create either format; Harbor prefers JSON over text.
    # Reclaim both paths so synchronous writes cannot override the final verdict.
    _remove_reward_artifact(reward.with_suffix(".json"))
    _remove_reward_artifact(reward)
    reward.parent.mkdir(parents=True, exist_ok=True)
    reward.write_text("1\n" if passed else "0\n")


if __name__ == "__main__":
    main()
