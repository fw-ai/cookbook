"""Prepare every DeepSWE task for OpenCode and write a content manifest."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from training.examples.harbor_rl.harnesses.opencode.constants import DEFAULT_OPENCODE_VERSION
from training.examples.harbor_rl.harnesses.opencode.prepare_tasks import prepare
from training.examples.harbor_rl.records import tree_sha256, write_json


def _git_value(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def prepare_deep_swe(
    source_repository: Path,
    destination: Path,
    manifest_path: Path,
    *,
    opencode_version: str,
) -> dict[str, object]:
    """Prepare the complete pinned checkout and record source/output hashes."""

    repository = source_repository.expanduser().resolve()
    task_root = repository / "tasks"
    if not (repository / ".git").exists() or not task_root.is_dir():
        raise ValueError("source must be a DeepSWE Git checkout containing tasks/")
    if _git_value(repository, "status", "--porcelain"):
        raise ValueError("DeepSWE source checkout must be clean")

    source_revision = _git_value(repository, "rev-parse", "HEAD")
    source_remote = _git_value(repository, "remote", "get-url", "origin")
    source_tasks = sorted(
        path for path in task_root.iterdir() if (path / "task.toml").is_file()
    )
    prepared = prepare(task_root, destination, opencode_version)
    if [path.name for path in prepared] != [path.name for path in source_tasks]:
        raise RuntimeError(
            "prepared DeepSWE membership differs from the source checkout"
        )

    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": "deep-swe",
        "source": {
            "remote": source_remote,
            "revision": source_revision,
        },
        "opencode_version": opencode_version,
        "task_count": len(source_tasks),
        "tasks": [path.name for path in source_tasks],
        "source_sha256": {path.name: tree_sha256(path) for path in source_tasks},
        "prepared_sha256": {path.name: tree_sha256(path) for path in prepared},
    }
    write_json(manifest_path.expanduser().resolve(), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repository", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--opencode-version", default=DEFAULT_OPENCODE_VERSION)
    args = parser.parse_args()
    manifest = prepare_deep_swe(
        args.source_repository,
        args.destination,
        args.manifest,
        opencode_version=args.opencode_version,
    )
    print(
        f"prepared {manifest['task_count']} DeepSWE tasks at "
        f"{args.destination} from {manifest['source']['revision']}"
    )


if __name__ == "__main__":
    main()
