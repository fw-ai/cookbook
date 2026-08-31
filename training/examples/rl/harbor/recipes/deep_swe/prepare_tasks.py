"""Prepare every DeepSWE task for OpenCode and write a content manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from training.examples.rl.harbor.opencode.constants import DEFAULT_OPENCODE_VERSION
from training.examples.rl.harbor.opencode.prepare_tasks import prepare


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix().encode()
        if path.is_symlink():
            digest.update(
                b"L\0" + relative + b"\0" + os.readlink(path).encode() + b"\0"
            )
        elif path.is_dir():
            digest.update(b"D\0" + relative + b"\0")
        elif path.is_file():
            digest.update(b"F\0" + relative + b"\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()


def _git_value(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _atomic_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


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
        "source_sha256": {path.name: _tree_sha256(path) for path in source_tasks},
        "prepared_sha256": {path.name: _tree_sha256(path) for path in prepared},
    }
    _atomic_json(manifest_path.expanduser().resolve(), manifest)
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
