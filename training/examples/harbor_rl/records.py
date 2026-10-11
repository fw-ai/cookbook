"""Lightweight task-row and run-record helpers for the Harbor recipes.

Kept free of training imports so dataset preparation stays fast.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any


def select_rows(
    rows: Sequence[Mapping[str, Any]], task_names: Iterable[str]
) -> list[dict[str, Any]]:
    """Copy the rows for ``task_names`` in the caller's order."""
    names = list(task_names)
    if len(names) != len(set(names)):
        raise ValueError("Harbor task selection contains duplicates")
    by_name = {str(row.get("task_name") or ""): row for row in rows}
    if len(by_name) != len(rows) or "" in by_name:
        raise ValueError("Harbor rows must have unique, non-empty task_name values")
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Harbor dataset is missing tasks: {missing}")
    return [dict(by_name[name]) for name in names]


def _json_value(value: Any) -> Any:
    """Encode enums and pydantic models found in recipe configs."""
    if isinstance(value, Enum):
        return value.value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return str(value)


def write_json(path: Path, document: Mapping[str, Any]) -> Path:
    """Atomically write one sorted, indented JSON document."""
    encoded = (
        json.dumps(document, indent=2, sort_keys=True, default=_json_value) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(root: Path) -> str:
    """Hash a directory's relative layout, file bytes, and symlink targets."""
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


def dirhash_sha256(path: Path) -> str:
    """Hash a task directory with ``dirhash`` (the published DABstep algorithm)."""
    try:
        from dirhash import dirhash
    except ImportError as exc:
        raise RuntimeError(
            "DABstep task hashing requires the example-only 'dirhash' "
            "dependency; install it alongside Harbor"
        ) from exc
    return str(dirhash(str(path), "sha256"))


__all__ = [
    "dirhash_sha256",
    "file_sha256",
    "select_rows",
    "tree_sha256",
    "write_json",
]
