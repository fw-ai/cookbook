"""Harbor task-image preparation shared by agent harness adapters."""

from __future__ import annotations

import json
import re
import shutil
import stat
import subprocess
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Callable

_NODE_22_INSTALL = r"""RUN set -eu; \
 if command -v apt-get >/dev/null 2>&1; then \
   apt-get update; \
   apt-get install -y --no-install-recommends curl ca-certificates python3 python3-pip python3-venv; \
   curl -fsSL https://deb.nodesource.com/setup_22.x | bash -; \
   TITO_HARNESS_NODE_DEB_VERSION="$(apt-cache madison nodejs | awk '$3 ~ /^22[.]/ { print $3; exit }')"; \
   test -n "$TITO_HARNESS_NODE_DEB_VERSION"; \
   apt-get install -y --no-install-recommends --allow-downgrades nodejs="$TITO_HARNESS_NODE_DEB_VERSION"; \
   apt-get clean; \
   rm -rf /var/lib/apt/lists/*; \
 elif command -v dnf >/dev/null 2>&1; then \
   dnf install -y curl ca-certificates python3 python3-pip; \
   curl -fsSL https://rpm.nodesource.com/setup_22.x | bash -; \
   dnf install -y nodejs; \
   dnf clean all; \
 else \
   echo 'unsupported base image: expected apt-get or dnf' >&2; \
   exit 1; \
 fi; \
 ln -sf /usr/bin/node /usr/local/bin/node; \
 ln -sf /usr/bin/npm /usr/local/bin/npm; \
 ln -sf /usr/bin/npx /usr/local/bin/npx; \
 node --version | grep -E '^v22[.]'; \
"""
_PYTHON_INSTALL = r"""RUN set -eu; \
 if command -v apt-get >/dev/null 2>&1; then \
   apt-get update; \
   apt-get install -y --no-install-recommends ca-certificates python3 python3-pip python3-venv; \
   apt-get clean; \
   rm -rf /var/lib/apt/lists/*; \
 elif command -v dnf >/dev/null 2>&1; then \
   dnf install -y ca-certificates python3 python3-pip; \
   dnf clean all; \
 else \
   echo 'unsupported base image: expected apt-get or dnf' >&2; \
   exit 1; \
 fi; \
"""
_SIDECAR_PYTHON_INSTALL = r"""python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'; \
 python3 -m venv /opt/fireworks-tito; \
 /opt/fireworks-tito/bin/python -m pip install --no-cache-dir \
   aiohttp==3.14.3 \
   httpx==0.28.1 \
   urllib3==2.7.0 \
   jinja2==3.1.6 \
   tokenizers==0.22.2 \
   transformers==5.5.4 \
   numpy==2.4.6; \
 /opt/fireworks-tito/bin/python -c 'import aiohttp, httpx, jinja2, numpy, tokenizers, transformers, urllib3'; \
"""


def build_python_sidecar_suffix(*, marker: str, restore_user: str | None) -> str:
    """Build a Python-only image layer for a harness installed by Harbor."""
    suffix = (
        f"\n\n{marker}\nUSER root\n{_PYTHON_INSTALL}{_SIDECAR_PYTHON_INSTALL}true\n"
    )
    if restore_user is not None:
        suffix += f"USER {restore_user}\n"
    return suffix


def build_node_22_harness_suffix(
    *,
    marker: str,
    package_install: str,
    version_check: str,
    restore_user: str | None,
) -> str:
    """Build the pinned-Node installer used by agent harness images."""
    suffix = (
        f"\n\n{marker}\n"
        "USER root\n"
        f"{_NODE_22_INSTALL}{_SIDECAR_PYTHON_INSTALL}{package_install}; \\\n"
        f" {version_check}\n"
    )
    if restore_user is not None:
        suffix += f"USER {restore_user}\n"
    return suffix


def _final_stage_user(dockerfile: str) -> str | None:
    user: str | None = None
    for raw_line in dockerfile.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        instruction = parts[0].upper()
        if instruction == "FROM":
            user = None
        elif instruction == "USER" and len(parts) == 2:
            user = parts[1]
    return user


def _base_image_user(image: str) -> str | None:
    inspected = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .Config.User}}", image],
        capture_output=True,
        check=False,
        text=True,
    )
    if inspected.returncode != 0:
        detail = inspected.stderr.strip() or inspected.stdout.strip()
        suffix = f": {detail}" if detail else ""
        raise ValueError(
            "base_image must be available to the local Docker daemon so its "
            f"runtime user can be preserved{suffix}"
        )
    try:
        user = json.loads(inspected.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("Docker returned an invalid base-image user") from exc
    if not isinstance(user, str):
        raise ValueError("Docker returned a non-string base-image user")
    return user or None


def task_output_path(target: Path, *parts: str) -> Path:
    """Resolve a write target without following task-provided symlinks."""
    path = target.joinpath(*parts)
    cursor = target
    for part in parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(
                f"task {target.name!r} write path contains a symlink: {Path(*parts)}"
            )
    try:
        path.resolve().relative_to(target.resolve())
    except ValueError as exc:
        raise ValueError(
            f"task {target.name!r} write path escapes the prepared task: {Path(*parts)}"
        ) from exc
    return path


def _make_environment_container_readable(environment: Path) -> None:
    for path in (environment, *environment.rglob("*")):
        if path.is_symlink():
            continue
        mode = path.stat().st_mode
        if path.is_dir():
            readable = stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
        elif path.is_file():
            readable = stat.S_IRGRP | stat.S_IROTH
        else:
            continue
        path.chmod(mode | readable)


def _disable_prebuilt_image(task_config: Path) -> None:
    source = task_config.read_text(encoding="utf-8")
    section: str | None = None
    removed = 0
    kept: list[str] = []
    for line in source.splitlines(keepends=True):
        header = re.match(r"^\s*\[([^]]+)]\s*(?:#.*)?$", line.rstrip("\r\n"))
        if header is not None:
            section = header.group(1).strip()
        if section == "environment" and re.match(r"^\s*docker_image\s*=", line):
            removed += 1
            continue
        kept.append(line)
    if removed > 1:
        raise ValueError(
            f"task config has duplicate environment.docker_image: {task_config}"
        )
    rewritten = "".join(kept)
    parsed = tomllib.loads(rewritten)
    if "docker_image" in parsed.get("environment", {}):
        raise ValueError(f"could not disable prebuilt image in {task_config}")
    task_config.write_text(rewritten, encoding="utf-8")


def prepare_with_installer(
    source: Path,
    destination: Path,
    *,
    marker: str,
    compatible_markers: tuple[str, ...] = (),
    suffix_builder: Callable[[str | None], str],
    base_image: str | None = None,
    task_names: Iterable[str] | None = None,
) -> list[Path]:
    """Copy task contexts and append one pinned agent installation layer."""
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(
            f"destination already exists: {destination}; choose a fresh path"
        )
    source_is_task = (source / "task.toml").is_file()
    if source_is_task:
        if task_names is not None:
            raise ValueError("task_names is valid only when source is a task root")
        tasks = [source]
    elif task_names is None:
        tasks = sorted(
            path for path in source.iterdir() if (path / "task.toml").is_file()
        )
    else:
        names = tuple(str(name) for name in task_names)
        if not names:
            raise ValueError("task_names must not be empty")
        if len(names) != len(set(names)):
            raise ValueError("task_names must be unique")
        invalid = [
            name for name in names if Path(name).name != name or name in {".", ".."}
        ]
        if invalid:
            raise ValueError(f"task_names contain invalid path components: {invalid}")
        tasks = [source / name for name in names]
        missing = [path.name for path in tasks if not (path / "task.toml").is_file()]
        if missing:
            raise ValueError(f"source is missing requested Harbor tasks: {missing}")
    if not tasks:
        raise ValueError(f"source contains no Harbor tasks: {source}")
    if base_image is not None:
        if len(tasks) != 1:
            raise ValueError("base_image is valid only for one prepared task")
        if re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", base_image) is None:
            raise ValueError("base_image must be an immutable image@sha256 digest")
    base_restore_user = _base_image_user(base_image) if base_image is not None else None

    destination.mkdir(parents=True)
    prepared: list[Path] = []
    for task in tasks:
        target = destination / task.name
        shutil.copytree(task, target, symlinks=True)
        _make_environment_container_readable(task_output_path(target, "environment"))
        dockerfile = task_output_path(target, "environment", "Dockerfile")
        if not dockerfile.is_file():
            raise ValueError(f"task {task.name!r} has no environment/Dockerfile")
        source_dockerfile = dockerfile.read_text(encoding="utf-8")
        existing_marker = next(
            (
                candidate
                for candidate in (marker, *compatible_markers)
                if candidate in source_dockerfile
            ),
            None,
        )
        if existing_marker is not None:
            raise ValueError(f"task {task.name!r} already contains {existing_marker!r}")
        original = (
            f"FROM {base_image}\n" if base_image is not None else source_dockerfile
        )
        restore_user = (
            base_restore_user
            if base_image is not None
            else _final_stage_user(source_dockerfile)
        )
        dockerfile.write_text(
            original.rstrip() + "\n" + suffix_builder(restore_user),
            encoding="utf-8",
        )
        _disable_prebuilt_image(task_output_path(target, "task.toml"))
        prepared.append(target)
    return prepared


__all__ = [
    "build_node_22_harness_suffix",
    "build_python_sidecar_suffix",
    "prepare_with_installer",
    "task_output_path",
]
