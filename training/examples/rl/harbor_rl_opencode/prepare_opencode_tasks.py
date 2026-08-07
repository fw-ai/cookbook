"""Copy a Harbor task tree and bake a pinned OpenCode CLI into each image.

Installing Node and OpenCode inside every rollout container is prohibitively
expensive for an 8-completion RL run.  This preparation step appends one cached
image layer per task, while preserving the task's native Docker networking and
verifier configuration.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from training.examples.rl.harbor_rl_opencode.harbor import DEFAULT_OPENCODE_VERSION

_MARKER = "# Added by harbor_rl_opencode.prepare_opencode_tasks"
_SUFFIX = r"""

# Added by harbor_rl_opencode.prepare_opencode_tasks
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && npm install -g opencode-ai@{version} \
 && opencode --version \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
"""

_ISOLATED_COMPOSE = """\
services:
  main:
    build:
      context: ${{CONTEXT_DIR}}
    image: ${{MAIN_IMAGE_NAME}}
    command: [ "sh", "-c", "sleep infinity" ]
    networks:
      - isolated
    environment:
      - TEST_DIR=${{TEST_DIR}}
    volumes:
      - ${{HOST_VERIFIER_LOGS_PATH}}:${{ENV_VERIFIER_LOGS_PATH}}
      - ${{HOST_AGENT_LOGS_PATH}}:${{ENV_AGENT_LOGS_PATH}}
    deploy:
      resources:
        limits:
          cpus: ${{CPUS}}
          memory: ${{MEMORY}}

networks:
  isolated:
    name: {network}
    external: true
"""


def _final_stage_user(dockerfile: str) -> str | None:
    """Return the last explicit user in the final Docker build stage."""
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


def _opencode_suffix(version: str, *, restore_user: str | None) -> str:
    suffix = _SUFFIX.format(version=version)
    if restore_user is not None:
        suffix += f"USER {restore_user}\n"
    return suffix


def _ensure_internal_network(name: str) -> None:
    inspect = subprocess.run(
        ["docker", "network", "inspect", name],
        capture_output=True,
        check=False,
    )
    if inspect.returncode == 0:
        options = subprocess.run(
            [
                "docker",
                "network",
                "inspect",
                name,
                "--format",
                "{{.Internal}}",
            ],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        if options != "true":
            raise ValueError(f"Docker network {name!r} exists but is not internal")
        return
    subprocess.run(
        ["docker", "network", "create", "--internal", name],
        capture_output=True,
        check=True,
    )


def _task_output_path(target: Path, *parts: str) -> Path:
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


def prepare(
    source: Path,
    destination: Path,
    version: str,
    *,
    internal_network: str | None = None,
) -> list[Path]:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(
            f"destination already exists: {destination}; choose a fresh path"
        )
    tasks = sorted(
        [source]
        if (source / "task.toml").is_file()
        else [path for path in source.iterdir() if (path / "task.toml").is_file()]
    )
    if not tasks:
        raise ValueError(f"source contains no Harbor tasks: {source}")

    destination.mkdir(parents=True)
    prepared: list[Path] = []
    for task in tasks:
        target = destination / task.name
        shutil.copytree(task, target, symlinks=True)
        dockerfile = _task_output_path(target, "environment", "Dockerfile")
        if not dockerfile.is_file():
            raise ValueError(f"task {task.name!r} has no environment/Dockerfile")
        original = dockerfile.read_text(encoding="utf-8")
        if _MARKER in original:
            raise ValueError(f"task {task.name!r} is already OpenCode-prepared")
        dockerfile.write_text(
            original.rstrip()
            + "\n"
            + _opencode_suffix(
                version,
                restore_user=_final_stage_user(original),
            ),
            encoding="utf-8",
        )
        if internal_network:
            compose = _task_output_path(
                target,
                "environment",
                "docker-compose.yaml",
            )
            compose.write_text(
                _ISOLATED_COMPOSE.format(network=internal_network),
                encoding="utf-8",
            )
        prepared.append(target)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--opencode-version", default=DEFAULT_OPENCODE_VERSION)
    parser.add_argument(
        "--internal-network",
        default=None,
        help="Optional Docker --internal network: no egress, host bridge remains reachable",
    )
    args = parser.parse_args()
    if args.internal_network:
        _ensure_internal_network(args.internal_network)
    prepared = prepare(
        args.source,
        args.destination,
        args.opencode_version,
        internal_network=args.internal_network,
    )
    print(
        f"prepared {len(prepared)} Harbor task image contexts in "
        f"{args.destination} with opencode-ai@{args.opencode_version}"
    )


if __name__ == "__main__":
    main()
