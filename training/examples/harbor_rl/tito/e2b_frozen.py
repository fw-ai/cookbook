"""Harbor 0.21 extension for already-built, content-pinned E2B templates.

Main and verifier snapshots remain separate. Runtime never rebuilds an image or
hashes a multi-GB build archive. Small task/runtime files still fail on drift.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from e2b import AsyncTemplate
from harbor.environments.definition import environment_content_hash
from harbor.environments.e2b import E2BEnvironment

IGNORED = {".DS_Store", ".git", "__pycache__"}


def context_files(directory, *, build_assets=()):
    files = {}
    assets = {}
    for path in sorted(Path(directory).rglob("*")):
        relative = path.relative_to(directory)
        if IGNORED & set(relative.parts):
            continue
        if path.is_symlink():
            raise ValueError("Frozen Dockerfile contexts do not support symlinks; materialize the input first")
        if not path.is_file():
            continue
        if relative.as_posix() in build_assets:
            assets[relative.as_posix()] = path.stat().st_size
        else:
            if path.stat().st_size > 16 * 1024**2:
                raise ValueError("Large build input requires an explicit build_assets pin")
            files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files, assets


@lru_cache(maxsize=8)
def _cached_manifest(path: str, identity: tuple):
    return json.loads(Path(path).read_text())


def _manifest(path: str):
    stat = Path(path).stat()
    return _cached_manifest(path, (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))


def _file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


async def pin_e2b_context(directory, task_name: str, *, cpus: int, memory_mb: int, build_assets=()) -> dict:
    """Pin an already-built Dockerfile context; hash build assets only once.

    Store the result in ``{"format_version": 1, "contexts":
    {"<task-short-name>:<environment|tests>": entry}}``. Pin both contexts
    when using a separate verifier. Template builds/qualification happen first.
    """
    directory = Path(directory)
    if directory.name not in {"environment", "tests"} or not (directory / "Dockerfile").is_file():
        raise ValueError("Frozen E2B pins require a prepared environment/ or tests/ Dockerfile context")
    small, assets = context_files(directory, build_assets=build_assets)
    content_hash = environment_content_hash(directory)
    alias = f"{task_name}__{content_hash[:12]}".replace("/", "__").replace(".", "-")
    # Never move a previous run's tag when the same context is rebuilt.
    tag = "frozen-" + uuid4().hex
    assigned = await AsyncTemplate.assign_tags(alias, tag)
    return {
        "role": directory.name,
        "content_hash": content_hash,
        "template_alias": alias,
        "template_ref": alias + ":" + tag,
        "tag": tag,
        "build_id": assigned.build_id,
        "cpus": cpus,
        "memory_mb": memory_mb,
        "small_files": small,
        "build_assets": {
            name: {"bytes": size, "sha256": _file_digest(directory / name)} for name, size in assets.items()
        },
    }


class FrozenE2BEnvironment(E2BEnvironment):
    """Use Harbor's environment extension point with a frozen template manifest."""

    def __init__(self, environment_dir, environment_name, *args, frozen_manifest, **kwargs):
        document = _manifest(str(Path(frozen_manifest).resolve()))
        if document["format_version"] != 1:
            raise ValueError("Unsupported frozen E2B manifest version")
        role = Path(environment_dir).name
        key = environment_name.split("/")[-1] + ":" + role
        try:
            self._frozen_pin = document["contexts"][key]
        except KeyError:
            raise ValueError(f"Frozen template manifest has no entry for {key}; pin every required context") from None
        if self._frozen_pin["role"] != role:
            raise ValueError("Frozen agent/verifier role mismatch")
        files, assets = context_files(Path(environment_dir), build_assets=self._frozen_pin["build_assets"])
        if files != self._frozen_pin["small_files"]:
            raise ValueError("Frozen task files changed; rebuild and requalify the template")
        if set(assets) - set(self._frozen_pin["build_assets"]):
            raise ValueError("Unexpected build asset in a frozen task context")
        if any(size != self._frozen_pin["build_assets"][name]["bytes"] for name, size in assets.items()):
            raise ValueError("Frozen build input size changed; rebuild and requalify the template")
        super().__init__(environment_dir, environment_name, *args, **kwargs)
        if self.task_env_config.docker_image is not None:
            raise ValueError("Frozen E2B manifests require Dockerfile contexts, not environment.docker_image")
        for name, actual in (("cpus", self._effective_cpus), ("memory_mb", self._effective_memory_mb)):
            if self._frozen_pin.get(name) != actual:
                raise ValueError(f"Frozen template {name} does not match the task resources")
        self._template_name = self._frozen_pin["template_ref"]

    @property
    def environment_id(self):
        return self._frozen_pin["content_hash"]

    async def _does_template_exist(self):
        pin = self._frozen_pin
        tags = await AsyncTemplate.get_tags(pin["template_alias"])
        if not any(tag.tag == pin["tag"] and tag.build_id == pin["build_id"] for tag in tags):
            raise RuntimeError("Frozen E2B template tag is missing or changed; do not rebuild during a run")
        return True

    async def _create_template(self):
        raise RuntimeError("Frozen E2B environments cannot build templates during a run")

    async def start(self, force_build):
        if force_build:
            raise ValueError("force_build is incompatible with a frozen template")
        await super().start(force_build=False)
