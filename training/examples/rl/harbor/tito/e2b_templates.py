"""Prebuild Harbor E2B templates once before rollout fan-out."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.examples.rl.harbor.tito.trial import (
    DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS,
    HARBOR_TASK_CONFIG_KEY,
    _build_trial_config,
    _is_retryable_e2b_default_tag_not_found,
    _require_harbor,
    _task_local_path,
    _task_prebuilt_image,
    load_harbor_trial_config,
    task_config_from_row,
    task_name_from_row,
)

_LOCAL_PREPARED_IMAGE_PREFIX = "fireworks-harbor-prepared--"
_E2B_RESOURCE_IDENTITY_FILE = ".harbor-e2b-resources.json"


def _ensure_e2b_resource_identity(environment: Any) -> bool:
    """Include E2B build resources in Harbor's content-addressed alias."""

    cpus = getattr(environment, "_effective_cpus", None)
    memory_mb = getattr(environment, "_effective_memory_mb", None)
    if cpus is None and memory_mb is None:
        return False
    marker = Path(environment.environment_dir) / _E2B_RESOURCE_IDENTITY_FILE
    docker_image = getattr(environment.task_env_config, "docker_image", None)
    encoded = json.dumps(
        {"cpus": cpus, "docker_image": docker_image, "memory_mb": memory_mb},
        sort_keys=True,
    ) + "\n"
    if marker.exists() and marker.read_text(encoding="utf-8") == encoded:
        return False
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(encoded, encoding="utf-8")
    return True


def _remove_local_docker_image_pin(config_path: Path) -> None:
    """Remove only the host-local image pin that E2B cannot resolve."""

    source = config_path.read_text(encoding="utf-8")
    section: str | None = None
    output: list[str] = []
    removed = False
    for line in source.splitlines(keepends=True):
        header = re.match(r"^\s*\[([^]]+)]\s*(?:#.*)?$", line.rstrip("\r\n"))
        if header is not None:
            section = header.group(1).strip()
        if section == "environment" and re.match(
            r'^\s*docker_image\s*=\s*["\']fireworks-harbor-prepared--', line
        ):
            removed = True
            continue
        output.append(line)
    if not removed:
        raise ValueError(f"could not remove local Docker image pin from {config_path}")
    config_path.write_text("".join(output), encoding="utf-8")


def isolate_e2b_task_rows(
    task_rows: Sequence[Mapping[str, Any]], *, task_root: str | Path
) -> list[dict[str, Any]]:
    """Copy local tasks and drop Docker-only cache pins before E2B builds.

    The local Docker backend writes a content-addressed, host-local image tag
    into ``task.toml``. E2B cannot pull that tag. Keeping a private task copy
    avoids mutating a dataset that an active Docker run may still be using.
    """

    destination_root = Path(task_root).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    isolated_configs: dict[Path, dict[str, Any]] = {}
    output: list[dict[str, Any]] = []
    for row in task_rows:
        task_config = task_config_from_row(row)
        source_path = _task_local_path(task_config)
        if source_path is None:
            output.append(dict(row))
            continue
        config = isolated_configs.get(source_path)
        if config is None:
            task_name = task_name_from_row(row)
            suffix = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:12]
            destination = destination_root / f"{task_name}-{suffix}"
            if not destination.exists():
                shutil.copytree(source_path, destination)
            if (_task_prebuilt_image(task_config) or "").startswith(
                _LOCAL_PREPARED_IMAGE_PREFIX
            ) and _task_prebuilt_image({"path": str(destination)}):
                _remove_local_docker_image_pin(destination / "task.toml")
            config = dict(row[HARBOR_TASK_CONFIG_KEY])
            config["path"] = str(destination)
            isolated_configs[source_path] = config
        isolated = dict(row)
        isolated[HARBOR_TASK_CONFIG_KEY] = dict(config)
        output.append(isolated)
    return output


@dataclass(frozen=True, slots=True)
class E2BTemplateRecord:
    task_name: str
    template_name: str
    existed: bool


def parse_e2b_task_memory_overrides(values: Sequence[str]) -> dict[str, int]:
    """Parse repeatable ``TASK=MB`` E2B memory overrides."""

    overrides: dict[str, int] = {}
    for value in values:
        task_name, separator, encoded_memory = value.partition("=")
        task_name = task_name.strip()
        if not separator or not task_name or not encoded_memory.strip():
            raise ValueError(
                f"invalid E2B task memory override {value!r}; expected TASK=MB"
            )
        if task_name in overrides:
            raise ValueError(f"duplicate E2B memory override for task {task_name!r}")
        try:
            memory_mb = int(encoded_memory)
        except ValueError as exc:
            raise ValueError(
                f"invalid E2B memory for task {task_name!r}: {encoded_memory!r}"
            ) from exc
        if memory_mb < 1:
            raise ValueError(
                f"E2B memory for task {task_name!r} must be positive"
            )
        overrides[task_name] = memory_mb
    return overrides


def e2b_trial_config_for_task(
    trial_config: Mapping[str, Any] | str | Path | None,
    *,
    task_name: str,
    task_memory_mb: Mapping[str, int] | None,
) -> dict[str, Any]:
    """Apply one task's E2B memory override without mutating the base config."""

    config = load_harbor_trial_config(trial_config)
    if not task_memory_mb or task_name not in task_memory_mb:
        return config
    configured_environment = config.get("environment")
    if configured_environment is not None and not isinstance(
        configured_environment, Mapping
    ):
        raise TypeError("Harbor trial config 'environment' must be a mapping")
    environment = dict(configured_environment or {})
    environment["override_memory_mb"] = int(task_memory_mb[task_name])
    config["environment"] = environment
    return config


async def prebuild_e2b_templates(
    task_rows: Sequence[Mapping[str, Any]],
    *,
    trials_dir: str | Path,
    agent_import_path: str,
    agent_version: str,
    agent_provider: str,
    context_limit: int,
    output_limit: int,
    trial_config: Any | None = None,
    task_memory_mb: Mapping[str, int] | None = None,
    max_concurrency: int = 8,
    timeout_seconds: float = 1_800.0,
    tool_timeout_seconds: int = DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS,
) -> tuple[E2BTemplateRecord, ...]:
    """Build each content-addressed E2B template exactly once.

    Harbor's E2B environment currently exposes template build primitives on
    the environment object rather than as a public job API. Keeping that
    dependency here prevents benchmark and harness code from reaching into
    E2B internals.
    """

    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if context_limit < 1 or output_limit < 1:
        raise ValueError("context_limit and output_limit must be positive")

    rows = list(task_rows)
    names = [task_name_from_row(dict(row)) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("E2B template prebuild requires unique task rows")

    harbor = _require_harbor()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def has_default_tag(template_name: str) -> bool:
        # E2B can retain an alias after a failed build even though the alias has
        # no usable default image. ``alias_exists`` alone is therefore not a
        # sufficient readiness check before rollout fan-out.
        from e2b import AsyncTemplate

        tags = await AsyncTemplate.get_tags(template_name)
        return any(tag.tag == "default" for tag in tags)

    async def is_launchable(
        template_name: str,
        *,
        expected_cpus: int | None,
        expected_memory_mb: int | None,
    ) -> bool:
        """Verify that an alias launches with the requested resources."""

        from e2b import AsyncSandbox

        sandbox = None
        try:
            sandbox = await AsyncSandbox.create(template=template_name, timeout=60)
            if expected_cpus is not None or expected_memory_mb is not None:
                info = await sandbox.get_info()
                if expected_cpus is not None and info.cpu_count != expected_cpus:
                    return False
                if (
                    expected_memory_mb is not None
                    and info.memory_mb != expected_memory_mb
                ):
                    return False
        except Exception as exc:
            if _is_retryable_e2b_default_tag_not_found(
                exc,
                harbor_environment="e2b",
            ):
                return False
            raise
        finally:
            if sandbox is not None:
                await sandbox.kill()
        return True

    async def build(index: int, row: Mapping[str, Any]) -> E2BTemplateRecord:
        task_name = names[index]
        task_trial_config = e2b_trial_config_for_task(
            trial_config,
            task_name=task_name,
            task_memory_mb=task_memory_mb,
        )
        config = _build_trial_config(
            harbor,
            template=task_trial_config,
            task_config=task_config_from_row(dict(row)),
            run_id=f"e2b-template-{index:03d}-{task_name}",
            trials_dir=trials_dir,
            harbor_environment="e2b",
            sidecar_bundle_path="/tmp/not-used-during-template-prebuild.zip",
            sidecar_launch_spec=json.dumps(
                {
                    "inference_base_url": "https://api.fireworks.ai",
                    "debug_enabled": False,
                }
            ),
            context_limit=context_limit,
            output_limit=output_limit,
            agent_import_path=agent_import_path,
            agent_version=agent_version,
            agent_provider=agent_provider,
            tool_timeout_seconds=tool_timeout_seconds,
        )
        async with semaphore:
            trial = await harbor.Trial.create(config)
            environment = trial.agent_environment
            if _ensure_e2b_resource_identity(environment):
                # ``environment_id`` is cached on construction, so recreate the
                # trial after writing the marker to derive the resource-aware alias.
                trial = await harbor.Trial.create(config)
                environment = trial.agent_environment
            expected_cpus = getattr(environment, "_effective_cpus", None)
            expected_memory_mb = getattr(
                environment,
                "_effective_memory_mb",
                None,
            )
            exists = await environment._does_template_exist()
            ready = (
                exists
                and await has_default_tag(environment._template_name)
                and await is_launchable(
                    environment._template_name,
                    expected_cpus=expected_cpus,
                    expected_memory_mb=expected_memory_mb,
                )
            )
            if not ready:
                await asyncio.wait_for(
                    environment._create_template(),
                    timeout=timeout_seconds,
                )
                if (
                    not await environment._does_template_exist()
                    or not await has_default_tag(environment._template_name)
                    or not await is_launchable(
                        environment._template_name,
                        expected_cpus=expected_cpus,
                        expected_memory_mb=expected_memory_mb,
                    )
                ):
                    raise RuntimeError(
                        "E2B template build returned without a launchable default tag "
                        f"for {task_name}"
                    )
            return E2BTemplateRecord(
                task_name=task_name,
                template_name=str(environment._template_name),
                existed=bool(exists),
            )

    results = await asyncio.gather(
        *(build(index, row) for index, row in enumerate(rows))
    )
    return tuple(results)


__all__ = [
    "e2b_trial_config_for_task",
    "E2BTemplateRecord",
    "isolate_e2b_task_rows",
    "parse_e2b_task_memory_overrides",
    "prebuild_e2b_templates",
]
