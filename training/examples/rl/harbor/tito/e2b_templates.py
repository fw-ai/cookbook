"""Prebuild Harbor E2B templates once before rollout fan-out."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.examples.rl.harbor.tito.trial import (
    DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS,
    _build_trial_config,
    _require_harbor,
    task_config_from_row,
    task_name_from_row,
)


@dataclass(frozen=True, slots=True)
class E2BTemplateRecord:
    task_name: str
    template_name: str
    existed: bool


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

    async def build(index: int, row: Mapping[str, Any]) -> E2BTemplateRecord:
        task_name = names[index]
        config = _build_trial_config(
            harbor,
            template=trial_config,
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
            exists = await environment._does_template_exist()
            if not exists:
                await asyncio.wait_for(
                    environment._create_template(),
                    timeout=timeout_seconds,
                )
                if not await environment._does_template_exist():
                    raise RuntimeError(
                        f"E2B template build returned without an alias for {task_name}"
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


__all__ = ["E2BTemplateRecord", "prebuild_e2b_templates"]
