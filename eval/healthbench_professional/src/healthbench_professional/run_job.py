"""Resolve and launch the pinned Harbor HealthBench Professional job safely."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import sysconfig
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from harbor.models.job.config import JobConfig

from .adapter import (
    DATASET_ID,
    DATASET_REVISION,
    EXPECTED_ROWS,
    _task_directory_name,
)

DEFAULT_MODEL = "accounts/fireworks/models/kimi-k2p6"
DEFAULT_ATTEMPTS = 8
DEFAULT_CONCURRENCY = 8

_SOURCE_JOB_PATH = Path(__file__).resolve().parents[2] / "job.yaml"
_INSTALLED_JOB_PATH = (
    Path(sysconfig.get_path("data")) / "share" / "healthbench-professional" / "job.yaml"
)
DEFAULT_JOB_PATH = (
    _SOURCE_JOB_PATH if _SOURCE_JOB_PATH.is_file() else _INSTALLED_JOB_PATH
)

_JOB_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def default_job_name() -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d__%H-%M-%S")
    return f"healthbench-professional-{timestamp}"


def _load_template(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"HealthBench Harbor job template not found: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("HealthBench Harbor job template must contain an object")
    return value


def _prepared_tasks(dataset_dir: Path) -> list[tuple[int, str]]:
    tasks: list[tuple[int, str]] = []
    seen_indices: set[int] = set()
    for task_dir in dataset_dir.iterdir() if dataset_dir.is_dir() else ():
        if not task_dir.is_dir() or not (task_dir / "task.toml").is_file():
            continue
        if not task_dir.name.startswith("healthbench_professional_"):
            raise ValueError(
                f"Prepared dataset contains unexpected Harbor task: {task_dir}"
            )
        source_path = task_dir / "tests" / "source.json"
        try:
            source = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Invalid prepared HealthBench task: {source_path}"
            ) from exc
        dataset_id = source.get("dataset_id")
        if (
            not isinstance(dataset_id, str)
            or not dataset_id
            or task_dir.name != _task_directory_name(dataset_id)
        ):
            raise ValueError(
                f"Prepared task directory does not match dataset_id: {source_path}"
            )
        if source.get("dataset_repo") != DATASET_ID:
            raise ValueError(
                f"Prepared task has unexpected dataset repository: {source_path}"
            )
        index = source.get("dataset_index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < EXPECTED_ROWS
        ):
            raise ValueError(f"Prepared task has invalid dataset_index: {source_path}")
        if source.get("dataset_revision") != DATASET_REVISION:
            raise ValueError(
                f"Prepared task has unexpected dataset revision: {source_path}"
            )
        if index in seen_indices:
            raise ValueError(
                f"Prepared dataset contains duplicate dataset_index {index}"
            )
        seen_indices.add(index)
        tasks.append((index, task_dir.name))
    return sorted(tasks)


def resolve_job_config(
    *,
    model: str = DEFAULT_MODEL,
    dataset_dir: Path,
    jobs_dir: Path = Path("jobs"),
    job_name: str = "healthbench-professional",
    concurrency: int = DEFAULT_CONCURRENCY,
    limit: int | None = None,
    quiet: bool = False,
    template_path: Path = DEFAULT_JOB_PATH,
) -> dict[str, Any]:
    """Return a validated config without Harbor's lossy config-mode overrides."""

    if not (
        model.startswith("accounts/")
        and ("/models/" in model or "/deployments/" in model)
    ):
        raise ValueError("model must be a full Fireworks model or deployment path")
    if not _JOB_NAME_RE.fullmatch(job_name):
        raise ValueError(
            "job_name must contain only letters, numbers, dots, underscores, or hyphens"
        )
    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or concurrency < 1
    ):
        raise ValueError("concurrency must be a positive integer")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("limit must be a positive integer")
    if limit is not None and limit > EXPECTED_ROWS:
        raise ValueError(f"limit cannot exceed {EXPECTED_ROWS}")

    prepared = _prepared_tasks(dataset_dir)
    if limit is None and len(prepared) != EXPECTED_ROWS:
        raise ValueError(
            f"full HealthBench run requires exactly {EXPECTED_ROWS} prepared tasks; "
            f"found {len(prepared)}"
        )
    if limit is not None and len(prepared) < limit:
        raise ValueError(
            f"prepared dataset requires at least {limit} tasks; found {len(prepared)} "
            f"in {dataset_dir}"
        )

    config = _load_template(template_path)
    agents = config.get("agents")
    datasets = config.get("datasets")
    if not isinstance(agents, list) or len(agents) != 1:
        raise ValueError("HealthBench job template must define exactly one agent")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise ValueError("HealthBench job template must define exactly one dataset")

    config["job_name"] = job_name
    config["jobs_dir"] = str(jobs_dir.expanduser().resolve())
    config["n_attempts"] = DEFAULT_ATTEMPTS
    config["n_concurrent_trials"] = concurrency
    config["quiet"] = quiet
    agents[0]["model_name"] = model
    datasets[0]["path"] = str(dataset_dir.expanduser().resolve())
    selected = prepared if limit is None else prepared[:limit]
    # Never let Harbor rediscover the dataset directory. Explicitly naming the
    # validated tasks keeps unrelated or stale task directories out of both
    # full and limited benchmark runs.
    datasets[0]["task_names"] = [name for _, name in selected]
    datasets[0]["n_tasks"] = len(selected)

    # Round-trip through Harbor's public schema here, before model/judge calls.
    validated = JobConfig.model_validate(config)
    return validated.model_dump(mode="json", exclude_none=True)


def write_resolved_job(config: dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def run_harbor(config_path: Path) -> int:
    command = [
        sys.executable,
        "-m",
        "harbor.cli.main",
        "run",
        "--config",
        str(config_path),
        "--yes",
    ]
    return subprocess.run(command, check=False).returncode


def resolve_and_run(**kwargs: Any) -> int:
    config = resolve_job_config(**kwargs)
    with tempfile.TemporaryDirectory(prefix="healthbench-professional-") as tmp:
        config_path = write_resolved_job(config, Path(tmp) / "job.json")
        return run_harbor(config_path)


__all__ = [
    "DEFAULT_ATTEMPTS",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_JOB_PATH",
    "DEFAULT_MODEL",
    "default_job_name",
    "resolve_and_run",
    "resolve_job_config",
    "run_harbor",
    "write_resolved_job",
]
