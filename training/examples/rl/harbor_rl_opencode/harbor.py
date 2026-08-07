"""Thin adapter from Fireworks rollouts to Harbor tasks and trials."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import re
import sys
import tempfile
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from training.utils.rl.async_rl.errors import RecoverableRolloutError

HARBOR_TASK_CONFIG_KEY = "harbor_task_config"
DEFAULT_OPENCODE_VERSION = "1.18.8"
_OPENCODE_IMPORT_PATH = (
    "training.examples.rl.harbor_rl_opencode.opencode:ConfigurableOpenCode"
)
# Keep this wire-format value local so importing dataset helpers does not import
# OpenCode and its optional Harbor dependency.
_POLICY_HOST_PLACEHOLDER = "{host}"
_TERMINAL_EXCEPTION_TYPES = frozenset(
    {
        "AgentTimeoutError",
        "NonZeroAgentExitCodeError",
        "VerifierTimeoutError",
    }
)

logger = logging.getLogger(__name__)


def _redact_policy_key(result_path: Path) -> None:
    """Remove the local policy credential from Harbor's persisted result."""
    if not result_path.is_file():
        return
    document = json.loads(result_path.read_text(encoding="utf-8"))
    kwargs = document.get("config", {}).get("agent", {}).get("kwargs", {})
    if "policy_api_key" not in kwargs:
        return
    kwargs["policy_api_key"] = "<redacted>"
    temporary_path = result_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(result_path)


def _require_harbor() -> Any:
    if sys.version_info < (3, 12):
        raise RuntimeError("Harbor RL requires Python 3.12 or newer")
    try:
        # lazy: Harbor is an example-only dependency installed by the user.
        import harbor
    except ImportError as exc:
        raise RuntimeError(
            "Harbor RL dependencies are missing; install `harbor>=0.20,<0.21`"
        ) from exc
    return harbor


def _split_dataset_spec(dataset: str) -> tuple[str, str | None]:
    name, separator, version = dataset.rpartition("@")
    if separator and name and version:
        return name, version
    return dataset, None


async def load_harbor_rows_async(
    dataset: str | Path,
    *,
    registry_path: str | Path | None = None,
    task_names: list[str] | None = None,
    n_tasks: int | None = None,
) -> list[dict[str, Any]]:
    """Resolve a Harbor task directory or registry dataset into rollout rows."""

    harbor = _require_harbor()
    candidate = Path(dataset).expanduser()
    cached_candidate = Path.home() / ".cache" / "harbor" / "tasks" / str(dataset)
    if not candidate.exists() and cached_candidate.exists():
        candidate = cached_candidate

    if candidate.is_dir() and (candidate / "task.toml").is_file():
        task_configs = [harbor.TrialTaskConfig(path=candidate.resolve())]
        return [_task_config_to_row(config) for config in task_configs]

    if candidate.is_dir():
        dataset_config = harbor.DatasetConfig(
            path=candidate.resolve(),
            task_names=task_names,
            n_tasks=n_tasks,
        )
    else:
        name, version = _split_dataset_spec(str(dataset))
        dataset_config = harbor.DatasetConfig(
            name=name,
            version=version,
            registry_path=Path(registry_path) if registry_path else None,
            task_names=task_names,
            n_tasks=n_tasks,
        )

    task_configs = await dataset_config.get_task_configs()
    return [_task_config_to_row(config) for config in task_configs]


def load_harbor_rows(
    dataset: str | Path,
    *,
    registry_path: str | Path | None = None,
    task_names: list[str] | None = None,
    n_tasks: int | None = None,
) -> list[dict[str, Any]]:
    """Synchronous entry point used by the example training script."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            load_harbor_rows_async(
                dataset,
                registry_path=registry_path,
                task_names=task_names,
                n_tasks=n_tasks,
            )
        )
    raise RuntimeError("Use `await load_harbor_rows_async(...)` inside an event loop")


def _task_config_to_row(task_config: Any) -> dict[str, Any]:
    task_id = task_config.get_task_id()
    return {
        "task_name": task_id.get_name(),
        HARBOR_TASK_CONFIG_KEY: task_config.model_dump(mode="json"),
    }


def task_config_from_row(row: Mapping[str, Any]) -> Any:
    """Rehydrate Harbor's full task source rather than a reduced task shape."""

    harbor = _require_harbor()
    value = row.get(HARBOR_TASK_CONFIG_KEY)
    if not isinstance(value, Mapping):
        raise ValueError(f"Harbor row is missing {HARBOR_TASK_CONFIG_KEY!r}")
    return harbor.TrialTaskConfig.model_validate(dict(value))


def task_name_from_row(row: Mapping[str, Any]) -> str:
    name = row.get("task_name")
    return str(name) if name else "unknown-task"


def _safe_trial_name(run_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id).strip("-.")
    return (normalized or "harbor-opencode")[-180:]


def load_harbor_trial_config(
    source: Mapping[str, Any] | str | Path | None,
) -> dict[str, Any]:
    """Load an optional Harbor ``TrialConfig`` template without reducing it."""

    if source is None:
        return {}
    if isinstance(source, Mapping):
        document = dict(source)
    elif isinstance(source, (str, Path)):
        config_path = Path(source).expanduser()
        with config_path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
    else:
        raise TypeError("Harbor trial config must be a mapping or YAML path")
    if not isinstance(document, Mapping):
        raise ValueError("Harbor trial config must contain a YAML mapping")
    return copy.deepcopy(dict(document))


def _config_section(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Harbor trial config {name!r} must be a mapping")
    return copy.deepcopy(dict(value))


def _build_trial_config(
    harbor: Any,
    *,
    template: Mapping[str, Any] | None,
    task_config: Any,
    policy_key: str,
    run_id: str,
    trials_dir: str | Path,
    policy_port: int,
    context_limit: int = 131072,
    output_limit: int = 8192,
    opencode_version: str = DEFAULT_OPENCODE_VERSION,
) -> Any:
    """Merge a native TrialConfig template with Fireworks-owned runtime fields."""

    document = load_harbor_trial_config(template)
    if document.get("install_only"):
        raise ValueError("Harbor RL does not support TrialConfig.install_only")
    if document.get("source_trial") is not None:
        raise ValueError("Harbor RL does not support TrialConfig.source_trial")

    environment = _config_section(document, "environment")
    environment_type = environment.get("type", "docker")
    environment_type = getattr(environment_type, "value", environment_type)
    if environment_type is None:
        environment_type = "docker"
    if environment_type != "docker" or environment.get("import_path"):
        raise ValueError(
            "Fireworks Harbor RL supports only Harbor's local Docker environment"
        )
    environment.pop("import_path", None)
    environment["type"] = harbor.EnvironmentType.DOCKER
    # Harbor's generated Compose project uses a per-trial local image tag.
    # Remove that tag and its volumes with the container after verification;
    # retaining it would leak one large image tag per rollout.
    environment["delete"] = True
    # Terminal-Bench task.toml files commonly name a prebuilt image. The
    # prepared task tree extends environment/Dockerfile with pinned OpenCode,
    # so build that Dockerfile instead of selecting the original image.
    environment["force_build"] = True

    agent = _config_section(document, "agent")
    agent["name"] = None
    if policy_port < 1:
        raise ValueError("OpenCode Harbor trials require a policy server port")
    agent["import_path"] = _OPENCODE_IMPORT_PATH
    agent["model_name"] = "fireworks-rl/policy"
    agent["kwargs"] = {
        "policy_base_url": (f"http://{_POLICY_HOST_PLACEHOLDER}:{int(policy_port)}/v1"),
        "policy_api_key": policy_key,
        "context_limit": int(context_limit),
        "output_limit": int(output_limit),
        "version": opencode_version,
    }

    document.update(
        task=task_config,
        trial_name=_safe_trial_name(f"{run_id}-{policy_key[:8]}"),
        trials_dir=Path(trials_dir),
        agent=agent,
        environment=environment,
    )
    return harbor.TrialConfig.model_validate(document)


@dataclass(frozen=True)
class HarborTrialOutcome:
    task_name: str
    trial_name: str
    reward: float
    rewards: dict[str, float]
    exception_type: str | None
    exception_message: str | None
    environment_type: str = "docker"


async def run_harbor_trial(
    *,
    task_config: Any,
    policy_key: str,
    run_id: str,
    policy_port: int,
    trial_config: Any | None = None,
    trials_dir: str | Path | None = None,
    context_limit: int = 131072,
    output_limit: int = 8192,
    opencode_version: str = DEFAULT_OPENCODE_VERSION,
    terminal_failure_reward: float | None = None,
) -> HarborTrialOutcome:
    """Run one OpenCode trajectory through Harbor's native Trial lifecycle."""

    harbor = _require_harbor()
    template = load_harbor_trial_config(trial_config)
    configured_trials_dir = trials_dir or template.get("trials_dir")
    temp_dir = (
        tempfile.TemporaryDirectory(prefix="harbor-opencode-")
        if configured_trials_dir is None
        else None
    )
    trial_root = (
        Path(temp_dir.name)
        if temp_dir is not None
        else Path(configured_trials_dir).expanduser()
    )
    cleanup = temp_dir if temp_dir is not None else nullcontext()
    with cleanup:
        config = _build_trial_config(
            harbor,
            template=template,
            task_config=task_config,
            policy_key=policy_key,
            run_id=run_id,
            trials_dir=trial_root,
            policy_port=policy_port,
            context_limit=context_limit,
            output_limit=output_limit,
            opencode_version=opencode_version,
        )
        try:
            trial = await harbor.Trial.create(config)
            result = await trial.run()
        except Exception as exc:
            raise RecoverableRolloutError(
                f"Harbor failed before producing a trial result: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            _redact_policy_key(trial_root / result.trial_name / "result.json")
        except Exception as exc:
            # This is an ephemeral loopback credential, and a completed trial
            # remains valid even when its optional artifact cannot be rewritten.
            logger.warning(
                "Could not redact the local policy key from Harbor trial %r: %s",
                result.trial_name,
                exc,
            )

        verifier_result = result.verifier_result
        raw_rewards = verifier_result.rewards if verifier_result is not None else None
        if not raw_rewards or "reward" not in raw_rewards:
            exception = result.exception_info
            exception_type = exception.exception_type if exception is not None else None
            if (
                exception_type in _TERMINAL_EXCEPTION_TYPES
                and terminal_failure_reward is not None
            ):
                logger.info(
                    "Harbor trial %r ended with %s; recording configured terminal "
                    "reward %s",
                    result.trial_name,
                    exception_type,
                    terminal_failure_reward,
                )
                raw_rewards = {"reward": float(terminal_failure_reward)}
            else:
                detail = "verifier produced no reward"
                if exception is not None:
                    detail = (
                        f"{exception.exception_type}: {exception.exception_message}"
                    )
                raise RecoverableRolloutError(
                    f"Harbor trial {result.trial_name!r} did not produce a usable "
                    f"reward ({detail})"
                )

        try:
            rewards = {str(key): float(value) for key, value in raw_rewards.items()}
        except (TypeError, ValueError) as exc:
            raise RecoverableRolloutError(
                f"Harbor trial {result.trial_name!r} produced non-numeric rewards"
            ) from exc
        reward = rewards["reward"]
        if not math.isfinite(reward):
            raise RecoverableRolloutError(
                f"Harbor trial {result.trial_name!r} produced a non-finite reward"
            )

        exception = result.exception_info
        return HarborTrialOutcome(
            task_name=result.task_name,
            trial_name=result.trial_name,
            reward=reward,
            rewards=rewards,
            exception_type=exception.exception_type if exception else None,
            exception_message=exception.exception_message if exception else None,
            environment_type=str(config.environment.type.value),
        )
