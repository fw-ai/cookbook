"""Harness-neutral adapter from TITO rollouts to Harbor tasks and trials."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib
import json
import logging
import math
import re
import shutil
import sys
import tempfile
import tomllib
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from training.examples.rl.harbor.tito.sidecar import (
    SIDECAR_ARTIFACT_MANIFEST_PATH,
    SIDECAR_ARTIFACT_PATH,
    SIDECAR_COMPLETE_PATH,
    SIDECAR_DEBUG_ROOT,
    SIDECAR_STDERR_PATH,
    SIDECAR_STDOUT_PATH,
)
from training.utils.rl.async_rl.errors import RecoverableRolloutError

HARBOR_TASK_CONFIG_KEY = "harbor_task_config"
DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS = 900
DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS = frozenset(
    {
        "EnvironmentStartTimeoutError",
        "ApiOverloadedError",
        "ApiInternalServerError",
        "ApiConnectionClosedError",
        "ApiRateLimitError",
        "ApiResponseStalledError",
        "UnknownApiError",
    }
)
_TERMINAL_EXCEPTION_TYPES = frozenset(
    {
        "AgentTimeoutError",
        "NonZeroAgentExitCodeError",
        "VerifierTimeoutError",
    }
)

logger = logging.getLogger(__name__)
_COMPACT_ARTIFACT_DESTINATION = Path("tito/compact")
_DEBUG_ARTIFACT_DESTINATION = Path("tito/debug")
_LOG_ARTIFACT_DESTINATION = Path("tito/logs")


def validate_harbor_retry_exceptions(names: Any) -> frozenset[str]:
    """Validate the pinned Harbor set plus observed provider-create types."""

    normalized = frozenset(str(name) for name in names)
    if not normalized or "" in normalized:
        raise ValueError("Harbor retry include_exceptions must not be empty")
    missing_required = sorted(DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS - normalized)
    if missing_required:
        raise ValueError(
            "Harbor retry include_exceptions is missing required exception types: "
            + ", ".join(missing_required)
        )

    # Harbor exposes the base retry contract as serialized exception names on a
    # trial result rather than importable classes. Extra provider-create errors
    # are accepted only when their installed dependency exports the type.
    known = set(DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS)
    for module_name in (
        "harbor.trial.errors",
        "harbor.agents.installed.base",
        "harbor.environments.e2b",
        "e2b.exceptions",
        "httpcore",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        known.update(
            name for name, value in vars(module).items() if isinstance(value, type)
        )
    missing = sorted(normalized - known)
    if missing:
        raise ValueError(
            "Harbor retry include_exceptions contains unknown exception types: "
            + ", ".join(missing)
        )
    return normalized


def _raise_trial_execution_failure(
    stage: str,
    error: BaseException,
    retry_include_exceptions: frozenset[str],
) -> None:
    detail = f"{type(error).__name__}: {error}"
    if type(error).__name__ in retry_include_exceptions:
        raise RecoverableRolloutError(
            f"Harbor failed {stage} with a retryable error: {detail}"
        ) from error
    raise RuntimeError(
        f"Harbor failed {stage} with a non-retryable error: {detail}"
    ) from error


def _is_retryable_e2b_stream_open_timeout(
    exception: Any,
    *,
    harbor_environment: str,
) -> bool:
    """Recognize the provider failure that proves no command stream opened.

    Harbor serializes trial failures to names, messages, and tracebacks.  E2B's
    generic ``TimeoutException`` name is too broad for the public retry
    allowlist, while agent setup may wrap the same provider error in
    ``RuntimeError``.  Require both the E2B backend and the exact first-event
    timeout evidence from the retained traceback.
    """

    if harbor_environment != "e2b" or exception is None:
        return False
    traceback = str(getattr(exception, "exception_traceback", "") or "")
    timeout_marker = (
        "Request timed out: the stream didn't open within 'request_timeout'"
    )
    if timeout_marker not in traceback:
        return False
    provider_trace = (
        "site-packages/e2b/" in traceback
        and "harbor/environments/e2b.py" in traceback
        and re.search(
            rf"(?m)^e2b\.exceptions\.TimeoutException: {re.escape(timeout_marker)}",
            traceback,
        )
        is not None
    )
    cleanup_trace = re.search(
        rf"(?m)^TITO sidecar failure cleanup failed: {re.escape(timeout_marker)}",
        traceback,
    )
    return provider_trace or cleanup_trace is not None


def _is_retryable_e2b_sidecar_readiness_timeout(
    exception: Any,
    *,
    harbor_environment: str,
) -> bool:
    """Recognize Harbor's wrapper around a live E2B sidecar readiness timeout."""

    if harbor_environment != "e2b" or exception is None:
        return False
    traceback = str(getattr(exception, "exception_traceback", "") or "")
    exception_type = str(getattr(exception, "exception_type", "") or "")
    exception_message = str(getattr(exception, "exception_message", "") or "")
    marker = r"TITO sidecar did not become ready within [0-9]+(?:\.[0-9]+)?s"
    inner_timeout = (
        "training/examples/rl/harbor/tito/sidecar.py" in traceback
        and re.search(rf"(?m)^TimeoutError: {marker}$", traceback) is not None
    )
    if not inner_timeout:
        return False
    explicit_wrapper = (
        re.search(rf"(?m)^RuntimeError: Agent install failed: {marker}$", traceback)
        is not None
    )
    harbor_wrapper = (
        exception_type == "RuntimeError" and exception_message == "Agent install failed"
    )
    return explicit_wrapper or harbor_wrapper


def _redact_sidecar_spec(result_path: Path) -> None:
    """Remove the inference credential-bearing launch spec from the result."""
    if not result_path.is_file():
        return
    document = json.loads(result_path.read_text(encoding="utf-8"))
    kwargs = document.get("config", {}).get("agent", {}).get("kwargs", {})
    if "sidecar_launch_spec" not in kwargs:
        return
    kwargs["sidecar_launch_spec"] = "<redacted>"
    temporary_path = result_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(result_path)


def _redact_trial_artifacts(trial_path: Path, inference_key: str) -> None:
    """Scrub the inference credential from every retained Harbor artifact."""
    if not trial_path.exists():
        return
    encoded_key = inference_key.encode("utf-8")
    for path in sorted(trial_path.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        encoded = path.read_bytes()
        if encoded_key not in encoded:
            continue
        try:
            decoded = encoded.decode("utf-8")
        except UnicodeDecodeError:
            # Generated evidence containing a credential but not safely
            # rewritable as text is unsafe to retain.
            path.unlink()
            continue
        temporary = path.with_name(f".{path.name}.redacting")
        temporary.write_text(
            decoded.replace(inference_key, "<redacted>"), encoding="utf-8"
        )
        temporary.chmod(path.stat().st_mode & 0o777)
        temporary.replace(path)
    remaining = [
        str(path)
        for path in trial_path.rglob("*")
        if not path.is_symlink() and path.is_file() and encoded_key in path.read_bytes()
    ]
    if remaining:
        # Delete only the generated trial evidence in scope; retaining a live
        # credential is not an acceptable debug fallback.
        shutil.rmtree(trial_path)
        raise RuntimeError(
            "could not scrub the TITO sidecar inference credential from trial artifacts: "
            + ", ".join(remaining)
        )


def _validate_tool_timeout_below_trial(
    trial: Any,
    *,
    tool_timeout_seconds: int,
) -> None:
    """Fail before agent execution when a local tool can consume the trial cap."""

    missing = object()
    outer_timeout = getattr(trial, "_agent_timeout_sec", missing)
    if outer_timeout is missing:
        # Harbor is deliberately pinned, so a lifecycle-field change must be
        # characterized instead of silently weakening process-tree cleanup.
        raise RuntimeError(
            "pinned Harbor no longer exposes its resolved agent timeout; "
            "re-characterize the harness timeout contract"
        )
    if outer_timeout is not None and tool_timeout_seconds >= float(outer_timeout):
        raise ValueError(
            "TITO Harbor tool timeout must be below the resolved outer agent "
            f"timeout ({tool_timeout_seconds}s >= {float(outer_timeout):g}s)"
        )


def _require_harbor() -> Any:
    if sys.version_info < (3, 12):
        raise RuntimeError("Harbor RL requires Python 3.12 or newer")
    try:
        # lazy: Harbor is an example-only dependency installed by the user.
        import harbor
    except ImportError as exc:
        raise RuntimeError(
            "Harbor RL dependencies are missing; install the production-pinned "
            "`harbor==0.21.0`"
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
    rows = [_task_config_to_row(config) for config in task_configs]
    if task_names is None:
        return rows
    if len(task_names) != len(set(task_names)):
        raise ValueError("Harbor task names must be unique")
    rows_by_name = {str(row["task_name"]): row for row in rows}
    missing = [name for name in task_names if name not in rows_by_name]
    if missing:
        raise ValueError(f"Harbor dataset is missing requested tasks: {missing}")
    return [rows_by_name[name] for name in task_names]


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


def _task_local_path(task_config: Any) -> Path | None:
    get_local_path = getattr(task_config, "get_local_path", None)
    if callable(get_local_path):
        return Path(get_local_path()).expanduser().resolve()
    if isinstance(task_config, Mapping) and task_config.get("path"):
        return Path(str(task_config["path"])).expanduser().resolve()
    return None


def task_initial_instruction(task_config: Any) -> str:
    """Read the exact local instruction Harbor will provide to the agent."""
    task_path = _task_local_path(task_config)
    if task_path is None:
        raise ValueError("TITO benchmark task needs a resolved local path")
    instruction_path = task_path / "instruction.md"
    if not instruction_path.is_file():
        raise ValueError(f"Harbor task instruction is missing: {instruction_path}")
    return instruction_path.read_text(encoding="utf-8")


def task_name_from_row(row: Mapping[str, Any]) -> str:
    name = row.get("task_name")
    return str(name) if name else "unknown-task"


def _safe_trial_name(run_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id).strip("-.")
    return (normalized or "harbor-tito")[-180:]


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


def _task_document(task_config: Any) -> Mapping[str, Any] | None:
    """Read one resolved local task's declarative Harbor configuration."""

    task_path = _task_local_path(task_config)
    if task_path is None:
        return None
    config_path = task_path / "task.toml"
    if not config_path.is_file():
        return None
    return tomllib.loads(config_path.read_text(encoding="utf-8"))


def _task_prebuilt_image(task_config: Any) -> str | None:
    """Return a prepared task's Harbor-native prebuilt image reference."""

    document = _task_document(task_config)
    if document is None:
        return None
    environment = document.get("environment")
    if not isinstance(environment, Mapping):
        return None
    image = environment.get("docker_image")
    if image is None:
        return None
    value = str(image).strip()
    return value or None


def _task_uses_compose(task_config: Any) -> bool:
    task_path = _task_local_path(task_config)
    if task_path is None:
        return False
    environment_dir = task_path / "environment"
    return any(
        (environment_dir / name).is_file()
        for name in ("docker-compose.yaml", "docker-compose.yml")
    )


def _build_trial_config(
    harbor: Any,
    *,
    template: Mapping[str, Any] | None,
    task_config: Any,
    run_id: str,
    trials_dir: str | Path,
    harbor_environment: str,
    sidecar_bundle_path: str | Path,
    sidecar_launch_spec: str,
    context_limit: int = 131072,
    output_limit: int = 8192,
    agent_import_path: str,
    agent_version: str,
    agent_provider: str = "fireworks-rl",
    tool_timeout_seconds: int = DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS,
) -> Any:
    """Merge a native TrialConfig template with Fireworks-owned runtime fields."""

    document = load_harbor_trial_config(template)
    if document.get("install_only"):
        raise ValueError("Harbor RL does not support TrialConfig.install_only")
    if document.get("source_trial") is not None:
        raise ValueError("Harbor RL does not support TrialConfig.source_trial")

    environment = _config_section(document, "environment")
    if environment.get("import_path"):
        raise ValueError("custom Harbor environment import_path is not supported")
    environment.pop("import_path", None)
    if harbor_environment == "docker":
        environment["type"] = harbor.EnvironmentType.DOCKER
    elif harbor_environment == "e2b":
        # Early duplicate of Harbor's capability gate: failing here avoids an
        # E2B template build. Harbor remains the authoritative backstop if its
        # environment capabilities change or the task path is not local.
        if _task_uses_compose(task_config) or environment.get("extra_docker_compose"):
            raise ValueError("Harbor E2B does not support Docker Compose tasks")
        environment["type"] = harbor.EnvironmentType.E2B
    else:
        raise ValueError(f"unsupported Harbor environment {harbor_environment!r}")
    # Remove trial-local containers and volumes after verification. A task's
    # explicitly configured prepared image has its own stable reference and is
    # not the disposable Harbor build tag.
    environment["delete"] = True
    # Docker must rebuild an untagged prepared context. Harbor's E2B backend
    # names templates by the environment-content hash, so force_build would
    # defeat safe reuse and rebuild the same task for every rollout member.
    # A changed context naturally gets a new E2B template name.
    environment["force_build"] = (
        harbor_environment == "docker" and _task_prebuilt_image(task_config) is None
    )

    try:
        sidecar_spec = json.loads(sidecar_launch_spec)
    except json.JSONDecodeError as exc:
        raise ValueError("sidecar launch spec is not valid JSON") from exc
    inference_base_url = str(sidecar_spec.get("inference_base_url") or "")
    parsed_inference_url = urlsplit(inference_base_url)
    if parsed_inference_url.scheme not in {"http", "https"} or not (
        parsed_inference_url.hostname
    ):
        raise ValueError("sidecar launch spec has no valid inference_base_url")

    agent = _config_section(document, "agent")
    agent["name"] = None
    if tool_timeout_seconds < 1:
        raise ValueError("TITO Harbor tool timeout must be positive")
    agent["import_path"] = agent_import_path
    agent["model_name"] = f"{agent_provider}/policy"
    agent["extra_allowed_hosts"] = list(
        dict.fromkeys(
            [
                *agent.get("extra_allowed_hosts", ()),
                parsed_inference_url.hostname,
            ]
        )
    )
    agent["kwargs"] = {
        "sidecar_bundle_path": str(sidecar_bundle_path),
        "sidecar_launch_spec": sidecar_launch_spec,
        "context_limit": int(context_limit),
        "output_limit": int(output_limit),
        "tool_timeout_seconds": int(tool_timeout_seconds),
        "version": agent_version,
    }

    artifacts = list(document.get("artifacts") or ())
    for source in (
        SIDECAR_ARTIFACT_PATH,
        SIDECAR_ARTIFACT_MANIFEST_PATH,
        SIDECAR_COMPLETE_PATH,
    ):
        artifacts.append(
            {
                "source": source,
                "destination": str(_COMPACT_ARTIFACT_DESTINATION / Path(source).name),
            }
        )
    for source in (SIDECAR_STDOUT_PATH, SIDECAR_STDERR_PATH):
        artifacts.append(
            {
                "source": source,
                "destination": str(_LOG_ARTIFACT_DESTINATION / Path(source).name),
            }
        )
    if bool(sidecar_spec.get("debug_enabled")):
        artifacts.append(
            {
                "source": SIDECAR_DEBUG_ROOT,
                "destination": str(_DEBUG_ARTIFACT_DESTINATION),
            }
        )
    document["artifacts"] = artifacts

    document.update(
        task=task_config,
        trial_name=_safe_trial_name(
            f"{run_id}-{hashlib.sha256(sidecar_launch_spec.encode()).hexdigest()[:8]}"
        ),
        trials_dir=Path(trials_dir),
        agent=agent,
        environment=environment,
    )
    return harbor.TrialConfig.model_validate(document)


@dataclass(frozen=True)
class HarborTrialOutcome:
    task_name: str
    trial_name: str
    trial_path: Path
    reward: float | None
    rewards: dict[str, float]
    exception_type: str | None
    exception_message: str | None
    environment_type: str = "docker"
    trajectory_artifact: Any | None = None
    artifact_manifest: Mapping[str, Any] | None = None


def _load_sidecar_artifact(trial_path: Path) -> tuple[Any, dict[str, Any]]:
    """Validate and decode the mandatory sidecar artifact collected by Harbor."""

    artifact_root = trial_path / "artifacts" / _COMPACT_ARTIFACT_DESTINATION
    artifact_path = artifact_root / Path(SIDECAR_ARTIFACT_PATH).name
    manifest_path = artifact_root / Path(SIDECAR_ARTIFACT_MANIFEST_PATH).name
    complete_path = artifact_root / "COMPLETE"
    missing = [
        str(path)
        for path in (artifact_path, manifest_path, complete_path)
        if not path.is_file()
    ]
    if missing:
        raise RecoverableRolloutError(
            "Harbor did not collect the complete TITO sidecar artifact: "
            + ", ".join(missing)
        )
    try:
        encoded = artifact_path.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoverableRolloutError(
            f"could not read the collected TITO sidecar artifact: {exc}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise RecoverableRolloutError("TITO sidecar artifact manifest is not an object")
    if manifest.get("schema_version") != 1:
        raise RecoverableRolloutError("unsupported TITO sidecar artifact manifest")
    if int(manifest.get("bytes") or -1) != len(encoded):
        raise RecoverableRolloutError("TITO sidecar artifact byte count mismatch")
    if manifest.get("sha256") != hashlib.sha256(encoded).hexdigest():
        raise RecoverableRolloutError("TITO sidecar artifact checksum mismatch")
    try:
        from fireworks.training.sdk import TITOTrajectoryArtifact

        artifact = TITOTrajectoryArtifact.unpack(encoded)
    except Exception as exc:
        raise RecoverableRolloutError(
            f"TITO sidecar artifact decoding failed: {type(exc).__name__}: {exc}"
        ) from exc
    if artifact.trajectory_id != manifest.get("trajectory_id"):
        raise RecoverableRolloutError(
            "TITO sidecar artifact trajectory identity mismatch"
        )
    if artifact.status != manifest.get("status"):
        raise RecoverableRolloutError("TITO sidecar artifact terminal status mismatch")
    if artifact.terminal_reason != manifest.get("terminal_reason"):
        raise RecoverableRolloutError("TITO sidecar artifact terminal reason mismatch")
    return artifact, dict(manifest)


async def run_harbor_trial(
    *,
    task_config: Any,
    inference_key: str,
    run_id: str,
    harbor_environment: str,
    sidecar_bundle_path: str | Path,
    sidecar_launch_spec: str,
    trial_config: Any | None = None,
    trials_dir: str | Path | None = None,
    context_limit: int = 131072,
    output_limit: int = 8192,
    agent_import_path: str,
    agent_version: str,
    agent_provider: str = "fireworks-rl",
    reward_key: str = "reward",
    terminal_failure_reward: float | None = None,
    tool_timeout_seconds: int = DEFAULT_HARNESS_TOOL_TIMEOUT_SECONDS,
    retry_include_exceptions: Any = DEFAULT_HARBOR_RETRYABLE_EXCEPTIONS,
) -> HarborTrialOutcome:
    """Run one TITO-backed agent through Harbor's native Trial lifecycle."""

    harbor = _require_harbor()
    reward_key = str(reward_key).strip()
    if not reward_key:
        raise ValueError("Harbor reward key must not be empty")
    retry_names = validate_harbor_retry_exceptions(retry_include_exceptions)
    template = load_harbor_trial_config(trial_config)
    configured_trials_dir = trials_dir or template.get("trials_dir")
    temp_dir = (
        tempfile.TemporaryDirectory(prefix="harbor-tito-")
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
            run_id=run_id,
            trials_dir=trial_root,
            harbor_environment=harbor_environment,
            sidecar_bundle_path=sidecar_bundle_path,
            sidecar_launch_spec=sidecar_launch_spec,
            context_limit=context_limit,
            output_limit=output_limit,
            agent_import_path=agent_import_path,
            agent_provider=agent_provider,
            agent_version=agent_version,
            tool_timeout_seconds=tool_timeout_seconds,
        )
        result = None
        trial_path = trial_root / config.trial_name
        cancellation: asyncio.CancelledError | None = None
        try:
            try:
                trial = await harbor.Trial.create(config)
            except Exception as exc:
                _raise_trial_execution_failure(
                    "before creating a trial", exc, retry_names
                )
            _validate_tool_timeout_below_trial(
                trial,
                tool_timeout_seconds=tool_timeout_seconds,
            )
            try:
                result = await trial.run()
            except Exception as exc:
                _raise_trial_execution_failure(
                    "before producing a trial result", exc, retry_names
                )
        except asyncio.CancelledError as exc:
            cancellation = exc
            raise
        finally:
            try:
                _redact_trial_artifacts(trial_path, inference_key)
                _redact_sidecar_spec(trial_path / "result.json")
            except Exception as exc:
                if cancellation is not None:
                    # A cleanup exception must not convert cancellation into a
                    # retryable rollout. Delete the generated trial as a
                    # credential-safe fallback and preserve cancellation.
                    try:
                        if trial_path.exists():
                            shutil.rmtree(trial_path)
                    except Exception as delete_error:  # noqa: BLE001
                        cancellation.add_note(
                            "Harbor credential scrub and fallback deletion failed: "
                            f"{delete_error}"
                        )
                    else:
                        cancellation.add_note(
                            "Harbor credential scrub failed; generated trial "
                            "artifacts were deleted"
                        )
                    logger.error(
                        "Harbor credential scrub failed during cancellation: %s",
                        exc,
                        exc_info=True,
                    )
                else:
                    raise RecoverableRolloutError(
                        f"Harbor trial credential scrub failed: {exc}"
                    ) from exc

        if result is None:
            raise RecoverableRolloutError(
                "Harbor returned from trial.run() without a trial result"
            )
        if result.trial_name != config.trial_name:
            raise RecoverableRolloutError(
                "Harbor result identity changed during one trial: "
                f"{result.trial_name!r} != {config.trial_name!r}"
            )

        exception = result.exception_info
        exception_type = exception.exception_type if exception is not None else None
        retryable_e2b_timeout = _is_retryable_e2b_stream_open_timeout(
            exception,
            harbor_environment=harbor_environment,
        )
        retryable_sidecar_readiness = _is_retryable_e2b_sidecar_readiness_timeout(
            exception,
            harbor_environment=harbor_environment,
        )
        try:
            trajectory_artifact, artifact_manifest = _load_sidecar_artifact(trial_path)
        except RecoverableRolloutError as exc:
            if retryable_e2b_timeout:
                raise RecoverableRolloutError(
                    "Harbor E2B command stream did not open before its provider "
                    "request timeout"
                ) from exc
            if retryable_sidecar_readiness:
                raise RecoverableRolloutError(
                    "Harbor E2B sidecar did not become ready within its bounded "
                    "startup window"
                ) from exc
            if exception_type and exception_type not in retry_names:
                raise RuntimeError(
                    "Harbor produced no valid TITO artifact after a non-retryable "
                    f"{exception_type}: {exc}"
                ) from exc
            raise
        if retryable_e2b_timeout:
            raise RecoverableRolloutError(
                "Harbor E2B command stream did not open before its provider "
                "request timeout"
            )

        verifier_result = result.verifier_result
        raw_rewards = verifier_result.rewards if verifier_result is not None else None
        context_budget_exhausted = (
            trajectory_artifact.status == "failed"
            and trajectory_artifact.terminal_reason == "context_budget_exhausted"
        )
        if context_budget_exhausted:
            if exception is None:
                raise RecoverableRolloutError(
                    "context budget exhaustion has no failed agent process"
                )
            if exception_type != "NonZeroAgentExitCodeError":
                logger.warning(
                    "Harbor misclassified context-budget exhaustion as %s; "
                    "using the explicit sidecar marker as the terminal authority",
                    exception_type,
                )
            exception_type = "NonZeroAgentExitCodeError"
        if not raw_rewards or reward_key not in raw_rewards:
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
                raw_rewards = {
                    **dict(raw_rewards or {}),
                    reward_key: float(terminal_failure_reward),
                }
            elif exception_type in retry_names:
                raise RecoverableRolloutError(
                    f"Harbor trial {result.trial_name!r} ended with retryable "
                    f"{exception_type}: {exception.exception_message}"
                )
            elif exception_type is not None:
                logger.warning(
                    "Harbor trial %r ended without reward after non-retryable %s; "
                    "retaining its exact trajectory artifact",
                    result.trial_name,
                    exception_type,
                )
                raw_rewards = {}
            else:
                raise RecoverableRolloutError(
                    f"Harbor trial {result.trial_name!r} did not produce a usable "
                    "reward (verifier produced no reward)"
                )

        try:
            rewards = {str(key): float(value) for key, value in raw_rewards.items()}
        except (TypeError, ValueError) as exc:
            raise RecoverableRolloutError(
                f"Harbor trial {result.trial_name!r} produced non-numeric rewards"
            ) from exc
        reward = rewards.get(reward_key)
        if reward is not None and not math.isfinite(reward):
            raise RecoverableRolloutError(
                f"Harbor trial {result.trial_name!r} produced a non-finite reward"
            )

        return HarborTrialOutcome(
            task_name=result.task_name,
            trial_name=result.trial_name,
            trial_path=trial_path,
            reward=reward,
            rewards=rewards,
            exception_type=exception_type,
            exception_message=exception.exception_message if exception else None,
            environment_type=str(config.environment.type.value),
            trajectory_artifact=trajectory_artifact,
            artifact_manifest=artifact_manifest,
        )
