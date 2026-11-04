"""WandB, MLflow, and metrics logging utilities."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from numbers import Integral, Real
from typing import Any

from training.utils.config import MLflowConfig, WandBConfig
from training.utils import fileio
from training.utils.phase_tracing import flush_phase_trace
from training.utils.runner import WandbConfigError

logger = logging.getLogger(__name__)

_MetricSteps = dict[str, str | None]
_METRIC_LOCK = threading.Lock()

_DEFAULT_WANDB_METRIC_STEPS: _MetricSteps = {
    "train/step": None,
    "train/*": "train/step",
    "perf/*": "train/step",
    "rollout/*": "train/step",
    "batch/*": "train/step",
    "infra/*": "train/step",
    "ctx/*": "train/step",
}

ASYNC_RL_WANDB_METRIC_STEPS: _MetricSteps = {
    "train/step": None,
    "train/*": "train/step",
    "rollout/step": None,
    "rollout/*": "rollout/step",
    "perf/*": "rollout/step",
    "infra/*": "rollout/step",
    "ctx/*": "rollout/step",
    "batch/*": "rollout/step",
    "async/*": "rollout/step",
    "eval/*": "rollout/step",
    "checkpoint/*": "rollout/step",
    "producer/event": None,
    "producer/*": "producer/event",
}


def _define_wandb_metrics(wandb_module: Any, metric_steps: _MetricSteps) -> None:
    for metric, step_metric in metric_steps.items():
        if step_metric is None:
            wandb_module.define_metric(metric)
        else:
            wandb_module.define_metric(metric, step_metric=step_metric)


def _is_wandb_auth_error(exc: BaseException) -> bool:
    """Heuristically classify a W&B exception as an auth/config error."""
    try:
        import wandb

        auth_types = tuple(
            t
            for t in (
                getattr(wandb.errors, "AuthenticationError", None),
                getattr(wandb.errors, "UsageError", None),
            )
            if isinstance(t, type)
        )
        if auth_types and isinstance(exc, auth_types):
            return True
    except Exception:
        pass

    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "401",
            "unauthorized",
            "permission",
            "api key",
            "api_key",
            "authentication",
        )
    )


def setup_wandb(
    wb: WandBConfig,
    config: dict[str, Any],
    *,
    metric_steps: _MetricSteps | None = None,
) -> bool:
    """Initialize WandB if enabled. Returns True if active.

    W&B is enabled when ``wb.entity`` is set or ``wb.mode`` is ``"offline"``.
    Offline mode logs the run locally without an entity or API key; sync it
    later with ``wandb sync``. In online mode, if ``WANDB_API_KEY`` is not
    set, falls back to offline mode so runs are logged locally without
    requiring authentication. Auth/permission/config failures from
    ``wandb.init`` are re-raised as :class:`WandbConfigError` so the
    orchestration layer surfaces them as a user-actionable error
    (FIR2-1774).
    """
    offline = wb.mode == "offline"
    if not wb.entity and not offline:
        return False
    try:
        import os
        import wandb
    except ImportError:
        logger.warning("wandb not installed; metrics will only be logged to console")
        return False

    if not offline and not os.environ.get("WANDB_API_KEY"):
        logger.warning(
            "WANDB_API_KEY not set; running wandb in offline mode. "
            "Set the key to sync runs to the dashboard."
        )
        os.environ["WANDB_MODE"] = "offline"

    try:
        wandb.init(
            entity=wb.entity,
            project=wb.project,
            name=wb.run_name,
            config=config,
            mode="offline" if offline else None,
        )
    except Exception as exc:
        if _is_wandb_auth_error(exc):
            raise WandbConfigError(
                f"Weights & Biases authentication/configuration failed: {exc}. "
                "Check your W&B API key, entity, and project."
            ) from exc
        raise
    if wandb.run is not None:
        _define_wandb_metrics(wandb, metric_steps or _DEFAULT_WANDB_METRIC_STEPS)
        logger.info("WandB: %s", wandb.run.url)
    return True


def setup_mlflow(mf: MLflowConfig, config: dict[str, Any]) -> bool:
    """Initialize MLflow if enabled. Returns True if active.

    MLflow is enabled when ``mf.tracking_uri`` or ``mf.experiment_name`` is
    set. Without a tracking URI, runs log to the local ``./mlruns`` directory
    (or the ``MLFLOW_TRACKING_URI`` environment variable). Flat scalar entries
    of ``config`` are logged as run params; param failures are warned and
    swallowed so a tracking-server hiccup never blocks training.
    """
    if not mf.tracking_uri and not mf.experiment_name:
        return False
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed; metrics will not be logged to MLflow")
        return False

    if mf.tracking_uri:
        mlflow.set_tracking_uri(mf.tracking_uri)
    if mf.experiment_name:
        mlflow.set_experiment(mf.experiment_name)
    mlflow.start_run(run_name=mf.run_name)

    params = {
        str(key): str(value)
        for key, value in config.items()
        if isinstance(value, (str, int, float, bool))
    }
    if params:
        try:
            mlflow.log_params(params)
        except Exception:
            logger.warning("Failed to log config params to MLflow", exc_info=True)
    run = mlflow.active_run()
    if run is not None:
        logger.info("MLflow run id: %s", run.info.run_id)
    return True


def wandb_log(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log metrics to WandB if available.

    When ``step`` is ``None`` we let wandb auto-increment its internal
    step counter; metrics with a declared ``step_metric`` (via
    ``define_metric``) read their x-axis from the dict instead.  Passing
    an explicit ``step`` is still supported for single-axis recipes that
    want to peg the global step manually.
    """
    try:
        import wandb

        if wandb.run is not None:
            if step is None:
                wandb.log(metrics, commit=True)
            else:
                wandb.log(metrics, step=step, commit=True)
    except ImportError:
        pass


def mlflow_log(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log numeric metrics to the active MLflow run, if any.

    MLflow metrics must be finite numbers; non-numeric and non-finite values
    in ``metrics`` are skipped. MLflow natively accepts the business ``step``
    per call, so no ``define_metric``-style axis tables are needed and
    repeated steps stay distinct data points. Log failures are warned and
    swallowed: MLflow is a mirror, the JSONL ledger is the canonical record.
    """
    try:
        import mlflow
    except ImportError:
        return
    if mlflow.active_run() is None:
        return
    numeric = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }
    if not numeric:
        return
    try:
        mlflow.log_metrics(numeric, step=step)
    except Exception:
        logger.warning("Failed to log metrics to MLflow", exc_info=True)


def _normalize_metric_value(value: Any) -> Any:
    """Return a finite, JSON-compatible copy of one metric value."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): _normalize_metric_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_metric_value(item) for item in value]

    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            return _normalize_metric_value(to_list())
        except Exception:
            pass

    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def _normalize_metrics(
    metrics: dict[str, Any],
    step: int,
) -> dict[str, Any]:
    """Build the plain metric dictionary consumed by every sink."""
    normalized_step = _normalize_metric_value(step)
    if isinstance(normalized_step, bool) or not isinstance(
        normalized_step, (int, float)
    ):
        raise ValueError("metric step must be a finite number")

    record = _normalize_metric_value(metrics)
    record["step"] = normalized_step
    return record


def log_metrics(
    metrics: dict[str, Any],
    *,
    step: int,
    metrics_file: str | None = None,
) -> None:
    """Normalize and send one plain metric dictionary to JSONL, W&B, and MLflow.

    ``COOKBOOK_METRICS_FILE`` enables the append-only canonical JSONL ledger.
    An explicit ``metrics_file`` overrides the environment for callers that
    manage multiple artifacts in one process. A configured ledger is required:
    write failures propagate so callers never mistake an incomplete artifact
    for a complete metric stream.

    All sinks receive the same dictionary in the same order. The producer must
    pass the business ``step`` explicitly; the logger never infers it from
    metric names. Values are recursively converted to finite JSON-compatible
    data and the canonical ``step`` field is added to that plain dictionary.

    The business step is deliberately never passed as W&B's transport
    ``_step``. W&B auto-increments its transport step once per log call, while
    ``define_metric`` maps dashboards to ``train/step`` or ``rollout/step``.
    JSONL line order and W&B's implicit transport step preserve distinct log
    calls that share the same business step.
    """
    path = (
        metrics_file
        if metrics_file is not None
        else os.environ.get("COOKBOOK_METRICS_FILE")
    )

    # Sync and async recipes can emit from different threads. Serialize record
    # creation and delivery so line/call order is identical in every sink.
    with _METRIC_LOCK:
        record = _normalize_metrics(metrics, step)
        if path:
            fileio.append_jsonl(path, record)

        # The exact JSONL record is the W&B payload. Its transport axis is
        # intentionally implicit and unique even when business steps repeat.
        wandb_log(record)
        # MLflow mirrors the same record with the business step passed
        # explicitly; it natively supports per-call steps.
        mlflow_log(record, step=step)


_RUN_RESOURCE_ATTRIBUTES = (
    "trainer_job_id",
    "reference_trainer_job_id",
    "deployment_id",
)


def log_run_resources(service: Any) -> dict[str, str]:
    """Record the run's trainer and deployment ids in the log and W&B config.

    These ids are the join keys between client metrics in W&B or the metrics
    JSONL and the trainer/deployment server metrics labeled ``trainer_id`` and
    ``deployment_id``. Call it once after the policy trainer and deployment
    exist. Ids the service does not expose are omitted.
    """
    resources: dict[str, str] = {}
    for name in _RUN_RESOURCE_ATTRIBUTES:
        try:
            value = getattr(service, name, None)
        except RuntimeError:
            # The SDK raises for ids a service never resolved.
            value = None
        if isinstance(value, str) and value:
            resources[name] = value
    if not resources:
        return resources
    logger.info(
        "Run resources: %s",
        " ".join(f"{name}={value}" for name, value in resources.items()),
    )
    try:
        import wandb

        if wandb.run is not None:
            wandb.config.update(resources, allow_val_change=True)
    except ImportError:
        pass
    return resources


def wandb_finish(*, metrics_file: str | None = None) -> None:
    """Flush tracing, optionally attach local artifacts, then finish W&B."""
    try:
        flush_phase_trace()
    except Exception:
        logger.warning(
            "Failed to flush client phase trace",
            exc_info=True,
        )
    try:
        import wandb

        if wandb.run is not None:
            if metrics_file and os.path.isfile(metrics_file):
                try:
                    artifact = wandb.Artifact(
                        f"{wandb.run.id}-metrics",
                        type="metrics",
                    )
                    artifact.add_file(metrics_file, name="metrics.jsonl")
                    wandb.run.log_artifact(artifact)
                except Exception:
                    logger.warning(
                        "Failed to attach canonical metrics to W&B",
                        exc_info=True,
                    )
            wandb.finish()
    except ImportError:
        pass


def mlflow_finish(*, metrics_file: str | None = None) -> None:
    """Attach the metrics ledger and end the active MLflow run, if any."""
    try:
        import mlflow
    except ImportError:
        return
    if mlflow.active_run() is None:
        return
    if metrics_file and os.path.isfile(metrics_file):
        try:
            mlflow.log_artifact(metrics_file, artifact_path="metrics")
        except Exception:
            logger.warning(
                "Failed to attach canonical metrics to MLflow",
                exc_info=True,
            )
    mlflow.end_run()


def log_metrics_json(step: int, **kwargs: float) -> None:
    """Print a JSON metrics line to stdout (for structured log parsing)."""
    print(json.dumps({"type": "metrics", "step": step, **kwargs}), flush=True)
