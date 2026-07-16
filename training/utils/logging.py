"""WandB and metrics logging utilities."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from math import comb
from numbers import Integral, Real
from typing import Any

from training.utils.config import WandBConfig
from training.utils import fileio
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
    "pipeline/*": "rollout/step",
    "version/*": "rollout/step",
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
    """Initialize WandB if entity is provided. Returns True if active.

    If ``WANDB_API_KEY`` is not set, falls back to offline mode so runs
    are logged locally without requiring authentication. Auth/permission/config
    failures from ``wandb.init`` are re-raised as :class:`WandbConfigError` so
    the orchestration layer surfaces them as a user-actionable error
    (FIR2-1774).
    """
    if not wb.entity:
        return False
    try:
        import os
        import wandb
    except ImportError:
        logger.warning("wandb not installed; metrics will only be logged to console")
        return False

    if not os.environ.get("WANDB_API_KEY"):
        logger.warning(
            "WANDB_API_KEY not set; running wandb in offline mode. "
            "Set the key to sync runs to the dashboard."
        )
        os.environ["WANDB_MODE"] = "offline"

    try:
        wandb.init(entity=wb.entity, project=wb.project, name=wb.run_name, config=config)
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


def compute_pass_at_k(
    reward_groups: list[list[float]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute pass@k from grouped reward samples.

    Each inner list contains per-sample rewards for one prompt group.
    A sample is "correct" if its reward > 0.5.
    """
    if k_values is None:
        k_values = [1]
    results: dict[str, float] = {}
    for k in k_values:
        pass_rates: list[float] = []
        for group_rewards in reward_groups:
            n = len(group_rewards)
            c = sum(1 for r in group_rewards if r > 0.5)
            if n < k:
                continue
            denom = comb(n, k)
            pass_rates.append(1.0 - comb(n - c, k) / denom if denom > 0 else 0.0)
        if pass_rates:
            results[f"rollout/pass@{k}"] = sum(pass_rates) / len(pass_rates)
    return results


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
    if isinstance(normalized_step, bool) or not isinstance(normalized_step, (int, float)):
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
    """Normalize and send one plain metric dictionary to JSONL and W&B.

    ``COOKBOOK_METRICS_FILE`` enables the append-only canonical JSONL ledger.
    An explicit ``metrics_file`` overrides the environment for callers that
    manage multiple artifacts in one process. A configured ledger is required:
    write failures propagate so callers never mistake an incomplete artifact
    for a complete metric stream.

    Both sinks receive the same dictionary in the same order. The producer must
    pass the business ``step`` explicitly; the logger never infers it from
    metric names. Values are recursively converted to finite JSON-compatible
    data and the canonical ``step`` field is added to that plain dictionary.

    The business step is deliberately never passed as W&B's transport
    ``_step``. W&B auto-increments its transport step once per log call, while
    ``define_metric`` maps dashboards to ``train/step`` or ``rollout/step``.
    JSONL line order and W&B's implicit transport step preserve distinct log
    calls that share the same business step.
    """
    path = metrics_file if metrics_file is not None else os.environ.get("COOKBOOK_METRICS_FILE")

    # Sync and async recipes can emit from different threads. Serialize record
    # creation and delivery so line/call order is identical in every sink.
    with _METRIC_LOCK:
        record = _normalize_metrics(metrics, step)
        if path:
            fileio.append_jsonl(path, record)

        # The exact JSONL record is the W&B payload. Its transport axis is
        # intentionally implicit and unique even when business steps repeat.
        wandb_log(record)


def wandb_finish(*, metrics_file: str | None = None) -> None:
    """Optionally attach the canonical ledger, then finish W&B if active."""
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


def log_metrics_json(step: int, **kwargs: float) -> None:
    """Print a JSON metrics line to stdout (for structured log parsing)."""
    print(json.dumps({"type": "metrics", "step": step, **kwargs}), flush=True)
