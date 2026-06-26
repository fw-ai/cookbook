"""WandB and metrics logging utilities."""

from __future__ import annotations

import json
import logging
from math import comb
from typing import Any

from training.utils.config import WandBConfig
from training.utils.runner import WandbConfigError

logger = logging.getLogger(__name__)


def _is_wandb_auth_error(exc: BaseException) -> bool:
    """Heuristically classify a W&B exception as an auth/permission/config error.

    Recognizes ``wandb.errors`` auth/usage types when available, and falls back
    to message inspection (401 / unauthorized / api key / permission) so the
    classification is robust across wandb versions.
    """
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
    return any(token in msg for token in ("401", "unauthorized", "permission", "api key", "api_key", "authentication"))


def setup_wandb(wb: WandBConfig, config: dict[str, Any]) -> bool:
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
            "WANDB_API_KEY not set; running wandb in offline mode. " "Set the key to sync runs to the dashboard."
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
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("perf/*", step_metric="train/step")
        wandb.define_metric("rollout/*", step_metric="train/step")
        wandb.define_metric("batch/*", step_metric="train/step")
        wandb.define_metric("infra/*", step_metric="train/step")
        wandb.define_metric("ctx/*", step_metric="train/step")
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


def wandb_finish() -> None:
    """Finish WandB run if active."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


def log_metrics_json(step: int, **kwargs: float) -> None:
    """Print a JSON metrics line to stdout (for structured log parsing)."""
    print(json.dumps({"type": "metrics", "step": step, **kwargs}), flush=True)
