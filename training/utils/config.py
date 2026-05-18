"""Shared configuration dataclasses for cookbook recipes.

The infra-provisioning dataclasses (``InfraConfig``, ``DeployConfig``,
``WeightSyncScope``) now live in the standalone ``fireworks-training-infra``
package. Import them from :mod:`fireworks_training_infra` directly. The
loop-level configs (``ConcurrencyConfig``, ``WeightSyncConfig``,
``WandBConfig``) and the callback type aliases continue to live in the cookbook.
"""

from __future__ import annotations

from typing import Dict, Callable
from dataclasses import dataclass

from fireworks.training.sdk.client import FiretitanTrainingClient

DEFAULT_ADAM = dict(beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, grad_clip_norm=1.0)

RewardFn = Callable[[str, dict], float]
"""Signature: (completion_text, dataset_row) -> reward_float."""

EvalFn = Callable[[int, FiretitanTrainingClient], Dict[str, float]]
"""Called every eval_every steps: (global_step, policy_client) -> metrics_dict."""

StepCallback = Callable[[int, Dict[str, float]], None]
"""Called after each optimizer step: (global_step, step_metrics) -> None."""


@dataclass
class ConcurrencyConfig:
    """Concurrency control settings for inference sampling.

    Two modes:

    * **adaptive** (default): Uses ``AdaptiveConcurrencyController`` which adjusts
      the concurrency window based on server-side ``prefill_queue_duration``.
    * **fixed**: Uses a fixed ``asyncio.Semaphore(max_concurrency)``.
    """

    mode: str | None = "adaptive"
    """``"adaptive"`` or ``"fixed"``.  ``None`` with ``max_concurrency`` set
    triggers the deprecated backward-compat path."""

    max_concurrency: int | None = None  # TODO: remove after deprecation period
    """Deprecated.  Set ``mode="fixed"`` and use ``FixedConcurrencyController``
    instead.  When set with ``mode=None``, creates a fixed controller and warns."""

    initial_window: int | None = None
    """Starting concurrency window for adaptive mode.
    Defaults to ``8 * replica_count`` when ``None``."""

    min_window: int = 1
    """Minimum concurrency window for adaptive mode."""

    max_window: int = 256
    """Maximum concurrency window for adaptive mode."""

    prefill_queue_target: float = 0.5
    """Target prefill queue duration (seconds) for the AIMD controller."""


@dataclass
class WeightSyncConfig:
    """Checkpoint and weight-sync settings."""

    weight_sync_interval: int = 1
    dcp_save_interval: int = 0
    dcp_timeout: int = 2700
    """Timeout in seconds for DCP save_state / load_state_with_optimizer (default 45 min)."""
    first_checkpoint_type: str = "base"
    weight_sync_before_training: bool = False
    weight_sync_timeout: int = 600


@dataclass
class WandBConfig:
    """Weights & Biases logging settings."""

    entity: str | None = None
    project: str | None = None
    run_name: str | None = None
