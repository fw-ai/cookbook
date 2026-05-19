"""Shared configuration dataclasses for cookbook recipes."""

from __future__ import annotations

import math
from enum import Enum
from typing import Dict, Callable
from dataclasses import dataclass, field

from fireworks.training.sdk.client import FiretitanTrainingClient
from fireworks.training.sdk.deployment import DeploymentConfig

DEFAULT_ADAM = dict(beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, grad_clip_norm=1.0)

# Schedule families follow the warmup + decay shapes studied in
# Naganuma et al. (2026), "What do near-optimal learning rate schedules
# look like?" https://arxiv.org/abs/2603.10301
LR_SCHEDULE_KINDS = {
    "constant",
    "cosine",
    "generalized_cosine",
    "linear",
    "two_point_linear",
}


@dataclass
class TwoPointLinearConfig:
    x1: float = 0.33
    """First interior x-position for two-point linear decay progress."""

    y1: float = 0.5
    """LR multiplier at ``x1``."""

    x2: float = 0.66
    """Second interior x-position for two-point linear decay progress."""

    y2: float = 0.1
    """LR multiplier at ``x2``."""


@dataclass
class LRScheduleConfig:
    kind: str = "constant"
    """Schedule after warmup: ``"constant"``, ``"cosine"``,
    ``"generalized_cosine"``, ``"linear"``, or ``"two_point_linear"``."""

    warmup_steps: int = 0
    """Linear LR warmup from 0 to learning_rate over the first N optimizer
    steps. 0 disables warmup."""

    min_lr_ratio: float = 0.0
    """Minimum LR as a fraction of ``learning_rate`` for decay schedules."""

    cosine_power: float = 1.0
    """Exponent for ``generalized_cosine``. 1.0 is the standard cosine shape."""

    two_point_linear: TwoPointLinearConfig = field(default_factory=TwoPointLinearConfig)


def validate_lr_schedule_config(schedule: LRScheduleConfig) -> None:
    if schedule.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if not 0.0 <= schedule.min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0.0 and 1.0")
    if schedule.kind not in LR_SCHEDULE_KINDS:
        allowed = ", ".join(sorted(LR_SCHEDULE_KINDS))
        raise ValueError(f"Unknown lr_schedule: {schedule.kind!r}. Use one of: {allowed}.")
    if schedule.cosine_power <= 0:
        raise ValueError("cosine_power must be > 0")

    if schedule.kind == "two_point_linear":
        x1 = schedule.two_point_linear.x1
        x2 = schedule.two_point_linear.x2
        y1 = schedule.two_point_linear.y1
        y2 = schedule.two_point_linear.y2
        if not 0.0 < x1 < x2 < 1.0:
            raise ValueError("two_point_linear.x1/x2 must satisfy 0.0 < x1 < x2 < 1.0")
        if not schedule.min_lr_ratio <= y2 <= y1 <= 1.0:
            raise ValueError(
                "two_point_linear.y1/y2 must satisfy min_lr_ratio <= y2 <= y1 <= 1.0"
            )


def _decay_progress(optim_step_idx: int, total_steps: int, warmup_steps: int) -> float:
    """Map a 1-indexed optimizer step to [0, 1] after warmup."""
    total_steps = max(total_steps, 1)
    if warmup_steps > 0:
        decay_steps = max(total_steps - warmup_steps, 1)
        decay_idx = min(max(optim_step_idx - warmup_steps, 0), decay_steps)
    else:
        decay_steps = max(total_steps - 1, 1)
        decay_idx = min(max(optim_step_idx - 1, 0), decay_steps)
    return decay_idx / decay_steps


def _piecewise_linear_multiplier(
    progress: float,
    *,
    min_lr_ratio: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    points = ((0.0, 1.0), (x1, y1), (x2, y2), (1.0, min_lr_ratio))
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        if progress <= right_x:
            span = right_x - left_x
            t = 0.0 if span == 0.0 else (progress - left_x) / span
            return left_y + t * (right_y - left_y)
    return min_lr_ratio


def compute_learning_rate(
    optim_step_idx: int,
    total_steps: int,
    peak_lr: float,
    schedule: LRScheduleConfig,
) -> float:
    """Return the LR for a 1-indexed optimizer step."""
    if optim_step_idx < 1:
        raise ValueError("optim_step_idx must be >= 1")

    if schedule.warmup_steps > 0 and optim_step_idx <= schedule.warmup_steps:
        return peak_lr * (optim_step_idx / schedule.warmup_steps)

    if schedule.kind == "constant":
        return peak_lr

    min_lr = peak_lr * schedule.min_lr_ratio
    progress = _decay_progress(optim_step_idx, total_steps, schedule.warmup_steps)

    if schedule.kind == "cosine":
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (peak_lr - min_lr) * multiplier
    if schedule.kind == "generalized_cosine":
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (peak_lr - min_lr) * (multiplier ** schedule.cosine_power)
    if schedule.kind == "linear":
        return peak_lr - (peak_lr - min_lr) * progress
    if schedule.kind == "two_point_linear":
        multiplier = _piecewise_linear_multiplier(
            progress,
            min_lr_ratio=schedule.min_lr_ratio,
            x1=schedule.two_point_linear.x1,
            y1=schedule.two_point_linear.y1,
            x2=schedule.two_point_linear.x2,
            y2=schedule.two_point_linear.y2,
        )
        return peak_lr * multiplier

    allowed = ", ".join(sorted(LR_SCHEDULE_KINDS))
    raise ValueError(f"Unknown lr_schedule: {schedule.kind!r}. Use one of: {allowed}.")

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
class InfraConfig:
    """GPU, region, and image settings.

    Two launch paths:

    * **Shape path** (``training_shape_id`` set): the backend owns all
      shape-derived fields (accelerator, image tag, node count).
      Setting shape-owned infra overrides raises ``ValueError``; trainer
      replica count remains a run-level control.
    * **Manual path** (``training_shape_id`` is ``None``): all fields
      are sent as-is; the server skips shape validation.
    """

    training_shape_id: str | None = None
    """Training shape ID for the policy trainer (e.g. ``ts-qwen3-8b-policy``).
    When set, infra config is auto-derived from the shape."""

    ref_training_shape_id: str | None = None
    """Training shape ID for a **separate** forward-only reference trainer.
    Only relevant for **full-parameter** training — LoRA runs use the
    shared-session reference (``policy.create_base_reference()``) on the
    policy trainer, so leave this ``None`` when ``lora_rank > 0``. For
    full-param, when set, a second trainer is provisioned; when not set,
    no reference is created. Can be the same value as ``training_shape_id``
    — the control plane auto-appends ``--forward-only``."""

    region: str | None = None
    custom_image_tag: str | None = None
    accelerator_type: str | None = None
    accelerator_count: int | None = None
    node_count: int | None = None
    trainer_timeout_s: float = 3600
    extra_args: list[str] | None = None
    trainer_replica_count: int | None = None
    """Data-parallel trainer replica count for service-mode HSDP launches.

    Leave unset for the backend default. Values greater than 1 request
    replicated HSDP for this trainer launch; this is intentionally a run-level
    knob, not part of the validated training shape.
    """
    purpose: str | None = None
    """Optional ``Purpose`` proto enum name (e.g. ``"PURPOSE_PILOT"``)."""
    managed_by: str | None = None
    """Internal. Populated automatically by the Fireworks platform when needed."""
    skip_validations: bool = False
    """Skip server-side shape validation. Requires superuser API key."""


class WeightSyncScope(Enum):
    """How trainer weights are synced to the inference deployment.

    ``PER_TRAINER`` (default)
        Trainer is provisioned first; the deployment is created (or re-wired on
        resume) to pull weights from that trainer's bucket.  Each trainer gets
        its own bucket.  On resume, the deployment is re-pointed to the new
        trainer, which briefly restarts the serving pod.

    ``PER_DEPLOYMENT``
        Deployment is provisioned first with a stable, deployment-scoped bucket.
        Trainers are created referencing that deployment so they all write to the
        same bucket URL.  On resume, a new trainer is created pointing at the
        same deployment — no serving pod restart required.
    """

    PER_TRAINER = "per_trainer"
    PER_DEPLOYMENT = "per_deployment"


@dataclass
class DeployConfig:
    """Inference deployment settings."""

    weight_sync_scope: WeightSyncScope = WeightSyncScope.PER_TRAINER
    """Controls how trainer weights reach the inference deployment; see :class:`WeightSyncScope`."""
    deployment_id: str | None = None
    """If set, use this existing deployment.  If ``None``, a new deployment
    is auto-created (ID derived from the base model name)."""
    deployment_shape: str | None = None
    """Deployment shape resource name.  Should always be a **versioned** path
    (e.g. ``accounts/fw/deploymentShapes/ds-x/versions/abc123``) to pin the
    exact shape config.  Recipes populate this from
    ``profile.deployment_shape`` which returns the versioned path."""
    deployment_region: str | None = None
    deployment_accelerator_type: str | None = None
    hot_load_bucket_type: str = "FW_HOSTED"
    hot_load_trainer_job: str | None = None
    """Trainer job name whose hot-load bucket this deployment should use.
    Format: accounts/{account}/rlorTrainerJobs/{job_id}.
    When set, the deployment copies the trainer's bucket URL at creation."""
    deployment_timeout_s: float = 5400
    reattach_settle_timeout_s: int = 600
    """How long to wait for the serving pod to cycle after a re-attach PATCH
    (separate from the full deployment creation timeout)."""
    deployment_extra_args: list[str] | None = None
    tokenizer_model: str | None = None
    """HuggingFace model name for the tokenizer (e.g. ``Qwen/Qwen3-1.7B``).
    Required for client-side tokenization (GRPO)."""
    tokenizer_revision: str | None = None
    """Optional HuggingFace revision for client-side tokenization."""
    sample_timeout: int = 600
    """HTTP read timeout in seconds for sampling completions (default 10 min).
    Increase for R3 + long completions where responses can be very large."""
    disable_speculative_decoding: bool = True
    """Disable base model's default draft/EAGLE speculation for hotload compatibility."""
    replica_count: int | None = None
    """If set, pin the deployment to a fixed replica count."""
    extra_values: dict[str, str] | None = None
    """Extra Helm values for the deployment (e.g. ``{"priorityClass": "deployment"}``)."""

    def to_deployment_config(
        self,
        base_model: str,
        infra: InfraConfig,
    ) -> DeploymentConfig:
        """Produce an SDK-level DeploymentConfig from cookbook settings."""
        skip_validation = False
        accel = None if self.deployment_shape else self.deployment_accelerator_type
        if not accel and not self.deployment_shape:
            accel = infra.accelerator_type
        replica_count = 1 if self.replica_count is None else self.replica_count
        return DeploymentConfig(
            deployment_id=self.deployment_id,
            base_model=base_model,
            deployment_shape=self.deployment_shape,
            region=self.deployment_region or None,
            hot_load_bucket_type=self.hot_load_bucket_type,
            hot_load_trainer_job=self.hot_load_trainer_job,
            skip_shape_validation=skip_validation,
            extra_args=self.deployment_extra_args,
            min_replica_count=replica_count,
            max_replica_count=replica_count,
            accelerator_type=accel,
            disable_speculative_decoding=self.disable_speculative_decoding,
            extra_values=self.extra_values,
        )


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
