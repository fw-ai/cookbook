"""Shared configuration dataclasses for cookbook recipes."""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Dict, Callable
from dataclasses import dataclass

from fireworks.training.sdk.client import FiretitanTrainingClient
from fireworks.training.sdk.deployment import DeploymentConfig

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

    max_concurrency: int | None = None  # Deprecated alias for max_concurrency_rollout_sample.
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

    rollout_adjustment_interval: int = 32
    """Adjust adaptive concurrency every N completed requests within an RL step.
    Remaining requests adjust at the step boundary; ``0`` adjusts only there."""


@dataclass
class InfraConfig:
    """GPU and image settings.

    .. deprecated::
        ``InfraConfig`` is retained only for backward compatibility. New code
        should configure provisioning with :class:`TrainerConfig` (and
        :class:`DeployConfig`) and use the recipe's SDK-managed provisioning
        path.
        The recipe ``Config`` objects no longer accept an ``infra=`` field.

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
    """Training shape ID for a **separate** frozen reference trainer.
    Only relevant for **full-parameter** training — LoRA runs reuse the
    policy session as their frozen reference (adapter disabled), so leave
    this ``None`` when ``lora_rank > 0``. For
    full-param, when set, a second trainer is provisioned; when not set,
    the backend auto-selects a LoRA-capable shape for the second trainer."""

    custom_image_tag: str | None = None
    region: str | None = None
    """Optional explicit trainer region. Prefer leaving unset so the backend
    can place the trainer from the validated training shape."""
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
    """Optional platform ``Purpose`` proto enum name."""
    skip_validations: bool = False
    """Skip server-side shape validation. Requires superuser API key."""

    def __post_init__(self) -> None:
        warnings.warn(
            "InfraConfig is deprecated; use TrainerConfig (and DeployConfig) with "
            "the recipe's SDK-managed provisioning path. It is retained only for backward compatibility "
            "and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )


@dataclass
class TrainerConfig:
    """Training client launch settings."""

    job_id: str | None = None
    """Existing trainer job ID to reattach to. Leave unset to create one."""

    training_shape_id: str | None = None
    """Training shape ID for the policy trainer."""

    reference_training_shape_id: str | None = None
    """Training shape ID for a separate frozen reference trainer.

    Leave unset for full-parameter KL/DPO runs unless you need to pin a
    specific LoRA-capable reference shape; backend trainer creation
    auto-selects one by default.
    """

    reference_job_id: str | None = None
    """Existing frozen reference trainer job ID to reattach to."""

    cleanup_reference_on_close: bool = True
    """Delete SDK-created separate reference trainers when the service closes."""

    custom_image_tag: str | None = None
    region: str | None = None
    """Optional explicit trainer region. Prefer leaving unset so the backend
    can place the trainer from the validated training shape."""
    accelerator_type: str | None = None
    """Deprecated and ignored. Trainer accelerator type is owned by the
    training shape; setting this emits a ``DeprecationWarning``. Use
    ``replica_count`` for data-parallel scaling."""
    accelerator_count: int | None = None
    """Deprecated and ignored. Trainer accelerator count is owned by the
    training shape; setting this emits a ``DeprecationWarning``. Use
    ``replica_count`` for data-parallel scaling."""
    node_count: int | None = None
    timeout_s: float = 3600
    """Post-placement budget for the trainer to become healthy and ready."""
    pending_timeout_s: float = 48 * 60 * 60
    """Capacity-placement budget while the trainer remains ``PENDING``."""
    extra_args: list[str] | None = None
    replica_count: int | None = None
    """Data-parallel trainer replica count for service-mode HSDP launches."""
    inactivity_timeout: str | None = None
    """Trainer inactivity timeout as a protobuf duration string, e.g. ``"7200s"``.

    Leave unset to use the backend default. Set ``disable_inactivity_cleanup``
    to keep a provisioned trainer alive while idle.
    """
    disable_inactivity_cleanup: bool = False
    """Disable automatic trainer cleanup after inactivity."""

    purpose: str | None = None
    """Optional platform ``Purpose`` proto enum name."""
    preemptible: bool = False
    """Request preemptible trainer scheduling. Requires an admin API key."""

    managed_by: str | None = None
    """Optional parent resource ID for managed trainer ownership."""

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
    deployment_accelerator_type: str | None = None
    """DEPRECATED and ignored on the SDK-managed path. The deployment shape
    owns accelerator selection; set the accelerator via ``deployment_shape``.
    Retained only for the legacy ``to_deployment_config`` path."""
    hot_load_bucket_type: str = "FW_HOSTED"
    hot_load_trainer_job: str | None = None
    """Trainer job name whose hot-load bucket this deployment should use.
    Format: accounts/{account}/rlorTrainerJobs/{job_id}.
    When set, the deployment copies the trainer's bucket URL at creation."""
    enable_hot_load: bool = True
    """Whether to create a hot-load-capable deployment."""
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
    disable_speculative_decoding: bool = False
    """When true, disable the base model's default draft/EAGLE speculation."""
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
            region=infra.region,
            deployment_shape=self.deployment_shape,
            hot_load_bucket_type=self.hot_load_bucket_type if self.enable_hot_load else None,
            hot_load_trainer_job=self.hot_load_trainer_job if self.enable_hot_load else None,
            enable_hot_load=self.enable_hot_load,
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
