"""Shared configuration dataclasses for cookbook recipes."""

from __future__ import annotations

from typing import Dict, Callable
from dataclasses import dataclass

from fireworks.training.sdk.client import FiretitanTrainingClient
from fireworks.training.sdk.deployment import DeploymentConfig

DEFAULT_ADAM = dict(beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)

RewardFn = Callable[[str, dict], float]
"""Signature: (completion_text, dataset_row) -> reward_float."""

EvalFn = Callable[[int, FiretitanTrainingClient], Dict[str, float]]
"""Called every eval_every steps: (global_step, policy_client) -> metrics_dict."""

StepCallback = Callable[[int, Dict[str, float]], None]
"""Called after each optimizer step: (global_step, step_metrics) -> None."""


@dataclass
class InfraConfig:
    """GPU, region, and image settings.

    Two launch paths:

    * **Shape path** (``training_shape_id`` set): the backend owns all
      shape-derived fields (accelerator, image tag, node count).
      Setting infra overrides raises ``ValueError``.
    * **Manual path** (``training_shape_id`` is ``None``): all fields
      are sent as-is; the server skips shape validation.
    """

    training_shape_id: str | None = None
    """Training shape ID for the policy trainer (e.g. ``ts-qwen3-8b-policy``).
    When set, infra config is auto-derived from the shape."""

    ref_training_shape_id: str | None = None
    """Training shape ID for the reference (forward-only) trainer.
    When set, a reference model is created.  When not set, no reference
    model is created.  No implicit fallback.  Can be the same value as
    ``training_shape_id`` -- the control plane auto-appends
    ``--forward-only`` via ``applyForwardOnlyConfig``."""

    region: str | None = None
    custom_image_tag: str | None = None
    accelerator_type: str | None = None
    accelerator_count: int | None = None
    node_count: int | None = None
    trainer_timeout_s: float = 3600
    extra_args: list[str] | None = None


@dataclass
class DeployConfig:
    """Inference deployment settings."""

    deployment_id: str | None = None
    """If set, use this existing deployment.  If ``None``, a new deployment
    is auto-created (ID derived from the base model name)."""
    deployment_shape: str | None = None
    deployment_region: str | None = None
    deployment_accelerator_type: str | None = None
    hot_load_bucket_type: str = "FW_HOSTED"
    deployment_timeout_s: float = 5400
    deployment_extra_args: list[str] | None = None
    tokenizer_model: str | None = None
    """HuggingFace model name for the tokenizer (e.g. ``Qwen/Qwen3-1.7B``).
    Required for client-side tokenization (GRPO)."""
    sample_timeout: int = 600
    """HTTP read timeout in seconds for sampling completions (default 10 min).
    Increase for R3 + long completions where responses can be very large."""
    disable_speculative_decoding: bool = True
    """Disable base model's default draft/EAGLE speculation for hotload compatibility."""
    hot_load_async_transition: bool = False
    """Enable serving-side async hotload transition via ``--hot-load-async-transition``."""
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
        extra_args = list(self.deployment_extra_args or [])
        if self.hot_load_async_transition and "--hot-load-async-transition" not in extra_args:
            extra_args.append("--hot-load-async-transition")
        return DeploymentConfig(
            deployment_id=self.deployment_id,
            base_model=base_model,
            deployment_shape=self.deployment_shape,
            region=self.deployment_region or None,
            hot_load_bucket_type=self.hot_load_bucket_type,
            skip_shape_validation=skip_validation,
            extra_args=extra_args or None,
            min_replica_count=1,
            max_replica_count=1,
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

