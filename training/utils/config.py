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
class ConcurrencyConfig:
    """Concurrency control settings for inference sampling.

    Two modes:

    * **fixed** (default): Uses a fixed ``asyncio.Semaphore(max_concurrency)``.
      Equivalent to the previous ``max_concurrency`` parameter.
    * **adaptive**: Uses ``AdaptiveConcurrencyController`` which adjusts
      the concurrency window based on server-side ``prefill_queue_duration``.
      Requires ``stream=True`` on the sampler to receive metrics.
    """

    mode: str = "fixed"
    """``"fixed"`` or ``"adaptive"``."""

    max_concurrency: int | None = None
    """Fixed concurrency limit.  Used when ``mode="fixed"``.
    ``None`` means unlimited."""

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
    max_concurrent: int = 0
    """Cap concurrent sampling requests.  Passed to both
    DeploymentSampler(max_concurrency=...) for HTTP-level gating and to
    run_rl_loop for coroutine-level gating.  0 = unlimited."""


@dataclass
class WandBConfig:
    """Weights & Biases logging settings."""

    entity: str | None = None
    project: str | None = None
    run_name: str | None = None

