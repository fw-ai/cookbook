"""Configuration dataclasses for trainer and deployment provisioning.

This module owns the dataclasses needed by :mod:`fireworks_training_infra.infra`.
The cookbook re-exports them from ``training.utils.config`` for backwards
compatibility, alongside its own loop-level configs (``WeightSyncConfig``,
``ConcurrencyConfig``, ``WandBConfig``).
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass

from fireworks.training.sdk.deployment import DeploymentConfig


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
