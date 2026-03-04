"""Infrastructure setup utilities: RLOR jobs, deployments, training clients."""

from __future__ import annotations

import re
import time
import logging

from fireworks.training.sdk.client import FiretitanServiceClient, FiretitanTrainingClient
from fireworks.training.sdk.trainer import (
    TrainerJobConfig,
    TrainerJobManager,
    TrainingShapeProfile,
    TrainerServiceEndpoint,
)
from fireworks.training.sdk.deployment import DeploymentInfo, DeploymentManager
from fireworks.training.cookbook.utils.config import InfraConfig, DeployConfig

logger = logging.getLogger(__name__)


def resolve_and_apply_shape(
    rlor_mgr: TrainerJobManager,
    base_model: str,
    infra: InfraConfig,
    deploy_cfg: DeployConfig | None = None,
) -> TrainingShapeProfile:
    """Fetch a training shape and apply its values to *infra* and *deploy_cfg*.

    The shape provides authoritative defaults for accelerator, image, node
    count, and deployment shape.  When ``skip_validations=True``, user-provided
    values take priority and a warning is logged for each override.
    """
    if not infra.training_shape_id:
        raise ValueError("training_shape_id is required for shape resolution")
    profile = rlor_mgr.resolve_training_profile(
        training_shape_id=infra.training_shape_id,
    )
    logger.info(
        "Resolved training shape: %s (accel=%s×%d, image=%s, pp=%d, max_ctx=%d)",
        profile.training_shape_version,
        profile.accelerator_type,
        profile.accelerator_count,
        profile.trainer_image_tag,
        profile.pipeline_parallelism,
        profile.max_supported_context_length,
    )

    overrides: list[str] = []

    def _apply(field_name: str, shape_val, default_val=None):
        """Apply *shape_val* to ``infra.<field_name>``.

        If the user already set a non-default value that differs from the
        shape and ``skip_validations`` is on, keep the user's value and
        record the override.  Otherwise the shape value wins.
        """
        if shape_val is None or shape_val == default_val:
            return
        current = getattr(infra, field_name)
        is_user_set = current is not None and current != default_val and current != shape_val
        if is_user_set and infra.skip_validations:
            overrides.append(f"{field_name}={current} (shape: {shape_val})")
        else:
            setattr(infra, field_name, shape_val)

    accel = profile.accelerator_type
    if accel == "ACCELERATOR_TYPE_UNSPECIFIED":
        accel = None
    _apply("accelerator_type", accel)
    _apply("accelerator_count", profile.accelerator_count, default_val=0)
    _apply("custom_image_tag", profile.trainer_image_tag, default_val="")
    _apply("node_count", profile.node_count, default_val=0)

    if deploy_cfg is not None and profile.deployment_shape_version:
        dsv = profile.deployment_shape_version
        shape_name = re.sub(r"/versions/[^/]+$", "", dsv)
        if deploy_cfg.deployment_shape and deploy_cfg.deployment_shape != shape_name:
            if infra.skip_validations:
                overrides.append(f"deployment_shape={deploy_cfg.deployment_shape} (shape: {shape_name})")
            else:
                deploy_cfg.deployment_shape = shape_name
        else:
            deploy_cfg.deployment_shape = shape_name

    if overrides:
        logger.warning(
            "Training shape '%s' values overridden (skip_validations=True): %s",
            infra.training_shape_id,
            "; ".join(overrides),
        )

    return profile


def create_trainer_job(
    rlor_mgr: TrainerJobManager,
    *,
    base_model: str,
    infra: InfraConfig,
    lora_rank: int = 0,
    max_seq_len: int | None = None,
    learning_rate: float = 1e-5,
    grad_accum: int = 4,
    display_name: str = "trainer",
    hot_load_deployment_id: str | None = None,
    extra_args: list[str] | None = None,
    job_id: str | None = None,
    forward_only: bool = False,
) -> TrainerServiceEndpoint:
    """Create a new RLOR trainer job (or reuse *job_id*).

    *forward_only* sets ``forwardOnly`` on the API request so the CP
    resolves FORWARD_ONLY training shapes and appends the runtime flags.
    """
    if job_id:
        return _reuse_or_resume_job(rlor_mgr, job_id)

    using_shape = bool(infra.training_shape_id and not infra.skip_validations)
    node_count = infra.node_count if infra.node_count is not None else 1
    logger.info("Creating trainer job '%s' (nodes=%d, shape=%s, forward_only=%s)...", display_name, node_count, using_shape, forward_only)
    return rlor_mgr.create_and_wait(
        TrainerJobConfig(
            base_model=base_model,
            lora_rank=lora_rank,
            max_context_length=max_seq_len if not using_shape else None,
            learning_rate=learning_rate,
            gradient_accumulation_steps=grad_accum,
            node_count=node_count,
            display_name=display_name,
            hot_load_deployment_id=hot_load_deployment_id,
            region=infra.region,
            custom_image_tag=infra.custom_image_tag if not using_shape else None,
            extra_args=(extra_args or infra.extra_args) if not using_shape else None,
            accelerator_type=infra.accelerator_type if not using_shape else None,
            accelerator_count=infra.accelerator_count if not using_shape else None,
            skip_validations=infra.skip_validations,
            forward_only=forward_only,
        )
    )


def setup_deployment(
    deploy_mgr: DeploymentManager,
    deploy_cfg: DeployConfig,
    base_model: str,
    infra: InfraConfig,
) -> DeploymentInfo:
    """Set up an inference deployment.

    * If ``deploy_cfg.deployment_id`` is set, the existing deployment is
      fetched and waited on until READY.
    * If ``deploy_cfg.deployment_id`` is ``None``, a new deployment is
      created with an auto-generated ID derived from the model name.
    """
    if deploy_cfg.deployment_id:
        info = deploy_mgr.get(deploy_cfg.deployment_id)
        if not info:
            raise RuntimeError(
                f"Deployment '{deploy_cfg.deployment_id}' not found. "
                f"Remove deployment_id to auto-create a new deployment, "
                f"or verify the ID is correct."
            )
    else:
        model_short = base_model.rsplit("/", 1)[-1]
        deploy_cfg.deployment_id = f"{model_short}-{int(time.time())}"
        dep_config = deploy_cfg.to_deployment_config(base_model, infra)
        info = deploy_mgr.create_or_get(dep_config)

    if info.state != "READY":
        info = deploy_mgr.wait_for_ready(
            deploy_cfg.deployment_id,
            timeout_s=deploy_cfg.deployment_timeout_s,
        )
    return info


def setup_training_client(
    endpoint: TrainerServiceEndpoint,
    base_model: str,
    api_key: str = "tml-local",
    lora_rank: int = 0,
) -> tuple[FiretitanServiceClient, FiretitanTrainingClient]:
    """Create a ServiceClient + TrainingClient for a trainer endpoint."""
    svc = FiretitanServiceClient(base_url=endpoint.base_url, api_key=api_key)
    cli = svc.create_training_client(base_model=base_model, lora_rank=lora_rank)
    return svc, cli


def _reuse_or_resume_job(rlor_mgr: TrainerJobManager, job_id: str) -> TrainerServiceEndpoint:
    job = rlor_mgr.get(job_id)
    state = job.get("state", "")
    resumable = ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED", "JOB_STATE_COMPLETED")
    if state in resumable:
        logger.info("Job %s is %s, resuming...", job_id, state)
        return rlor_mgr.resume_and_wait(job_id)
    return rlor_mgr.wait_for_existing(job_id)
