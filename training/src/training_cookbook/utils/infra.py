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
from training_cookbook.utils.config import InfraConfig, DeployConfig

logger = logging.getLogger(__name__)


def resolve_and_apply_shape(
    rlor_mgr: TrainerJobManager,
    base_model: str,
    infra: InfraConfig,
    deploy_cfg: DeployConfig | None = None,
) -> TrainingShapeProfile:
    """Fetch training shape and populate unset fields on *infra* (and *deploy_cfg* if given)."""
    if not infra.training_shape_id:
        raise ValueError("training_shape_id is required for shape resolution")
    profile = rlor_mgr.resolve_training_profile(
        training_shape_id=infra.training_shape_id,
    )
    logger.info(
        "Resolved training shape: %s (accel=%s, image=%s, pp=%d)",
        profile.training_shape_version,
        profile.accelerator_type,
        profile.trainer_image_tag,
        profile.pipeline_parallelism,
    )

    if not infra.accelerator_type and profile.accelerator_type and profile.accelerator_type != "ACCELERATOR_TYPE_UNSPECIFIED":
        infra.accelerator_type = profile.accelerator_type
    if not infra.accelerator_count and profile.accelerator_count:
        infra.accelerator_count = profile.accelerator_count
    if not infra.custom_image_tag and profile.trainer_image_tag:
        infra.custom_image_tag = profile.trainer_image_tag
    if profile.node_count and profile.node_count > infra.node_count:
        infra.node_count = profile.node_count

    if deploy_cfg is not None:
        if not deploy_cfg.deployment_shape and profile.deployment_shape_version:
            dsv = profile.deployment_shape_version
            shape_name = re.sub(r"/versions/[^/]+$", "", dsv)
            deploy_cfg.deployment_shape = shape_name
        if not deploy_cfg.deployment_id:
            model_short = base_model.rsplit("/", 1)[-1]
            deploy_cfg.deployment_id = f"{model_short}-{int(time.time())}"

    return profile


def create_trainer_job(
    rlor_mgr: TrainerJobManager,
    *,
    base_model: str,
    infra: InfraConfig,
    lora_rank: int = 0,
    max_seq_len: int = 4096,
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

    using_shape = bool(infra.training_shape_id)
    node_count = infra.node_count if infra.node_count is not None else 1
    logger.info("Creating trainer job '%s' (nodes=%d, shape=%s, forward_only=%s)...", display_name, node_count, using_shape, forward_only)
    return rlor_mgr.create_and_wait(
        TrainerJobConfig(
            base_model=base_model,
            lora_rank=lora_rank,
            max_context_length=max_seq_len,
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
) -> DeploymentInfo | None:
    """Create or get a deployment. Returns None if no deployment configured."""
    if not deploy_cfg.create_deployment or not deploy_cfg.deployment_id:
        if deploy_cfg.deployment_id:
            return deploy_mgr.get(deploy_cfg.deployment_id)
        return None
    dep_config = deploy_cfg.to_deployment_config(base_model, infra)
    info = deploy_mgr.create_or_get(dep_config)
    if info.state != "READY":
        info = deploy_mgr.wait_for_ready(
            dep_config.deployment_id,
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
