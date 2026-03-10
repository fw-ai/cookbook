"""Infrastructure setup utilities: RLOR jobs, deployments, training clients."""

from __future__ import annotations

import time
import logging

from fireworks.training.sdk.client import (
    FiretitanServiceClient,
    FiretitanTrainingClient,
)
from fireworks.training.sdk.trainer import (
    TrainerJobConfig,
    TrainerJobManager,
    TrainingShapeProfile,
    TrainerServiceEndpoint,
)
from fireworks.training.sdk.deployment import DeploymentInfo, DeploymentManager
from training.utils.config import InfraConfig, DeployConfig

logger = logging.getLogger(__name__)


def _reject_shape_overrides(infra: InfraConfig) -> None:
    """Raise ``ValueError`` if infra has shape-derived overrides set.

    On the validated-shape path the backend populates accelerator,
    image tag, node count, etc. from the shape.  User-side overrides
    are only honoured on the ``skip_validations`` path.
    """
    overrides = []
    if infra.accelerator_type:
        overrides.append("accelerator_type")
    if infra.accelerator_count:
        overrides.append("accelerator_count")
    if infra.custom_image_tag:
        overrides.append("custom_image_tag")
    if infra.node_count is not None:
        overrides.append("node_count")
    if overrides:
        raise ValueError(
            f"InfraConfig.{', '.join(overrides)} cannot be set on a validated "
            f"training-shape launch. Use skip_validations=True to override."
        )


def create_trainer_job(
    rlor_mgr: TrainerJobManager,
    *,
    base_model: str,
    infra: InfraConfig,
    profile: TrainingShapeProfile | None = None,
    lora_rank: int = 0,
    max_seq_len: int | None = None,
    learning_rate: float = 1e-5,
    grad_accum: int = 1,
    display_name: str = "trainer",
    hot_load_deployment_id: str | None = None,
    extra_args: list[str] | None = None,
    job_id: str | None = None,
    forward_only: bool = False,
    base_url_override: str | None = None,
) -> TrainerServiceEndpoint:
    """Create a new RLOR trainer job (or reuse *job_id*).

    Two shape launch paths:

    * **Validated** (profile + ``skip_validations=False``): sends only
      ``training_shape_ref`` plus algorithm fields.  The backend
      populates accelerator, image tag, node count, sharding, etc.
      from the validated training shape.  ``InfraConfig`` overrides
      for shape-derived fields are rejected.
    * **Override** (profile + ``skip_validations=True``): resolves
      shape defaults via ``apply_shape``, lets user overrides win,
      and sends everything.
    * **Manual** (no profile): sends all ``InfraConfig`` fields as-is.

    When *base_url_override* is provided alongside *job_id*, skip health
    polling and return an endpoint pointing at that URL directly.
    Otherwise for pre-created jobs, the SDK routes through the gateway
    via /training/v1/rlorTrainerJobs/{accountId}/{jobId}/*.
    """
    if job_id:
        if base_url_override:
            job_name = f"accounts/{rlor_mgr.account_id}/rlorTrainerJobs/{job_id}"
            logger.info("Using pre-created job %s at %s", job_id, base_url_override)
            return TrainerServiceEndpoint(
                job_name=job_name,
                job_id=job_id,
                base_url=base_url_override,
            )
        return _reuse_or_resume_job(rlor_mgr, job_id)

    validated_shape = profile is not None and not infra.skip_validations

    if validated_shape:
        _reject_shape_overrides(infra)
        config = TrainerJobConfig(
            base_model=base_model,
            lora_rank=lora_rank,
            max_context_length=max_seq_len or profile.max_supported_context_length,
            learning_rate=learning_rate,
            gradient_accumulation_steps=grad_accum,
            display_name=display_name,
            hot_load_deployment_id=hot_load_deployment_id,
            region=infra.region,
            extra_args=extra_args or infra.extra_args,
            forward_only=forward_only,
            training_shape_ref=profile.training_shape_version,
        )
    else:
        config = TrainerJobConfig(
            base_model=base_model,
            lora_rank=lora_rank,
            max_context_length=max_seq_len,
            learning_rate=learning_rate,
            gradient_accumulation_steps=grad_accum,
            node_count=infra.node_count,
            display_name=display_name,
            hot_load_deployment_id=hot_load_deployment_id,
            region=infra.region,
            custom_image_tag=infra.custom_image_tag,
            extra_args=extra_args or infra.extra_args,
            accelerator_type=infra.accelerator_type,
            accelerator_count=infra.accelerator_count,
            skip_validations=infra.skip_validations,
            forward_only=forward_only,
        )
        if profile is not None:
            config.apply_shape(profile)
            config.training_shape_ref = profile.training_shape_version

    logger.info(
        "Creating trainer job '%s' (forward_only=%s)...",
        display_name,
        forward_only,
    )
    return rlor_mgr.create_and_wait(config, timeout_s=infra.trainer_timeout_s)


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
    if not deploy_cfg.deployment_id:
        model_short = base_model.rsplit("/", 1)[-1]
        deploy_cfg.deployment_id = f"{model_short}-{int(time.time())}"

    info = deploy_mgr.get(deploy_cfg.deployment_id)
    if not info:
        dep_config = deploy_cfg.to_deployment_config(base_model, infra)
        info = deploy_mgr.create_or_get(dep_config)

    if info.state not in ("READY", "UPDATING"):
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


def _reuse_or_resume_job(
    rlor_mgr: TrainerJobManager, job_id: str
) -> TrainerServiceEndpoint:
    job = rlor_mgr.get(job_id)
    state = job.get("state", "")
    resumable = (
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_PAUSED",
        "JOB_STATE_COMPLETED",
    )
    if state in resumable:
        logger.info("Job %s is %s, resuming...", job_id, state)
        return rlor_mgr.resume_and_wait(job_id)
    return rlor_mgr.wait_for_existing(job_id)
