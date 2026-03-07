"""Infrastructure setup utilities: RLOR jobs, deployments, training clients."""

from __future__ import annotations

import time
import logging
from typing import Any

import requests as _requests

from fireworks.training.sdk.client import FiretitanServiceClient, FiretitanTrainingClient
from fireworks.training.sdk.trainer import (
    TrainerJobConfig,
    TrainerJobManager,
    TrainingShapeProfile,
    TrainerServiceEndpoint,
)
from fireworks.training.sdk.deployment import (
    DeploymentConfig,
    DeploymentInfo,
    DeploymentManager,
    request_with_retries,
)
from training.utils.config import InfraConfig, DeployConfig

logger = logging.getLogger(__name__)


def create_trainer_job(
    rlor_mgr: TrainerJobManager,
    *,
    base_model: str,
    infra: InfraConfig,
    profile: TrainingShapeProfile | None = None,
    lora_rank: int = 0,
    max_seq_len: int | None = None,
    learning_rate: float = 1e-5,
    display_name: str = "trainer",
    hot_load_deployment_id: str | None = None,
    extra_args: list[str] | None = None,
    job_id: str | None = None,
    forward_only: bool = False,
) -> TrainerServiceEndpoint:
    """Create a new RLOR trainer job (or reuse *job_id*).

    When *profile* is provided, shape fields (image tag, node count,
    accelerator type/count, context length) are applied to the config.
    """
    if job_id:
        return _reuse_or_resume_job(rlor_mgr, job_id)

    config = TrainerJobConfig(
        base_model=base_model,
        lora_rank=lora_rank,
        max_context_length=max_seq_len,
        learning_rate=learning_rate,
        node_count=infra.node_count if infra.node_count is not None else 1,
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

    logger.info(
        "Creating trainer job '%s' (forward_only=%s)...",
        display_name, forward_only,
    )
    return rlor_mgr.create_and_wait(config)


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
        if deploy_cfg.extra_values:
            info = _create_deployment_with_extra_values(
                deploy_mgr, dep_config, deploy_cfg.extra_values,
            )
        else:
            info = deploy_mgr.create_or_get(dep_config)

    if info.state != "READY":
        info = deploy_mgr.wait_for_ready(
            deploy_cfg.deployment_id,
            timeout_s=deploy_cfg.deployment_timeout_s,
        )
    return info


def _create_deployment_with_extra_values(
    deploy_mgr: DeploymentManager,
    config: DeploymentConfig,
    extra_values: dict[str, str],
) -> DeploymentInfo:
    """Create a deployment with ``extraValues`` injected into the API body.

    The SDK's ``DeploymentManager`` doesn't support ``extraValues`` natively,
    so we replicate the essential creation logic here.
    """
    url = (
        f"{deploy_mgr.base_url}/v1/accounts/{deploy_mgr.account_id}"
        f"/deployments?deploymentId={config.deployment_id}"
        f"&skipShapeValidation={'true' if config.skip_shape_validation else 'false'}"
    )
    if config.disable_speculative_decoding:
        url += "&disableSpeculativeDecoding=true"

    body: dict[str, Any] = {
        "baseModel": config.base_model,
        "minReplicaCount": config.min_replica_count,
        "maxReplicaCount": config.max_replica_count,
        "enableHotLoad": True,
        "placement": {"region": config.region},
        "extraValues": extra_values,
    }
    if config.hot_load_bucket_type:
        body["hotLoadBucketType"] = config.hot_load_bucket_type
    if config.deployment_shape:
        body["deploymentShape"] = config.deployment_shape
    if config.accelerator_type:
        body["acceleratorType"] = config.accelerator_type
    if config.extra_args:
        flat: list[str] = []
        for arg in config.extra_args:
            flat.extend(arg.split()) if " " in arg else flat.append(arg)
        body["extraArgs"] = flat

    headers = deploy_mgr._headers()
    logger.info("Creating deployment with extra_values: %s", config.deployment_id)
    resp = request_with_retries(
        _requests.post, url, headers=headers, json=body, timeout=60,
        verify=deploy_mgr._verify_ssl,
    )
    resp.raise_for_status()
    data = resp.json()
    return deploy_mgr._parse_deployment_info(config.deployment_id, data)


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
