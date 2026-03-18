"""Infrastructure setup utilities: RLOR jobs, deployments, training clients."""

from __future__ import annotations

import time
import logging
from typing import Any
from urllib.parse import urlencode

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
from fireworks.training.sdk.deployment import DeploymentConfig, DeploymentInfo, DeploymentManager
from training.utils.config import InfraConfig, DeployConfig

logger = logging.getLogger(__name__)

_DEPLOYMENT_ACCELERATOR_REGION_PREFIXES: tuple[tuple[str, str], ...] = (
    ("NVIDIA_H200", "US_VIRGINIA_1"),
    ("NVIDIA_B200", "US_OHIO_1"),
)


class ResourceCleanup:
    """Register resource IDs for automatic cleanup on scope exit.

    Wrap recipe logic in ``with ResourceCleanup(...) as cleanup:`` and
    call ``cleanup.trainer(job_id)`` / ``cleanup.deployment(dep_id)``
    after creating each resource.  On scope exit (including exceptions),
    registered resources are deleted in reverse creation order.

    Pre-created resources that should survive simply aren't registered.
    """

    def __init__(
        self,
        rlor_mgr: TrainerJobManager,
        deploy_mgr: DeploymentManager | None = None,
    ):
        self._rlor_mgr = rlor_mgr
        self._deploy_mgr = deploy_mgr
        self._jobs: list[str] = []
        self._deployments: list[tuple[str, str]] = []
        self._job_ids: set[str] = set()
        self._deployment_actions: dict[str, str] = {}

    def trainer(self, job_id: str) -> None:
        """Register a trainer job for deletion on exit."""
        if job_id in self._job_ids:
            return
        self._job_ids.add(job_id)
        self._jobs.append(job_id)

    def deployment(self, dep_id: str, action: str = "delete") -> None:
        """Register a deployment for cleanup on exit.

        *action*: ``"delete"`` (default) or ``"scale_to_zero"``.
        """
        existing_action = self._deployment_actions.get(dep_id)
        if existing_action is not None:
            if existing_action != action:
                logger.info(
                    "Cleanup: deployment %s already registered with action %s; "
                    "ignoring duplicate action %s",
                    dep_id,
                    existing_action,
                    action,
                )
            return
        self._deployment_actions[dep_id] = action
        self._deployments.append((dep_id, action))

    def __enter__(self) -> ResourceCleanup:
        return self

    def __exit__(self, *exc) -> None:
        for jid in reversed(self._jobs):
            try:
                logger.info("Cleanup: deleting trainer job %s", jid)
                self._rlor_mgr.delete(jid)
            except Exception as e:
                logger.warning("Cleanup: failed to delete trainer %s: %s", jid, e)
        for did, action in reversed(self._deployments):
            try:
                if action == "scale_to_zero":
                    logger.info("Cleanup: scaling deployment %s to zero", did)
                    self._deploy_mgr.scale_to_zero(did)
                else:
                    logger.info("Cleanup: deleting deployment %s", did)
                    self._deploy_mgr.delete(did)
            except Exception as e:
                logger.warning("Cleanup: failed to clean deployment %s: %s", did, e)


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
    cleanup: ResourceCleanup | None = None,
) -> TrainerServiceEndpoint:
    """Create a new RLOR trainer job (or reuse *job_id*).

    Two launch paths:

    * **Shape path** (profile provided): sends ``training_shape_ref``
      plus algorithm fields only.  The backend populates accelerator,
      image tag, node count, sharding, etc. from the validated
      training shape.
    * **Manual path** (no profile): sends all ``InfraConfig`` fields
      as-is; the server skips shape validation.

    When *base_url_override* is provided alongside *job_id*, skip health
    polling and return an endpoint pointing at that URL directly.
    Otherwise for pre-created jobs, the SDK routes through the gateway
    via /training/v1/rlorTrainerJobs/{accountId}/{jobId}/*.
    """
    trainer_role = "reference" if forward_only else "policy"

    if job_id:
        if base_url_override:
            job_name = f"accounts/{rlor_mgr.account_id}/rlorTrainerJobs/{job_id}"
            logger.info(
                "Using pre-created %s trainer job %s at %s",
                trainer_role,
                job_id,
                base_url_override,
            )
            return TrainerServiceEndpoint(
                job_name=job_name,
                job_id=job_id,
                base_url=base_url_override,
            )
        return _reuse_or_resume_job(rlor_mgr, job_id)

    if profile is not None:
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
            forward_only=forward_only,
        )

    logger.info(
        "Creating %s trainer job '%s' (forward_only=%s)...",
        trainer_role,
        display_name,
        forward_only,
    )
    create = getattr(rlor_mgr, "create", None)
    wait_for_ready = getattr(rlor_mgr, "wait_for_ready", None)
    if callable(create) and callable(wait_for_ready):
        created = create(config)
        if cleanup is not None:
            cleanup.trainer(created.job_id)
        return wait_for_ready(
            created.job_id,
            job_name=created.job_name,
            timeout_s=infra.trainer_timeout_s,
        )
    return rlor_mgr.create_and_wait(config, timeout_s=infra.trainer_timeout_s)


def setup_deployment(
    deploy_mgr: DeploymentManager,
    deploy_cfg: DeployConfig,
    base_model: str,
    infra: InfraConfig,
    cleanup: ResourceCleanup | None = None,
    cleanup_action: str = "delete",
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
        if dep_config.region is None and dep_config.deployment_shape:
            dep_config.region = _infer_region_from_deployment_shape(
                deploy_mgr, dep_config.deployment_shape
            )
        if dep_config.region is None:
            info = _create_deployment_via_cookbook(deploy_mgr, dep_config)
        else:
            info = deploy_mgr.create_or_get(dep_config)
        if cleanup is not None:
            cleanup.deployment(deploy_cfg.deployment_id, action=cleanup_action)

    if info.state not in ("READY", "UPDATING"):
        info = deploy_mgr.wait_for_ready(
            deploy_cfg.deployment_id,
            timeout_s=deploy_cfg.deployment_timeout_s,
        )
    return info


def _infer_region_from_deployment_shape(
    deploy_mgr: DeploymentManager,
    deployment_shape: str,
) -> str | None:
    """Infer a rollout region from a validated deployment shape snapshot."""
    version = _get_deployment_shape_version(deploy_mgr, deployment_shape)
    snapshot = version.get("snapshot", {}) or {}
    accelerator = snapshot.get("acceleratorType", "")
    for prefix, region in _DEPLOYMENT_ACCELERATOR_REGION_PREFIXES:
        if accelerator.startswith(prefix):
            logger.info(
                "Inferred deployment region %s from deployment shape %s (accelerator=%s)",
                region,
                deployment_shape,
                accelerator,
            )
            return region
    if accelerator:
        logger.info(
            "No cookbook deployment-region override for deployment shape %s (accelerator=%s); using auto placement",
            deployment_shape,
            accelerator,
        )
    else:
        logger.info(
            "Deployment shape %s returned no accelerator type; using auto placement",
            deployment_shape,
        )
    return None


def _get_deployment_shape_version(
    deploy_mgr: DeploymentManager,
    deployment_shape: str,
) -> dict[str, Any]:
    """Fetch an exact or latest-validated deployment-shape version."""
    if "/versions/" in deployment_shape:
        path = f"/v1/{deployment_shape}"
    else:
        path = (
            f"/v1/{deployment_shape}/versions?"
            f"{urlencode({'filter': 'latest_validated=true', 'pageSize': 1})}"
        )
    resp = deploy_mgr._get(path, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "/versions/" in deployment_shape:
        return data
    versions = data.get("deploymentShapeVersions", []) or []
    if not versions:
        raise RuntimeError(
            f"No latest validated deployment-shape version was returned for '{deployment_shape}'"
        )
    return versions[0]


def _create_deployment_via_cookbook(
    deploy_mgr: DeploymentManager,
    config: DeploymentConfig,
) -> DeploymentInfo:
    """Create a deployment while leaving placement selection to the control plane."""
    path = f"/v1/accounts/{deploy_mgr.account_id}/deployments?deploymentId={config.deployment_id}"
    if config.skip_shape_validation:
        path += "&skipShapeValidation=true"
    if config.disable_speculative_decoding:
        path += "&disableSpeculativeDecoding=true"

    body: dict[str, Any] = {
        "baseModel": config.base_model,
        "minReplicaCount": config.min_replica_count,
        "maxReplicaCount": config.max_replica_count,
        "enableHotLoad": True,
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
    if config.extra_values:
        body["extraValues"] = config.extra_values

    logger.info(
        "Creating deployment: %s (placement_region=auto, extra_values=%s)",
        config.deployment_id,
        bool(config.extra_values),
    )
    resp = deploy_mgr._post(path, json=body, timeout=60)
    resp.raise_for_status()
    return deploy_mgr._parse_deployment_info(config.deployment_id, resp.json())


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
