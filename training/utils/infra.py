"""Infrastructure setup utilities: RLOR jobs, deployments, training clients."""

from __future__ import annotations

import os
import time
import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from fireworks.training.sdk.client import (
    FiretitanServiceClient,
    FiretitanTrainingClient,
)
from fireworks.training.sdk import (
    TrainerJobConfig,
    TrainerJobManager,
    TrainerServiceEndpoint,
)
try:
    from fireworks.training.sdk.trainer import TrainingShapeProfile
except ImportError:
    from fireworks.training.sdk import TrainingShapeProfile
from fireworks.training.sdk.deployment import DeploymentConfig, DeploymentInfo, DeploymentManager
from training.utils.config import InfraConfig, DeployConfig

if TYPE_CHECKING:
    from fireworks.training.sdk.weight_syncer import WeightSyncer

logger = logging.getLogger(__name__)

_DEPLOYMENT_ACCELERATOR_REGION_PREFIXES: tuple[tuple[str, str], ...] = (
    ("NVIDIA_H200", "US_VIRGINIA_1"),
    ("NVIDIA_B200", "US_OHIO_1"),
)

_TRAINER_CANCEL_GRACE_ENV = "FW_TRAINER_CANCEL_GRACE_PERIOD_S"
_TRAINER_DELETE_GRACE_ENV = "FW_TRAINER_DELETE_GRACE_PERIOD_S"


def _default_trainer_cancel_grace_period_s() -> float:
    raw = os.environ.get(_TRAINER_CANCEL_GRACE_ENV)
    source_env = _TRAINER_CANCEL_GRACE_ENV
    if raw is None:
        raw = os.environ.get(_TRAINER_DELETE_GRACE_ENV, "30")
        source_env = _TRAINER_DELETE_GRACE_ENV
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(
            "Invalid %s=%r; falling back to 30s trainer cancel grace period",
            source_env,
            raw,
        )
        return 30.0


class ResourceCleanup:
    """Register resource IDs for automatic cleanup on scope exit.

    Wrap recipe logic in ``with ResourceCleanup(...) as cleanup:`` and
    call ``cleanup.trainer(job_id)`` / ``cleanup.deployment(dep_id)``
    after creating each resource.  On scope exit (including exceptions),
    registered trainers are cancelled and deployments are cleaned in
    reverse creation order.

    Pre-created resources that should survive simply aren't registered.
    """

    def __init__(
        self,
        rlor_mgr: TrainerJobManager,
        deploy_mgr: DeploymentManager | None = None,
        trainer_delete_grace_period_s: float | None = None,
        trainer_cancel_grace_period_s: float | None = None,
    ):
        self._rlor_mgr = rlor_mgr
        self._deploy_mgr = deploy_mgr
        self._jobs: list[str] = []
        self._deployments: list[tuple[str, str]] = []
        grace_period_s = (
            trainer_cancel_grace_period_s
            if trainer_cancel_grace_period_s is not None
            else trainer_delete_grace_period_s
        )
        self._trainer_cancel_grace_period_s = (
            _default_trainer_cancel_grace_period_s()
            if grace_period_s is None
            else max(0.0, float(grace_period_s))
        )

    def trainer(self, job_id: str) -> None:
        """Register a trainer job for cancellation on exit."""
        self._jobs.append(job_id)

    def cancel_trainer(self, job_id: str) -> None:
        """Cancel a trainer job now and unregister it from cleanup."""
        self._cancel_trainer_with_grace(job_id)
        try:
            self._jobs.remove(job_id)
        except ValueError:
            pass

    def delete_trainer(self, job_id: str) -> None:
        """Backward-compatible alias for ``cancel_trainer``."""
        self.cancel_trainer(job_id)

    def deployment(self, dep_id: str, action: str = "delete") -> None:
        """Register a deployment for cleanup on exit.

        *action*: ``"delete"`` (default) or ``"scale_to_zero"``.
        """
        self._deployments.append((dep_id, action))

    def __enter__(self) -> ResourceCleanup:
        return self

    def _cancel_trainer(self, job_id: str) -> None:
        cancel = getattr(self._rlor_mgr, "cancel", None)
        if callable(cancel):
            cancel(job_id)
            return

        post = getattr(self._rlor_mgr, "_post", None)
        account_id = getattr(self._rlor_mgr, "account_id", None)
        if callable(post) and account_id is not None:
            path = f"/v1/accounts/{account_id}/rlorTrainerJobs/{job_id}:cancel"
            resp = post(path, timeout=60)
            resp.raise_for_status()
            return

        raise AttributeError(
            "TrainerJobManager does not expose trainer cancellation support"
        )

    def _cancel_trainer_with_grace(self, job_id: str) -> None:
        if self._trainer_cancel_grace_period_s > 0:
            logger.info(
                "Cleanup: waiting %.1fs before canceling trainer job %s",
                self._trainer_cancel_grace_period_s,
                job_id,
            )
            time.sleep(self._trainer_cancel_grace_period_s)
        logger.info("Cleanup: canceling trainer job %s", job_id)
        self._cancel_trainer(job_id)

    def __exit__(self, *exc) -> None:
        for jid in reversed(self._jobs):
            try:
                self._cancel_trainer_with_grace(jid)
            except Exception as e:
                logger.warning("Cleanup: failed to cancel trainer %s: %s", jid, e)
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


StatusCallback = Any
"""``Callable[[str], None]`` -- invoked with a human-readable provisioning
status message at each lifecycle step (creating, waiting, ready, failed).
Declared as ``Any`` to avoid import-time coupling with callers."""


def _fetch_job_failure_reason(
    rlor_mgr: TrainerJobManager,
    job_id: str,
) -> str | None:
    """Try to retrieve a detailed failure reason from the server.

    Returns a human-readable string when the job has error details,
    ``None`` when the server doesn't expose them or the fetch fails.
    """
    try:
        job = rlor_mgr.get(job_id)
        state = job.get("state", "")
        reason_parts: list[str] = []
        if state:
            reason_parts.append(f"state={state}")
        error_msg = (
            job.get("error", {}).get("message")
            or job.get("errorMessage")
            or job.get("statusMessage")
        )
        if error_msg:
            reason_parts.append(error_msg)
        return "; ".join(reason_parts) if reason_parts else None
    except Exception:
        return None


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
    on_status: StatusCallback | None = None,
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

    *on_status* is an optional callback invoked with a human-readable
    string at each provisioning milestone (e.g. ``"creating trainer job"``).
    Recipes can use it to write fine-grained progress to ``RunnerIO``.
    """
    trainer_role = "reference" if forward_only else "policy"

    def _emit(msg: str) -> None:
        if on_status is not None:
            try:
                on_status(msg)
            except Exception:
                pass

    if job_id:
        if base_url_override:
            job_name = f"accounts/{rlor_mgr.account_id}/rlorTrainerJobs/{job_id}"
            logger.info(
                "Using pre-created %s trainer job %s at %s",
                trainer_role,
                job_id,
                base_url_override,
            )
            _emit(f"using pre-created {trainer_role} trainer {job_id}")
            return TrainerServiceEndpoint(
                job_name=job_name,
                job_id=job_id,
                base_url=base_url_override,
            )
        _emit(f"reusing existing {trainer_role} trainer {job_id}")
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
            skip_validations=infra.skip_validations,
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

    if infra.purpose:
        config.purpose = infra.purpose

    config.managed_by = infra.managed_by

    logger.info(
        "Creating %s trainer job '%s' (forward_only=%s)...",
        trainer_role,
        display_name,
        forward_only,
    )
    _emit(f"creating {trainer_role} trainer '{display_name}'")

    created_job_id: str | None = None
    try:
        created_job = rlor_mgr.create(config)
        created_job_id = created_job.job_id
        if cleanup:
            cleanup.trainer(created_job.job_id)
        _emit(
            f"waiting for {trainer_role} trainer '{display_name}' "
            f"to become ready (job {created_job.job_id})"
        )
        endpoint = rlor_mgr.wait_for_ready(
            created_job.job_id,
            job_name=created_job.job_name,
            timeout_s=infra.trainer_timeout_s,
        )
    except Exception as e:
        detail = str(e)
        if created_job_id:
            server_reason = _fetch_job_failure_reason(rlor_mgr, created_job_id)
            if server_reason:
                detail = f"{detail} — server: {server_reason}"
        logger.error(
            "Failed to create %s trainer job '%s' (forward_only=%s): %s",
            trainer_role,
            display_name,
            forward_only,
            detail,
        )
        error_msg = (
            f"Failed to create {trainer_role} trainer job '{display_name}' "
            f"(forward_only={forward_only}): {detail}"
        )
        _emit(error_msg)
        raise RuntimeError(error_msg) from e

    logger.info(
        "Created %s trainer job '%s': %s",
        trainer_role,
        display_name,
        endpoint.job_id,
    )
    _emit(f"{trainer_role} trainer '{display_name}' is ready (job {endpoint.job_id})")
    return endpoint


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
        if dep_config.region is None and dep_config.deployment_shape:
            dep_config.region = _infer_region_from_deployment_shape(
                deploy_mgr, dep_config.deployment_shape
            )
        if dep_config.region is None:
            info = _create_deployment_via_cookbook(deploy_mgr, dep_config, purpose=infra.purpose)
        else:
            info = deploy_mgr.create_or_get(dep_config)

    if info.state not in ("READY", "UPDATING"):
        info = deploy_mgr.wait_for_ready(
            deploy_cfg.deployment_id,
            timeout_s=deploy_cfg.deployment_timeout_s,
        )
    return info


def setup_or_reattach_deployment(
    deploy_mgr: DeploymentManager,
    deploy_cfg: DeployConfig,
    base_model: str,
    infra: InfraConfig,
    trainer_job_name: str,
    weight_syncer: "WeightSyncer | None" = None,
    reattach_settle_timeout_s: int = 600,
) -> DeploymentInfo:
    """Set up a deployment, re-attaching to a new trainer if it already exists.

    If ``deploy_cfg.deployment_id`` names a live deployment (not FAILED /
    DELETED / DELETING), its hotload bucket is re-pointed to
    *trainer_job_name* and the function blocks until the rolling restart
    settles. Otherwise a fresh deployment is created with the trainer
    reference baked in at creation.

    The replica-identity tracking and rolling-restart wait are handled by
    ``DeploymentManager.reattach_trainer()`` in the SDK.

    Caller responsibility: do not invoke this while a hotload is in progress
    on the same deployment.
    """
    existing = (
        deploy_mgr.get(deploy_cfg.deployment_id)
        if deploy_cfg.deployment_id
        else None
    )
    if not existing or existing.state in ("FAILED", "DELETED", "DELETING"):
        deploy_cfg.hot_load_trainer_job = trainer_job_name
        return setup_deployment(deploy_mgr, deploy_cfg, base_model, infra)

    deploy_mgr.reattach_trainer(
        deployment_id=deploy_cfg.deployment_id,
        trainer_job_name=trainer_job_name,
        base_model=base_model,
        timeout_s=reattach_settle_timeout_s,
    )
    if weight_syncer is not None:
        weight_syncer.reset_delta_chain()
    return existing


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


def get_deployment_gpu_count(
    deploy_mgr: DeploymentManager,
    deploy_cfg: "DeployConfig",
) -> int:
    """Return total GPU count (accelerator_count × replica_count) for a deployment.

    Reads accelerator_count from the deployment shape snapshot.
    Falls back to 1 if the shape doesn't expose it.
    """
    replica_count = deploy_cfg.replica_count or 1
    if not deploy_cfg.deployment_shape:
        return replica_count  # no shape, assume 1 GPU per replica

    try:
        version = _get_deployment_shape_version(deploy_mgr, deploy_cfg.deployment_shape)
        snapshot = version.get("snapshot", {}) or {}
        accel_count = snapshot.get("acceleratorCount", 1) or 1
        total = accel_count * replica_count
        logger.info(
            "Deployment GPU count: %d (accelerator_count=%d × replica_count=%d)",
            total, accel_count, replica_count,
        )
        return total
    except Exception as e:
        logger.warning("Could not determine GPU count from shape: %s", e)
        return replica_count


def _create_deployment_via_cookbook(
    deploy_mgr: DeploymentManager,
    config: DeploymentConfig,
    purpose: str | None = None,
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
    if purpose:
        body["annotations"] = {
            "internal/purpose": purpose.removeprefix("PURPOSE_").lower(),
        }

    logger.info("Creating deployment: %s", config.deployment_id)
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
