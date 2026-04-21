"""Infrastructure setup utilities: RLOR jobs, deployments, training clients."""

from __future__ import annotations

import os
import threading
import time
import logging
from typing import TYPE_CHECKING, Any, Union
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
    from fireworks.training.sdk.trainer import TrainingShapeProfile, CreatedTrainerJob
except ImportError:
    try:
        from fireworks.training.sdk import TrainingShapeProfile, CreatedTrainerJob
    except ImportError:
        from fireworks.training.sdk import TrainingShapeProfile
        CreatedTrainerJob = None  # type: ignore[assignment,misc]
from fireworks.training.sdk.deployment import DeploymentConfig, DeploymentInfo, DeploymentManager
from training.utils.config import InfraConfig, DeployConfig

if TYPE_CHECKING:
    from fireworks.training.sdk.weight_syncer import WeightSyncer

# Union of handles returned by request_trainer_job:
# CreatedTrainerJob for a freshly-POSTed job, TrainerServiceEndpoint for the reuse path.
TrainerHandle = Any

logger = logging.getLogger(__name__)

_DEPLOYMENT_ACCELERATOR_REGION_PREFIXES: tuple[tuple[str, str], ...] = (
    ("NVIDIA_H200", "US_VIRGINIA_1"),
    ("NVIDIA_B200", "US_OHIO_1"),
    ("NVIDIA_B300", "NA_BRITISHCOLUMBIA_1"),
)

_TRAINER_CANCEL_GRACE_ENV = "FW_TRAINER_CANCEL_GRACE_PERIOD_S"
_TRAINER_DELETE_GRACE_ENV = "FW_TRAINER_DELETE_GRACE_PERIOD_S"

_FIREWORKS_API_EXTRA_HEADERS_ENV = "FIREWORKS_API_EXTRA_HEADERS"


def read_api_extra_headers_env() -> dict[str, str] | None:
    """Parse ``FIREWORKS_API_EXTRA_HEADERS`` into a header dict.

    The env var, when set, must be a JSON object whose values are strings.
    Used to pass additional HTTP headers (e.g. routing, auth, correlation IDs)
    to every Fireworks API request. Returns ``None`` when the var is unset or
    empty.
    """
    raw = os.environ.get(_FIREWORKS_API_EXTRA_HEADERS_ENV, "").strip()
    if not raw:
        return None
    import json as _json

    try:
        parsed = _json.loads(raw)
    except _json.JSONDecodeError as exc:
        logger.warning(
            "Invalid %s (not valid JSON); ignoring: %s",
            _FIREWORKS_API_EXTRA_HEADERS_ENV,
            exc,
        )
        return None
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
    ):
        logger.warning(
            "%s must be a JSON object of string->string; ignoring",
            _FIREWORKS_API_EXTRA_HEADERS_ENV,
        )
        return None
    return parsed or None


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
        self._lock = threading.Lock()
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
        with self._lock:
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
        with self._lock:
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


def request_trainer_job(
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
    cleanup: ResourceCleanup | None = None,
    on_status: StatusCallback | None = None,
) -> TrainerHandle:
    """POST a trainer job creation request; return immediately without waiting for READY.

    Returns a ``CreatedTrainerJob`` for a freshly-POSTed job (callers must
    pass the result to :func:`wait_trainer_job` to block until READY), or a
    ``TrainerServiceEndpoint`` for the reuse path (already polled by
    ``_reuse_or_resume_job``, so :func:`wait_trainer_job` is a no-op).

    Two launch paths:

    * **Shape path** (profile provided): sends ``training_shape_ref``
      plus algorithm fields only.  The backend populates accelerator,
      image tag, node count, sharding, etc. from the validated shape.
    * **Manual path** (no profile): sends all ``InfraConfig`` fields
      as-is; the server skips shape validation.

    *cleanup*, when provided, registers the created job for cancellation
    on scope exit. Thread-safe: ``ResourceCleanup.trainer`` uses a lock.
    """
    trainer_role = "reference" if forward_only else "policy"

    def _emit(msg: str) -> None:
        if on_status is not None:
            try:
                on_status(msg)
            except Exception:
                pass

    if job_id:
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

    try:
        created_job = rlor_mgr.create(config)
    except Exception as e:
        error_msg = (
            f"Failed to create {trainer_role} trainer job '{display_name}' "
            f"(forward_only={forward_only}): {e}"
        )
        logger.error(error_msg)
        _emit(error_msg)
        raise RuntimeError(error_msg) from e

    if cleanup:
        cleanup.trainer(created_job.job_id)

    _emit(
        f"waiting for {trainer_role} trainer '{display_name}' "
        f"to become ready (job {created_job.job_id})"
    )
    return created_job


def wait_trainer_job(
    rlor_mgr: TrainerJobManager,
    created: TrainerHandle,
    *,
    infra: InfraConfig,
    display_name: str = "trainer",
    forward_only: bool = False,
    on_status: StatusCallback | None = None,
) -> TrainerServiceEndpoint:
    """Wait for a previously-requested trainer job to become READY.

    When *created* is already a ``TrainerServiceEndpoint`` (reuse path),
    returns it unchanged — ``_reuse_or_resume_job`` already polled.

    May be called from worker threads; the *on_status* callback and logging
    are both thread-safe.
    """
    if isinstance(created, TrainerServiceEndpoint):
        return created

    trainer_role = "reference" if forward_only else "policy"

    def _emit(msg: str) -> None:
        if on_status is not None:
            try:
                on_status(msg)
            except Exception:
                pass

    created_job_id = created.job_id
    try:
        endpoint = rlor_mgr.wait_for_ready(
            created.job_id,
            job_name=created.job_name,
            timeout_s=infra.trainer_timeout_s,
        )
    except Exception as e:
        detail = str(e)
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
    cleanup: ResourceCleanup | None = None,
    on_status: StatusCallback | None = None,
) -> TrainerServiceEndpoint:
    """Create and wait for a trainer job. Thin wrapper kept for backward compatibility."""
    created = request_trainer_job(
        rlor_mgr,
        base_model=base_model,
        infra=infra,
        profile=profile,
        lora_rank=lora_rank,
        max_seq_len=max_seq_len,
        learning_rate=learning_rate,
        grad_accum=grad_accum,
        display_name=display_name,
        hot_load_deployment_id=hot_load_deployment_id,
        extra_args=extra_args,
        job_id=job_id,
        forward_only=forward_only,
        cleanup=cleanup,
        on_status=on_status,
    )
    return wait_trainer_job(
        rlor_mgr,
        created,
        infra=infra,
        display_name=display_name,
        forward_only=forward_only,
        on_status=on_status,
    )


def request_deployment(
    deploy_mgr: DeploymentManager,
    deploy_cfg: DeployConfig,
    base_model: str,
    infra: InfraConfig,
) -> DeploymentInfo:
    """POST the deployment creation request; return immediately without waiting for READY.

    Auto-generates ``deploy_cfg.deployment_id`` when unset. Returns the
    post-create :class:`DeploymentInfo` (typically ``state=CREATING``).
    Callers pass the result to :func:`wait_deployment` to block until READY.
    """
    if not deploy_cfg.deployment_id:
        model_short = base_model.rsplit("/", 1)[-1]
        deploy_cfg.deployment_id = f"{model_short}-{int(time.time())}"

    info = deploy_mgr.get(deploy_cfg.deployment_id)
    if info:
        return info

    dep_config = deploy_cfg.to_deployment_config(base_model, infra)
    if dep_config.region is None and dep_config.deployment_shape:
        dep_config.region = _infer_region_from_deployment_shape(
            deploy_mgr, dep_config.deployment_shape
        )
    if dep_config.region is None:
        return _create_deployment_via_cookbook(deploy_mgr, dep_config, purpose=infra.purpose)
    return deploy_mgr.create_or_get(dep_config)


def wait_deployment(
    deploy_mgr: DeploymentManager,
    info: DeploymentInfo,
    deploy_cfg: DeployConfig,
) -> DeploymentInfo:
    """Wait for a previously-requested deployment to become READY.

    When *info* is already READY or UPDATING, returns it unchanged.
    May be called from worker threads; SDK polling and logging are thread-safe.
    """
    if info.state not in ("READY", "UPDATING"):
        info = deploy_mgr.wait_for_ready(
            deploy_cfg.deployment_id,
            timeout_s=deploy_cfg.deployment_timeout_s,
        )
    return info


def setup_deployment(
    deploy_mgr: DeploymentManager,
    deploy_cfg: DeployConfig,
    base_model: str,
    infra: InfraConfig,
) -> DeploymentInfo:
    """Set up an inference deployment (create + wait). Thin wrapper kept for backward compatibility.

    * If ``deploy_cfg.deployment_id`` is set, the existing deployment is
      fetched and waited on until READY.
    * If ``deploy_cfg.deployment_id`` is ``None``, a new deployment is
      created with an auto-generated ID derived from the model name.
    """
    info = request_deployment(deploy_mgr, deploy_cfg, base_model, infra)
    return wait_deployment(deploy_mgr, info, deploy_cfg)


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
    *trainer_job_name* via a PATCH. Otherwise a fresh deployment is created
    with the trainer reference baked in at creation.

    When *weight_syncer* is provided and a re-attach happens, its delta-chain
    state is reset so the next checkpoint is treated as ``base`` (the new
    trainer's bucket has no prior snapshots).

    PATCHing ``hot_load_trainer_job`` causes the serving container to be
    rolled (the watch dir is in container args, so the new value only takes
    effect on a fresh pod). This function blocks until the new pod's
    hotload manager exposes itself, so callers can immediately follow up
    with a hotload without racing against the rolling restart.

    Caller responsibility: do not invoke this while a hotload is in progress
    on the same deployment. Switching the bucket URL mid-load would leave
    the serving container in an undefined state. Server-side enforcement is
    tracked in fw-ai/fireworks#21732.
    """
    existing = (
        deploy_mgr.get(deploy_cfg.deployment_id)
        if deploy_cfg.deployment_id
        else None
    )
    if not existing or existing.state in ("FAILED", "DELETED", "DELETING"):
        deploy_cfg.hot_load_trainer_job = trainer_job_name
        return setup_deployment(deploy_mgr, deploy_cfg, base_model, infra)

    # Capture the previous replica's pod identity before PATCH so we can
    # detect when the rolling restart has produced a fresh pod.
    prev_identity = _read_replica_identity(
        deploy_mgr, deploy_cfg.deployment_id, base_model
    )

    deploy_mgr.update(
        deploy_cfg.deployment_id,
        body={"hotLoadTrainerJob": trainer_job_name},
        update_mask="hot_load_trainer_job",
    )
    logger.info(
        "Re-attached deployment %s to trainer %s (prev_pod=%s)",
        deploy_cfg.deployment_id,
        trainer_job_name,
        prev_identity,
    )
    _wait_for_reattach_settled(
        deploy_mgr,
        deploy_cfg.deployment_id,
        base_model,
        prev_identity=prev_identity,
        timeout_s=reattach_settle_timeout_s,
    )
    if weight_syncer is not None:
        weight_syncer.reset_delta_chain()
        # Force the syncer to re-run its one-time deployment-state check on
        # the next hotload. This re-validates that the *new* pod's hotload
        # manager is up and clears any stale base_identity bookkeeping.
        weight_syncer._deployment_checked = False  # noqa: SLF001
    return existing


def _read_replica_identity(
    deploy_mgr: DeploymentManager,
    deployment_id: str,
    base_model: str,
) -> str | None:
    """Return the first replica's pod identity from the hotload status, or None."""
    try:
        status = deploy_mgr.hotload_check_status(deployment_id, base_model)
    except Exception as e:  # noqa: BLE001
        logger.debug("hotload_check_status failed for identity probe: %s", e)
        return None
    replicas = status.get("replicas") or []
    if not replicas:
        return None
    return replicas[0].get("identity")


def _wait_for_reattach_settled(
    deploy_mgr: DeploymentManager,
    deployment_id: str,
    base_model: str,
    *,
    prev_identity: str | None,
    timeout_s: int,
    poll_interval_s: float = 5.0,
) -> None:
    """Block until a fresh pod is exposing the hotload manager.

    PATCHing ``hot_load_trainer_job`` triggers a rolling restart of the
    serving container. We need to wait for two things:

    1. The old pod has gone (its replica identity is no longer reported).
    2. A new pod has come up *and* its hotload manager is initialized
       (the replicas list is non-empty with a different identity).

    Without this wait the SDK's first post-reattach hotload polls the
    *old* pod (still ready, but about to die) or polls during the gap
    where no replica is reporting, and bails out with stage=error.
    """
    deadline = time.time() + max(timeout_s, 1)
    saw_pod_gone = prev_identity is None
    while time.time() < deadline:
        current = _read_replica_identity(deploy_mgr, deployment_id, base_model)
        if prev_identity is None:
            if current is not None:
                logger.info(
                    "Re-attach settled: hotload manager up on pod %s",
                    current,
                )
                return
        else:
            if current is None:
                if not saw_pod_gone:
                    logger.info(
                        "Old pod %s has gone; waiting for new pod...",
                        prev_identity,
                    )
                saw_pod_gone = True
            elif current != prev_identity:
                logger.info(
                    "Re-attach settled: new pod %s replaced %s",
                    current,
                    prev_identity,
                )
                return
        time.sleep(poll_interval_s)
    raise TimeoutError(
        f"Re-attach for deployment {deployment_id!r} did not produce a "
        f"fresh pod within {timeout_s}s (prev_identity={prev_identity!r})."
    )


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
