"""Infrastructure setup utilities: RLOR jobs, deployments, training clients.

One high-level entry point: :func:`setup_infra`. Recipes pass a config + a
couple of flags (``needs_reference``, ``needs_inference``) and get back an
:class:`Infra` bundle of trainer clients. All control flow — shape
auto-selection, parallel trainer provisioning, LoRA shared-reference
branching, deployment setup — lives inside :func:`setup_infra`.

Inference-side objects (sampler, weight syncer, concurrency controller) are
intentionally *not* created here — each recipe builds them after calling
:func:`setup_infra` so the construction is visible and customisable in the
loop. Use :func:`build_concurrency_controller` for the controller.

Generic across loop types:

* RL (``rl_loop.py``)   → ``needs_reference=(kl_beta > 0)``, ``needs_inference=True``
* DPO (``dpo_loop.py``) → ``needs_reference=True``, ``needs_inference=False``
* SFT                   → ``needs_reference=False``, ``needs_inference=False``
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlencode

from fireworks.training.sdk.client import (
    FiretitanServiceClient,
    FiretitanTrainingClient,
)
from fireworks.training.sdk import (
    TrainerJobConfig,
    TrainerJobManager,
    TrainerServiceEndpoint,
    TrainingShapeProfile,
    CreatedTrainerJob,
)
from fireworks.training.sdk.deployment import (
    DeploymentConfig,
    DeploymentInfo,
    DeploymentManager,
)
from fireworks.training.sdk.weight_syncer import WeightSyncer
from training.utils.config import InfraConfig, DeployConfig, WeightSyncScope
from training.utils.training_shapes import auto_select_training_shape
from training.utils.client import ReconnectableClient, DEFAULT_TIMEOUT_S

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
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
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

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
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


# ---------------------------------------------------------------------------
# Bundled infrastructure setup (setup_infra) — generic across RL, DPO, SFT
# ---------------------------------------------------------------------------


@dataclass
class _ReattachHandle:
    """Pending re-attach: PATCH has been issued; pod-roll settle is still pending.

    *dep_info* is the existing ``DeploymentInfo`` captured before the PATCH.
    *prev_identity* is the replica identity before the PATCH so
    :func:`_wait_for_reattach_settled` can detect the pod roll.
    *deployment_id* mirrors ``dep_info`` for convenience.
    *timeout_s* is forwarded to the settle function.
    """

    dep_info: DeploymentInfo
    prev_identity: str | None
    deployment_id: str
    timeout_s: int


@dataclass
class Infra:
    """Wired-up infrastructure handed back to the recipe.

    Inference-side objects (sampler, weight syncer, concurrency controller)
    are *not* included — each recipe constructs them after calling
    :func:`setup_infra` so they remain visible and customisable in the loop.

    Attributes:
        policy: Policy trainer client. Always set.
        reference: Reference trainer client when ``needs_reference``;
            either a separate :class:`ReconnectableClient` (full-param)
            or a base-only handle on the policy session (LoRA shared).
            ``None`` when no reference is needed.
        policy_profile: Resolved policy training-shape profile.
        policy_job_id: Policy trainer job ID.
        reference_job_id: Reference trainer job ID, or ``None`` when no
            separate reference trainer was provisioned (LoRA shared or
            ``needs_reference=False``).
        inference_model: Base model name for inference (may differ from
            ``cfg.base_model`` when a deployment adapter is mounted).
            ``None`` when ``needs_inference=False``.
        boot_metrics: One-shot boot metrics suitable for ``wandb_log``.
        closeables: Objects with ``.close()`` the caller should register
            onto an ``ExitStack``.
    """

    policy: Any
    reference: Any | None
    policy_profile: Any
    policy_job_id: str
    reference_job_id: str | None
    inference_model: str | None
    boot_metrics: dict
    closeables: list[Any] = field(default_factory=list)
    max_seq_len: int | None = None
    """Resolved max sequence length (auto-populated from shape when not provided)."""
    training_shape_id: str | None = None
    """Resolved policy training-shape ID (auto-selected if InfraConfig.training_shape_id was None)."""
    ref_training_shape_id: str | None = None
    """Resolved reference training-shape ID (auto-selected for full-param + needs_reference)."""
    deployment_id: str | None = None
    """Final deployment ID, including auto-generated ones when DeployConfig.deployment_id was unset."""


def setup_infra(
    *,
    rlor_mgr: TrainerJobManager,
    deploy_mgr: DeploymentManager | None = None,
    base_model: str,
    infra: InfraConfig,
    deploy_cfg: DeployConfig | None = None,
    lora_rank: int = 0,
    max_seq_len: int | None = None,
    learning_rate: float = 1e-5,
    step_timeout: float | None = None,
    policy_job_id: str | None = None,
    reference_job_id: str | None = None,
    needs_reference: bool = False,
    needs_inference: bool = False,
    role_prefix: str,
    api_key: str,
    cleanup: ResourceCleanup | None = None,
    on_status: Callable[[str], None] | None = None,
) -> Infra:
    """Build all training-side infrastructure in one call.

    Generic across loop types (RL, DPO, SFT).

    Args:
        rlor_mgr: Trainer job manager.
        deploy_mgr: Deployment manager. Required when ``needs_inference=True``.
        base_model: Base model resource name.
        infra: Infrastructure config (shape IDs, region, etc.). Never mutated.
        deploy_cfg: Deployment config. Required when ``needs_inference=True``. Never mutated.
        lora_rank: LoRA rank; 0 means full-parameter.
        max_seq_len: Max sequence length. Auto-populated from shape when ``None``;
            exposed on the returned :class:`Infra` as ``max_seq_len``.
        learning_rate: Optimizer learning rate forwarded to trainer jobs.
        step_timeout: Per-step RPC timeout. Falls back to SDK default when ``None``.
        policy_job_id: Reuse an existing policy trainer job instead of creating one.
        reference_job_id: Reuse an existing reference trainer job.
        needs_reference: Provision a reference trainer (or shared LoRA session).
        needs_inference: Provision an inference deployment.
        role_prefix: Display-name prefix for trainer jobs (e.g. ``"grpo"``, ``"dpo"``).
        api_key: Fireworks API key forwarded to trainer clients.
        cleanup: When provided, register created trainers for cancellation and
            deployments for scale-to-zero on scope exit. Pass ``None`` to skip
            registration (e.g. when resuming and wanting resources to outlive the run).
        on_status: Optional callback receiving human-readable status messages.

    Returns:
        An :class:`Infra` bundle. Caller registers ``infra.closeables`` on an
        ``ExitStack``.
    """
    if needs_inference and deploy_mgr is None:
        raise ValueError("deploy_mgr is required when needs_inference=True")

    emit: Callable[[str], None] = on_status or (lambda _: None)
    boot_start = time.time()

    policy_profile, resolved_max_seq_len, resolved_shape_id, resolved_deploy_shape = (
        _resolve_policy_shape(
            rlor_mgr,
            base_model=base_model,
            training_shape_id=infra.training_shape_id,
            deploy_shape=deploy_cfg.deployment_shape if deploy_cfg else None,
            lora_rank=lora_rank,
            max_seq_len=max_seq_len,
            needs_inference=needs_inference,
        )
    )
    ref_profile, resolved_ref_shape_id = _resolve_reference_shape(
        rlor_mgr,
        base_model=base_model,
        ref_training_shape_id=infra.ref_training_shape_id,
        lora_rank=lora_rank,
        max_seq_len=resolved_max_seq_len,
        needs_reference=needs_reference,
    )
    logger.info(
        "Policy shape=%s  ref shape=%s  deployment_shape=%s",
        resolved_shape_id,
        resolved_ref_shape_id,
        resolved_deploy_shape if needs_inference else None,
    )

    # Build an internal InfraConfig carrying the resolved shape IDs (for
    # request_trainer_job / wait_trainer_job which read infra.trainer_timeout_s etc.)
    resolved_infra = dataclasses.replace(
        infra,
        training_shape_id=resolved_shape_id,
        ref_training_shape_id=resolved_ref_shape_id,
    )
    # Build a local deploy_cfg copy with deployment_shape filled in; never mutates caller's object.
    local_deploy_cfg: DeployConfig | None = None
    if deploy_cfg is not None:
        local_deploy_cfg = (
            dataclasses.replace(deploy_cfg, deployment_shape=resolved_deploy_shape)
            if resolved_deploy_shape and not deploy_cfg.deployment_shape
            else deploy_cfg
        )

    weight_sync_scope = (
        local_deploy_cfg.weight_sync_scope if local_deploy_cfg else WeightSyncScope.PER_TRAINER
    )

    # Phase 2a (PER_DEPLOYMENT only): create/verify deployment before trainers so
    # we have the deployment ID available for hot_load_deployment_id on trainer creation.
    # For PER_TRAINER this is deferred to phase 2b so both POST in parallel.
    dep_info = None
    resolved_deployment_id = local_deploy_cfg.deployment_id if local_deploy_cfg else None
    if needs_inference and weight_sync_scope == WeightSyncScope.PER_DEPLOYMENT:
        dep_info, resolved_deployment_id = _request_deployment_or_none(
            deploy_mgr=deploy_mgr,
            deploy_cfg=local_deploy_cfg,
            base_model=base_model,
            infra=resolved_infra,
            policy_handle=None,
            weight_sync_scope=weight_sync_scope,
            cleanup=cleanup,
        )

    # Phase 2b: POST both trainer creation requests (fast, seconds).
    # PER_DEPLOYMENT passes resolved_deployment_id so each trainer registers
    # hot_load_deployment_id, making the deployment bucket stable across resumes.
    hot_load_dep_id = resolved_deployment_id if weight_sync_scope == WeightSyncScope.PER_DEPLOYMENT else None
    policy_handle, ref_handle = _request_trainers(
        rlor_mgr=rlor_mgr,
        base_model=base_model,
        infra=resolved_infra,
        policy_profile=policy_profile,
        ref_profile=ref_profile,
        lora_rank=lora_rank,
        max_seq_len=resolved_max_seq_len,
        learning_rate=learning_rate,
        policy_job_id=policy_job_id,
        reference_job_id=reference_job_id,
        role_prefix=role_prefix,
        hot_load_deployment_id=hot_load_dep_id,
        cleanup=cleanup,
        on_status=emit,
    )

    # Phase 2c: POST a fresh deployment or PATCH a re-attach for PER_TRAINER.
    # Skipped when needs_inference=False or PER_DEPLOYMENT already handled it.
    if needs_inference and weight_sync_scope == WeightSyncScope.PER_TRAINER:
        dep_info, resolved_deployment_id = _request_deployment_or_none(
            deploy_mgr=deploy_mgr,
            deploy_cfg=local_deploy_cfg,
            base_model=base_model,
            infra=resolved_infra,
            policy_handle=policy_handle,
            weight_sync_scope=weight_sync_scope,
            cleanup=cleanup,
        )

    # Phase 2d: Wait for all pending resources in parallel.
    policy_ep, reference_ep, ready_dep_info = _await_in_parallel(
        rlor_mgr=rlor_mgr,
        policy_handle=policy_handle,
        ref_handle=ref_handle,
        dep_info=dep_info,
        deploy_mgr=deploy_mgr if needs_inference else None,
        infra=resolved_infra,
        deploy_cfg=local_deploy_cfg,
        base_model=base_model,
        role_prefix=role_prefix,
        on_status=emit,
    )

    inference_model = None
    if needs_inference:
        inference_model = ready_dep_info.inference_model or base_model

    closeables: list[Any] = []
    policy = _make_policy_client(
        rlor_mgr, policy_ep,
        base_model=base_model,
        lora_rank=lora_rank,
        step_timeout=step_timeout,
        api_key=api_key,
        closeables=closeables,
    )
    reference = _make_reference_client(
        rlor_mgr=rlor_mgr,
        reference_ep=reference_ep,
        policy=policy,
        base_model=base_model,
        lora_rank=lora_rank,
        step_timeout=step_timeout,
        needs_reference=needs_reference,
        api_key=api_key,
        closeables=closeables,
    )

    boot_metrics = _make_boot_metrics(boot_start, deploy_mgr if needs_inference else None)

    return Infra(
        policy=policy,
        reference=reference,
        policy_profile=policy_profile,
        policy_job_id=policy_ep.job_id,
        reference_job_id=reference_ep.job_id if reference_ep is not None else None,
        inference_model=inference_model,
        boot_metrics=boot_metrics,
        closeables=closeables,
        max_seq_len=resolved_max_seq_len,
        training_shape_id=resolved_shape_id,
        ref_training_shape_id=resolved_ref_shape_id,
        deployment_id=resolved_deployment_id,
    )


def _register_closeable(obj: Any, closeables: list[Any]) -> None:
    if hasattr(obj, "close"):
        closeables.append(obj)


def _resolve_policy_shape(
    rlor_mgr: TrainerJobManager,
    *,
    base_model: str,
    training_shape_id: str | None,
    deploy_shape: str | None,
    lora_rank: int,
    max_seq_len: int | None,
    needs_inference: bool,
) -> tuple[Any, int, str | None, str | None]:
    """Return (profile, resolved_max_seq_len, resolved_shape_id, resolved_deploy_shape)."""
    if training_shape_id is None:
        training_shape_id = auto_select_training_shape(
            rlor_mgr,
            base_model=base_model,
            trainer_role="policy",
            lora_rank=lora_rank,
            max_seq_len=max_seq_len,
        )
        logger.info("Auto-selected policy training shape: %s", training_shape_id)
    profile = rlor_mgr.resolve_training_profile(training_shape_id)

    resolved_deploy_shape = deploy_shape
    if needs_inference and not deploy_shape and profile.deployment_shape_version:
        resolved_deploy_shape = profile.deployment_shape_version

    if max_seq_len is None:
        max_seq_len = profile.max_supported_context_length
    if max_seq_len is None:
        raise ValueError(
            "max_seq_len is required. Set it in Config, or use a training shape "
            "(InfraConfig.training_shape_id) to auto-populate it."
        )
    return profile, max_seq_len, training_shape_id, resolved_deploy_shape


def _resolve_reference_shape(
    rlor_mgr: TrainerJobManager,
    *,
    base_model: str,
    ref_training_shape_id: str | None,
    lora_rank: int,
    max_seq_len: int,
    needs_reference: bool,
) -> tuple[Any | None, str | None]:
    """Return (ref_profile, resolved_ref_shape_id)."""
    if ref_training_shape_id:
        return rlor_mgr.resolve_training_profile(ref_training_shape_id), ref_training_shape_id
    if not (needs_reference and lora_rank == 0):
        return None, None
    ref_training_shape_id = auto_select_training_shape(
        rlor_mgr,
        base_model=base_model,
        trainer_role="reference",
        lora_rank=lora_rank,
        max_seq_len=max_seq_len,
    )
    logger.info("Auto-selected reference training shape: %s", ref_training_shape_id)
    return rlor_mgr.resolve_training_profile(ref_training_shape_id), ref_training_shape_id


def _strip_args(infra: InfraConfig, flags: set[str]) -> InfraConfig:
    """Return a copy of *infra* with the named flags removed from extra_args."""
    if not infra.extra_args:
        return infra
    stripped = [a for a in infra.extra_args if a not in flags]
    if stripped == infra.extra_args:
        return infra
    return dataclasses.replace(infra, extra_args=stripped or None)


def _request_trainers(
    *,
    rlor_mgr: TrainerJobManager,
    base_model: str,
    infra: InfraConfig,
    policy_profile: Any,
    ref_profile: Any | None,
    lora_rank: int,
    max_seq_len: int,
    learning_rate: float,
    policy_job_id: str | None,
    reference_job_id: str | None,
    role_prefix: str,
    hot_load_deployment_id: str | None,
    cleanup: ResourceCleanup | None,
    on_status: Callable[[str], None],
) -> tuple[Any, Any | None]:
    on_status("provisioning policy trainer")
    policy_handle = request_trainer_job(
        rlor_mgr,
        base_model=base_model,
        infra=infra,
        profile=policy_profile,
        lora_rank=lora_rank,
        max_seq_len=max_seq_len,
        learning_rate=learning_rate,
        display_name=f"{role_prefix}-policy",
        forward_only=False,
        hot_load_deployment_id=hot_load_deployment_id,
        job_id=policy_job_id,
        cleanup=cleanup,
        on_status=on_status,
    )

    ref_handle = None
    if ref_profile is not None:
        on_status("provisioning reference trainer")
        # Reference trainer runs forward-only: strip --full-oom-check, which
        # triggers a backward warmup that OOMs on smaller reference shapes.
        ref_infra = _strip_args(infra, {"--full-oom-check"})
        ref_handle = request_trainer_job(
            rlor_mgr,
            base_model=base_model,
            infra=ref_infra,
            profile=ref_profile,
            lora_rank=0,
            max_seq_len=max_seq_len,
            learning_rate=learning_rate,
            display_name=f"{role_prefix}-reference",
            forward_only=True,
            hot_load_deployment_id=hot_load_deployment_id,
            job_id=reference_job_id,
            cleanup=cleanup,
            on_status=on_status,
        )

    return policy_handle, ref_handle


def _request_deployment_or_none(
    *,
    deploy_mgr: DeploymentManager,
    deploy_cfg: DeployConfig,
    base_model: str,
    infra: InfraConfig,
    policy_handle: Any,
    weight_sync_scope: WeightSyncScope,
    cleanup: ResourceCleanup | None,
) -> "tuple[DeploymentInfo | _ReattachHandle, str | None]":
    """POST a fresh deployment or handle re-attach; return immediately.

    ``PER_TRAINER``: sets ``hot_load_trainer_job`` on a fresh deployment, or
    PATCHes it on an existing one (triggering a pod rolling restart).

    ``PER_DEPLOYMENT``: deployment is wired at trainer creation time via
    ``hot_load_deployment_id``; this function only creates a fresh deployment
    when none exists yet, or returns the existing one unchanged.

    Returns ``(dep_info_or_handle, resolved_deployment_id)``.
    """
    dep_id = deploy_cfg.deployment_id
    timeout_s = getattr(deploy_cfg, "reattach_settle_timeout_s", None) or 600
    job_name = getattr(policy_handle, "job_name", None)

    if dep_id:
        existing = deploy_mgr.get(dep_id)
        if existing and existing.state not in ("FAILED", "DELETED", "DELETING"):
            if weight_sync_scope == WeightSyncScope.PER_DEPLOYMENT:
                # Trainer was created with hot_load_deployment_id pointing here;
                # the deployment keeps its existing bucket — nothing to patch.
                logger.info(
                    "Re-using deployment %s for deployment-first weight sync (trainer=%s)",
                    dep_id, job_name,
                )
                if cleanup is not None:
                    cleanup.deployment(dep_id, action="scale_to_zero")
                return existing, dep_id

            if job_name:
                # PER_TRAINER re-attach: PATCH deployment to point at new trainer.
                prev_identity = _read_replica_identity(deploy_mgr, dep_id, base_model)
                deploy_mgr.update(
                    dep_id,
                    body={"hotLoadTrainerJob": job_name},
                    update_mask="hot_load_trainer_job",
                )
                logger.info(
                    "Re-attached deployment %s to trainer %s (prev_pod=%s) — "
                    "settling in parallel with trainer waits",
                    dep_id, job_name, prev_identity,
                )
                if cleanup is not None:
                    cleanup.deployment(dep_id, action="scale_to_zero")
                return _ReattachHandle(
                    dep_info=existing,
                    prev_identity=prev_identity,
                    deployment_id=dep_id,
                    timeout_s=timeout_s,
                ), dep_id

    # Fresh deployment path — work on a local copy so we never mutate caller's object.
    local = dataclasses.replace(deploy_cfg)
    if weight_sync_scope == WeightSyncScope.PER_TRAINER and job_name:
        local = dataclasses.replace(local, hot_load_trainer_job=job_name)

    info = request_deployment(deploy_mgr, local, base_model, infra)
    resolved_dep_id = local.deployment_id
    if cleanup is not None:
        cleanup.deployment(resolved_dep_id, action="scale_to_zero")
    return info, resolved_dep_id


def _await_in_parallel(
    *,
    rlor_mgr: TrainerJobManager,
    policy_handle: Any,
    ref_handle: Any | None,
    dep_info: "DeploymentInfo | _ReattachHandle | None",
    deploy_mgr: "DeploymentManager | None",
    infra: InfraConfig,
    deploy_cfg: "DeployConfig | None",
    base_model: str,
    role_prefix: str,
    on_status: Callable[[str], None],
) -> tuple[Any, Any | None, "DeploymentInfo | None"]:
    """Wait for all pending resources in parallel using a thread pool."""
    max_workers = 1 + (1 if ref_handle is not None else 0) + (1 if dep_info is not None else 0)

    policy_ep = ref_ep = final_dep_info = None
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=max(max_workers, 1)) as pool:
        policy_future = pool.submit(
            wait_trainer_job,
            rlor_mgr,
            policy_handle,
            infra=infra,
            display_name=f"{role_prefix}-policy",
            forward_only=False,
            on_status=on_status,
        )
        ref_future = (
            pool.submit(
                wait_trainer_job,
                rlor_mgr,
                ref_handle,
                infra=infra,
                display_name=f"{role_prefix}-reference",
                forward_only=True,
                on_status=on_status,
            )
            if ref_handle is not None
            else None
        )
        if dep_info is None:
            dep_future = None
        elif isinstance(dep_info, _ReattachHandle):
            dep_future = pool.submit(
                _wait_for_reattach_settled,
                deploy_mgr,
                dep_info.deployment_id,
                base_model,
                prev_identity=dep_info.prev_identity,
                timeout_s=dep_info.timeout_s,
            )
        else:
            dep_future = pool.submit(wait_deployment, deploy_mgr, dep_info, deploy_cfg)

        try:
            policy_ep = policy_future.result()
        except Exception as e:
            errors.append(f"Policy trainer: {e}")

        if ref_future is not None:
            try:
                ref_ep = ref_future.result()
            except Exception as e:
                errors.append(f"Reference trainer: {e}")

        if dep_future is not None:
            try:
                settle_result = dep_future.result()
                final_dep_info = (
                    dep_info.dep_info
                    if isinstance(dep_info, _ReattachHandle)
                    else settle_result
                )
            except Exception as e:
                errors.append(f"Deployment: {e}")

    if errors:
        raise RuntimeError("Infrastructure provisioning failed:\n" + "\n".join(errors))
    assert policy_ep is not None
    return policy_ep, ref_ep, final_dep_info


def _make_policy_client(
    rlor_mgr: TrainerJobManager,
    policy_ep: Any,
    *,
    base_model: str,
    lora_rank: int,
    step_timeout: float | None,
    api_key: str,
    closeables: list[Any],
) -> Any:
    policy = ReconnectableClient(
        rlor_mgr, policy_ep.job_id, base_model,
        lora_rank=lora_rank,
        fw_api_key=api_key,
        default_timeout=step_timeout or DEFAULT_TIMEOUT_S,
    )
    _register_closeable(policy, closeables)
    return policy


def _make_reference_client(
    *,
    rlor_mgr: TrainerJobManager,
    reference_ep: Any | None,
    policy: Any,
    base_model: str,
    lora_rank: int,
    step_timeout: float | None,
    needs_reference: bool,
    api_key: str,
    closeables: list[Any],
) -> Any | None:
    if reference_ep is not None:
        reference = ReconnectableClient(
            rlor_mgr, reference_ep.job_id, base_model,
            lora_rank=0,
            fw_api_key=api_key,
            default_timeout=step_timeout or DEFAULT_TIMEOUT_S,
        )
        _register_closeable(reference, closeables)
        return reference
    if needs_reference and lora_rank > 0:
        reference = policy.create_base_reference()
        _register_closeable(reference, closeables)
        return reference
    return None


def _make_boot_metrics(boot_start: float, deploy_mgr: DeploymentManager | None) -> dict:
    metrics: dict = {
        "train/step": 0,
        "infra/total_boot_time": time.time() - boot_start,
    }
    if deploy_mgr is not None and deploy_mgr.boot_time_s is not None:
        metrics["infra/deploy_boot_time"] = deploy_mgr.boot_time_s
    return metrics


