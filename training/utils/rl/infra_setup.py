"""Bundled infrastructure setup for training recipes.

One entry point: :func:`setup_infra`. Recipes pass a config + a couple
of flags (``needs_reference``, ``needs_inference``) and get back an
:class:`Infra` bundle of fully-wired clients. All control flow —
shape auto-selection, parallel trainer provisioning, LoRA
shared-reference branching, deployment setup, sampler, weight syncer,
concurrency controller — lives inside :func:`setup_infra`.

Generic across loop types:

* RL (``rl_loop.py``)   → ``needs_reference=(kl_beta > 0)``, ``needs_inference=True``
* DPO (``dpo_loop.py``) → ``needs_reference=True``, ``needs_inference=False``
* SFT                   → ``needs_reference=False``, ``needs_inference=False``

LoRA optimisation: when ``cfg.lora_rank > 0`` and a reference is needed,
the policy trainer serves reference logprobs via a base-only handle on
the *same* trainer session (``policy.create_base_reference()``). No
second GPU job, and the policy LoRA is never unloaded.

The recipe still owns lifecycle: register ``infra.closeables`` onto an
``ExitStack``, and register the deployment on a ``ResourceCleanup`` if
``cleanup_on_exit`` is set.
"""

from __future__ import annotations

import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import transformers

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from fireworks.training.sdk.deployment import (
    AdaptiveConcurrencyController,
    DeploymentInfo,
    DeploymentSampler,
    FixedConcurrencyController,
)
from fireworks.training.sdk.weight_syncer import WeightSyncer

from training.utils import (
    ReconnectableClient,
    ResourceCleanup,
    auto_select_training_shape,
    get_deployment_gpu_count,
    request_deployment,
    request_trainer_job,
    wait_deployment,
    wait_trainer_job,
)
from training.utils.client import DEFAULT_TIMEOUT_S
from training.utils.infra import _read_replica_identity, _wait_for_reattach_settled

logger = logging.getLogger(__name__)

__all__ = ["Infra", "setup_infra"]


# Default concurrent requests per GPU when AdaptiveConcurrencyController's
# initial_window is unspecified.
_SLOTS_PER_GPU = 8


@dataclass
class _ReattachHandle:
    """Pending re-attach: PATCH has been issued; pod-roll settle is still pending.

    *dep_info* is the existing ``DeploymentInfo`` captured before the PATCH.
    *prev_identity* is the replica identity before the PATCH so
    :func:`_wait_for_reattach_settled` can detect the pod roll.
    *deployment_id* mirrors ``dep_info`` for convenience.
    *timeout_s* is forwarded to the settle function.
    """

    dep_info: "DeploymentInfo"
    prev_identity: str | None
    deployment_id: str
    timeout_s: int


@dataclass
class Infra:
    """Wired-up infrastructure handed back to the recipe.

    Attributes:
        policy: Policy trainer client. Always set.
        reference: Reference trainer client when ``needs_reference``;
            either a separate :class:`ReconnectableClient` (full-param)
            or a base-only handle on the policy session (LoRA shared).
            ``None`` when no reference is needed.
        sampler: Inference sampler. ``None`` when ``needs_inference=False``.
        weight_syncer: Save-and-hotload helper. ``None`` when no inference.
        concurrency_controller: Optional concurrency controller bound to
            the sampler. Recipe pulls it for ``step_completed()`` metrics.
        policy_profile: Resolved policy training-shape profile.
        policy_job_id: Policy trainer job ID.
        reference_job_id: Reference trainer job ID, or ``None`` when no
            separate reference trainer was provisioned (LoRA shared or
            ``needs_reference=False``).
        inference_model: Base model name for inference (may differ from
            ``cfg.base_model`` when a deployment adapter is mounted).
        boot_metrics: One-shot boot metrics suitable for ``wandb_log``.
        closeables: Objects with ``.close()`` the caller should register
            onto an ``ExitStack``.
    """

    policy: Any
    reference: Any | None
    sampler: Any | None
    weight_syncer: Any | None
    concurrency_controller: Any | None
    policy_profile: Any
    policy_job_id: str
    reference_job_id: str | None
    inference_model: str | None
    boot_metrics: dict
    closeables: list[Any] = field(default_factory=list)


def setup_infra(
    cfg: Any,
    *,
    rlor_mgr: TrainerJobManager,
    deploy_mgr: DeploymentManager | None = None,
    needs_reference: bool,
    needs_inference: bool = True,
    api_key: str,
    cleanup: ResourceCleanup | None = None,
    cleanup_on_exit: bool = False,
    on_status: Callable[[str], None] | None = None,
) -> Infra:
    """Build all training-side infrastructure in one call.

    Inputs (duck-typed on ``cfg``):

    Always required:
        ``base_model``, ``lora_rank``, ``max_seq_len`` (or auto from shape),
        ``learning_rate``, ``step_timeout``, ``infra`` (InfraConfig),
        ``policy_job_id`` / ``reference_job_id`` (set to reuse existing
        trainers).

    When ``needs_inference``:
        ``deployment`` (DeployConfig), ``weight_sync`` (WeightSyncConfig),
        ``concurrency`` (ConcurrencyConfig), ``rollout_base_model`` (optional).
        Plus ``deploy_mgr`` must be provided.

    Mutates ``cfg``:
        * ``cfg.infra.training_shape_id`` ← auto-selected when ``None``.
        * ``cfg.infra.ref_training_shape_id`` ← auto-selected when needed.
        * ``cfg.deployment.deployment_shape`` ← profile default when unset.
        * ``cfg.max_seq_len`` ← profile's max context when ``None``.

    Returns:
        An :class:`Infra` bundle. Caller registers ``infra.closeables`` and
        (if ``cleanup_on_exit``) the deployment scale-to-zero.
    """
    if needs_inference and deploy_mgr is None:
        raise ValueError("deploy_mgr is required when needs_inference=True")

    on_status_cb = on_status or _noop_status
    boot_start = time.time()

    policy_profile = _resolve_policy_shape(cfg, rlor_mgr, needs_inference)
    ref_profile = _resolve_reference_shape(cfg, rlor_mgr, needs_reference)
    logger.info(
        "Policy shape=%s  ref shape=%s  deployment_shape=%s",
        cfg.infra.training_shape_id,
        cfg.infra.ref_training_shape_id,
        getattr(cfg.deployment, "deployment_shape", None) if needs_inference else None,
    )

    trainer_cleanup = cleanup if cleanup_on_exit else None
    role_prefix = "grpo" if needs_inference else "dpo"

    # Phase 2a: POST both trainer creation requests (fast, seconds).
    policy_handle, ref_handle = _request_trainers(
        cfg=cfg,
        rlor_mgr=rlor_mgr,
        policy_profile=policy_profile,
        ref_profile=ref_profile,
        role_prefix=role_prefix,
        cleanup=trainer_cleanup,
        on_status=on_status_cb,
    )

    # Phase 2b: POST a fresh deployment or PATCH a re-attach — both using
    # policy_handle.job_name which is available immediately from CreatedTrainerJob
    # before the trainer is READY. Skipped when needs_inference=False.
    dep_info = None
    if needs_inference:
        dep_info = _request_deployment_or_none(
            cfg=cfg,
            deploy_mgr=deploy_mgr,
            policy_handle=policy_handle,
            cleanup=cleanup,
            cleanup_on_exit=cleanup_on_exit,
        )

    # Phase 2c: Wait for all pending resources in parallel.
    policy_ep, reference_ep, ready_dep_info = _await_in_parallel(
        rlor_mgr=rlor_mgr,
        policy_handle=policy_handle,
        ref_handle=ref_handle,
        dep_info=dep_info,
        deploy_mgr=deploy_mgr if needs_inference else None,
        cfg=cfg,
        role_prefix=role_prefix,
        on_status=on_status_cb,
    )

    inference_model = None
    if needs_inference:
        inference_model = ready_dep_info.inference_model or cfg.base_model

    closeables: list[Any] = []
    policy = _make_policy_client(cfg, rlor_mgr, policy_ep, api_key, closeables)
    reference = _make_reference_client(
        cfg=cfg,
        rlor_mgr=rlor_mgr,
        policy=policy,
        reference_ep=reference_ep,
        needs_reference=needs_reference,
        api_key=api_key,
        closeables=closeables,
    )

    sampler = None
    weight_syncer = None
    concurrency_controller = None
    if needs_inference:
        sampler, weight_syncer, concurrency_controller = _make_inference_components(
            cfg=cfg,
            deploy_mgr=deploy_mgr,
            policy=policy,
            inference_model=inference_model,
            api_key=api_key,
        )

    boot_metrics = _make_boot_metrics(boot_start, deploy_mgr if needs_inference else None)

    return Infra(
        policy=policy,
        reference=reference,
        sampler=sampler,
        weight_syncer=weight_syncer,
        concurrency_controller=concurrency_controller,
        policy_profile=policy_profile,
        policy_job_id=policy_ep.job_id,
        reference_job_id=reference_ep.job_id if reference_ep is not None else None,
        inference_model=inference_model,
        boot_metrics=boot_metrics,
        closeables=closeables,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _noop_status(_message: str) -> None:
    """No-op status callback used when caller passes ``on_status=None``."""


def _register_closeable(obj: Any, closeables: list[Any]) -> None:
    """Append ``obj`` to ``closeables`` if it has a ``.close()`` method."""
    if hasattr(obj, "close"):
        closeables.append(obj)


def _resolve_policy_shape(
    cfg: Any, rlor_mgr: TrainerJobManager, needs_inference: bool,
) -> Any:
    """Auto-select policy shape if unset; resolve to profile; auto-fill cfg."""
    if cfg.infra.training_shape_id is None:
        cfg.infra.training_shape_id = auto_select_training_shape(
            rlor_mgr,
            base_model=cfg.base_model,
            trainer_role="policy",
            lora_rank=cfg.lora_rank,
            max_seq_len=cfg.max_seq_len,
        )
        logger.info(
            "Auto-selected policy training shape: %s",
            cfg.infra.training_shape_id,
        )
    profile = rlor_mgr.resolve_training_profile(cfg.infra.training_shape_id)

    if (
        needs_inference
        and not cfg.deployment.deployment_shape
        and profile.deployment_shape_version
    ):
        cfg.deployment.deployment_shape = profile.deployment_shape_version
    if cfg.max_seq_len is None:
        cfg.max_seq_len = profile.max_supported_context_length
    if cfg.max_seq_len is None:
        raise ValueError(
            "max_seq_len is required. Set it in Config, or use a training shape "
            "(InfraConfig.training_shape_id) to auto-populate it."
        )
    return profile


def _resolve_reference_shape(
    cfg: Any, rlor_mgr: TrainerJobManager, needs_reference: bool,
) -> Any | None:
    """Resolve reference profile; ``None`` for no-ref or LoRA-shared paths.

    Returns ``None`` when:
      * ``not needs_reference`` (no reference at all), OR
      * ``needs_reference`` AND ``lora_rank > 0`` AND no explicit ref shape
        — the policy trainer will serve reference via base-only handle.

    Explicit ``cfg.infra.ref_training_shape_id`` is always honored.
    """
    if cfg.infra.ref_training_shape_id:
        return rlor_mgr.resolve_training_profile(cfg.infra.ref_training_shape_id)
    if not (needs_reference and cfg.lora_rank == 0):
        return None
    cfg.infra.ref_training_shape_id = auto_select_training_shape(
        rlor_mgr,
        base_model=cfg.base_model,
        trainer_role="reference",
        lora_rank=cfg.lora_rank,
        max_seq_len=cfg.max_seq_len,
    )
    logger.info(
        "Auto-selected reference training shape: %s",
        cfg.infra.ref_training_shape_id,
    )
    return rlor_mgr.resolve_training_profile(cfg.infra.ref_training_shape_id)


def _request_trainers(
    *,
    cfg: Any,
    rlor_mgr: TrainerJobManager,
    policy_profile: Any,
    ref_profile: Any | None,
    role_prefix: str,
    cleanup: ResourceCleanup | None,
    on_status: Callable[[str], None],
) -> tuple[Any, Any | None]:
    """POST trainer creation requests sequentially (fast, returns in seconds).

    Both requests are issued before any waiting begins so that the caller can
    immediately proceed to :func:`_request_deployment_or_none` and then
    :func:`_await_in_parallel`.

    Returns ``(policy_handle, ref_handle)``.  Each handle is either a
    ``CreatedTrainerJob`` (fresh job, needs :func:`wait_trainer_job`) or a
    ``TrainerServiceEndpoint`` (reuse path, already polled).
    """
    on_status("provisioning policy trainer")
    policy_handle = request_trainer_job(
        rlor_mgr,
        base_model=cfg.base_model,
        infra=cfg.infra,
        profile=policy_profile,
        lora_rank=cfg.lora_rank,
        max_seq_len=cfg.max_seq_len,
        learning_rate=cfg.learning_rate,
        display_name=f"{role_prefix}-policy",
        forward_only=False,
        job_id=cfg.policy_job_id,
        cleanup=cleanup,
        on_status=on_status,
    )

    ref_handle = None
    if ref_profile is not None:
        on_status("provisioning reference trainer")
        ref_handle = request_trainer_job(
            rlor_mgr,
            base_model=cfg.base_model,
            infra=cfg.infra,
            profile=ref_profile,
            lora_rank=0,  # reference trainers never carry LoRA
            max_seq_len=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            display_name=f"{role_prefix}-reference",
            forward_only=True,
            job_id=cfg.reference_job_id,
            cleanup=cleanup,
            on_status=on_status,
        )

    return policy_handle, ref_handle


def _request_deployment_or_none(
    *,
    cfg: Any,
    deploy_mgr: DeploymentManager,
    policy_handle: Any,
    cleanup: ResourceCleanup | None,
    cleanup_on_exit: bool,
) -> "DeploymentInfo | _ReattachHandle":
    """POST a fresh deployment or issue a re-attach PATCH; return immediately.

    Both paths use ``policy_handle.job_name`` which is available from
    ``CreatedTrainerJob`` before the trainer is READY, so this can run
    immediately after the trainer POST — no waiting required.

    **Fresh deployment**: creates a new deployment with the trainer's
    ``job_name`` baked in at creation time.

    **Re-attach path** (existing live deployment found): issues a
    ``hotLoadTrainerJob`` PATCH immediately, captures the current replica
    identity for settling, and returns a :class:`_ReattachHandle`.
    :func:`_await_in_parallel` will then run :func:`_wait_for_reattach_settled`
    in parallel with the trainer waits — no serial phase needed.

    Cleanup is registered here so the scope-exit handler fires even if a
    later step raises.
    """
    dep_id = cfg.deployment.deployment_id
    timeout_s = getattr(cfg.deployment, "deployment_timeout_s", None) or 600
    job_name = getattr(policy_handle, "job_name", None)

    if dep_id:
        existing = deploy_mgr.get(dep_id)
        if existing and existing.state not in ("FAILED", "DELETED", "DELETING"):
            if job_name:
                # Re-attach: PATCH now with job_name (no need to wait for trainer READY).
                prev_identity = _read_replica_identity(deploy_mgr, dep_id, cfg.base_model)
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
                if cleanup_on_exit and cleanup is not None:
                    cleanup.deployment(dep_id, action="scale_to_zero")
                return _ReattachHandle(
                    dep_info=existing,
                    prev_identity=prev_identity,
                    deployment_id=dep_id,
                    timeout_s=timeout_s,
                )
            # job_name unavailable (SDK bug) — fall through to fresh create.

    # Fresh deployment: bake job_name at creation time.
    if job_name:
        cfg.deployment.hot_load_trainer_job = job_name

    info = request_deployment(deploy_mgr, cfg.deployment, cfg.base_model, cfg.infra)
    if cleanup_on_exit and cleanup is not None:
        cleanup.deployment(cfg.deployment.deployment_id, action="scale_to_zero")
    return info


def _await_in_parallel(
    *,
    rlor_mgr: TrainerJobManager,
    policy_handle: Any,
    ref_handle: Any | None,
    dep_info: "DeploymentInfo | _ReattachHandle | None",
    deploy_mgr: "DeploymentManager | None",
    cfg: Any,
    role_prefix: str,
    on_status: Callable[[str], None],
) -> tuple[Any, Any | None, "DeploymentInfo | None"]:
    """Wait for all pending resources in parallel using a thread pool.

    Submits up to three futures (policy wait, optional reference wait,
    optional deployment wait or re-attach settle) and collects all results.
    On any failure, all errors are gathered and re-raised together so
    operators see the full picture.

    For :class:`_ReattachHandle`, runs :func:`_wait_for_reattach_settled`
    in parallel with the trainer waits (the PATCH was already issued in
    :func:`_request_deployment_or_none`).

    *on_status* callbacks may fire from worker threads; callers only
    print/log, both of which are thread-safe.

    Returns ``(policy_ep, ref_ep, final_dep_info)``.
    """
    max_workers = 1 + (1 if ref_handle is not None else 0) + (1 if dep_info is not None else 0)

    policy_ep = ref_ep = final_dep_info = None
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=max(max_workers, 1)) as pool:
        policy_future = pool.submit(
            wait_trainer_job,
            rlor_mgr,
            policy_handle,
            infra=cfg.infra,
            display_name=f"{role_prefix}-policy",
            forward_only=False,
            on_status=on_status,
        )
        ref_future = (
            pool.submit(
                wait_trainer_job,
                rlor_mgr,
                ref_handle,
                infra=cfg.infra,
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
                cfg.base_model,
                prev_identity=dep_info.prev_identity,
                timeout_s=dep_info.timeout_s,
            )
        else:
            dep_future = pool.submit(wait_deployment, deploy_mgr, dep_info, cfg.deployment)

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
        raise RuntimeError(
            "Infrastructure provisioning failed:\n" + "\n".join(errors)
        )
    return policy_ep, ref_ep, final_dep_info


def _make_policy_client(
    cfg: Any,
    rlor_mgr: TrainerJobManager,
    policy_ep: Any,
    api_key: str,
    closeables: list[Any],
) -> Any:
    """Build the policy ReconnectableClient and register its close."""
    policy = ReconnectableClient(
        rlor_mgr, policy_ep.job_id, cfg.base_model,
        lora_rank=cfg.lora_rank,
        fw_api_key=api_key,
        default_timeout=cfg.step_timeout or DEFAULT_TIMEOUT_S,
    )
    _register_closeable(policy, closeables)
    return policy


def _make_reference_client(
    *,
    cfg: Any,
    rlor_mgr: TrainerJobManager,
    policy: Any,
    reference_ep: Any | None,
    needs_reference: bool,
    api_key: str,
    closeables: list[Any],
) -> Any | None:
    """Build the reference client (separate trainer, shared session, or None)."""
    if reference_ep is not None:
        reference = ReconnectableClient(
            rlor_mgr, reference_ep.job_id, cfg.base_model,
            lora_rank=0,                               # ref trainers never carry LoRA
            fw_api_key=api_key,
            default_timeout=cfg.step_timeout or DEFAULT_TIMEOUT_S,
        )
        _register_closeable(reference, closeables)
        return reference
    if needs_reference and cfg.lora_rank > 0:
        # LoRA shared-session: policy serves reference via base-only handle on
        # the same FiretitanServiceClient. Avoids unloading the policy LoRA.
        reference = policy.create_base_reference()
        _register_closeable(reference, closeables)
        return reference
    return None


def _make_inference_components(
    *,
    cfg: Any,
    deploy_mgr: DeploymentManager,
    policy: Any,
    inference_model: str | None,
    api_key: str,
) -> tuple[Any, Any, Any | None]:
    """Build the sampler, weight syncer, and concurrency controller."""
    if not cfg.deployment.tokenizer_model:
        raise ValueError(
            "deployment.tokenizer_model is required for client-side tokenization."
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.deployment.tokenizer_model, trust_remote_code=True,
    )
    concurrency_controller = _build_concurrency_controller(
        cfg.concurrency, deploy_mgr, cfg.deployment,
    )
    sampler = DeploymentSampler(
        inference_url=deploy_mgr.inference_url,
        model=inference_model,
        api_key=api_key,
        tokenizer=tokenizer,
        concurrency_controller=concurrency_controller,
    )
    weight_syncer = WeightSyncer(
        policy_client=policy.inner,
        deploy_mgr=deploy_mgr,
        deployment_id=cfg.deployment.deployment_id,
        base_model=getattr(cfg, "rollout_base_model", None) or cfg.base_model,
        hotload_timeout=cfg.weight_sync.weight_sync_timeout,
        first_checkpoint_type=cfg.weight_sync.first_checkpoint_type,
        lora_rank=cfg.lora_rank,
    )
    return sampler, weight_syncer, concurrency_controller


def _make_boot_metrics(
    boot_start: float, deploy_mgr: DeploymentManager | None,
) -> dict:
    """Build the one-shot boot metrics dict."""
    metrics: dict = {
        "train/step": 0,
        "infra/total_boot_time": time.time() - boot_start,
    }
    if deploy_mgr is not None and deploy_mgr.boot_time_s is not None:
        metrics["infra/deploy_boot_time"] = deploy_mgr.boot_time_s
    return metrics


def _build_concurrency_controller(
    concurrency_config: Any,
    deploy_mgr: DeploymentManager,
    deployment_config: Any,
) -> Any | None:
    """Build a concurrency controller from a ``ConcurrencyConfig``."""
    mode = concurrency_config.mode
    if mode == "adaptive":
        gpu_count = get_deployment_gpu_count(deploy_mgr, deployment_config)
        initial_window = (
            concurrency_config.initial_window or (_SLOTS_PER_GPU * gpu_count)
        )
        controller = AdaptiveConcurrencyController(
            initial_window=initial_window,
            min_window=concurrency_config.min_window,
            max_window=concurrency_config.max_window,
            prefill_queue_target=concurrency_config.prefill_queue_target,
        )
        logger.info(
            "Using adaptive concurrency (initial=%d, range=%d-%d, target_pq=%.2fs)",
            initial_window,
            concurrency_config.min_window,
            concurrency_config.max_window,
            concurrency_config.prefill_queue_target,
        )
        return controller
    if mode == "fixed":
        logger.info("Using fixed concurrency: unlimited")
        return None
    if mode is None and concurrency_config.max_concurrency is not None:
        warnings.warn(
            "ConcurrencyConfig.max_concurrency is deprecated. "
            "Use mode='adaptive' (default) or mode='fixed' with "
            "FixedConcurrencyController instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        controller = FixedConcurrencyController(concurrency_config.max_concurrency)
        logger.info(
            "Using fixed concurrency (deprecated max_concurrency=%d)",
            concurrency_config.max_concurrency,
        )
        return controller
    raise ValueError(
        f"Unknown concurrency mode: {mode!r}. Must be 'adaptive' or 'fixed'."
    )
