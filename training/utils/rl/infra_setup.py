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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import transformers

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from fireworks.training.sdk.deployment import (
    AdaptiveConcurrencyController,
    DeploymentSampler,
    FixedConcurrencyController,
)
from fireworks.training.sdk.weight_syncer import WeightSyncer

from training.utils import (
    ReconnectableClient,
    ResourceCleanup,
    auto_select_training_shape,
    create_trainer_job,
    get_deployment_gpu_count,
    setup_or_reattach_deployment,
)
from training.utils.client import DEFAULT_TIMEOUT_S

logger = logging.getLogger(__name__)

__all__ = ["Infra", "setup_infra"]


# Default concurrent requests per GPU when AdaptiveConcurrencyController's
# initial_window is unspecified.
_SLOTS_PER_GPU = 8


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
    cleanup: Optional[ResourceCleanup] = None,
    cleanup_on_exit: bool = False,
    on_status: Optional[Callable[[str], None]] = None,
) -> Infra:
    """Build all training-side infrastructure in one call.

    Inputs (duck-typed on ``cfg``):

    Always required:
        ``base_model``, ``lora_rank``, ``max_seq_len`` (or auto from shape),
        ``learning_rate``, ``step_timeout``, ``infra`` (InfraConfig),
        ``policy_job_id`` / ``reference_job_id`` (set to reuse existing
        trainers), ``policy_base_url`` / ``reference_base_url``.

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
    _on_status = on_status or (lambda _msg: None)
    if needs_inference and deploy_mgr is None:
        raise ValueError("deploy_mgr is required when needs_inference=True")

    boot_start = time.time()

    # -- Shape resolution (auto-select then fetch profile, in one place) ----

    if cfg.infra.training_shape_id is None:
        cfg.infra.training_shape_id = auto_select_training_shape(
            rlor_mgr,
            base_model=cfg.base_model,
            trainer_role="policy",
            lora_rank=cfg.lora_rank,
            max_seq_len=cfg.max_seq_len,
        )
        logger.info("Auto-selected policy training shape: %s", cfg.infra.training_shape_id)
    policy_profile = rlor_mgr.resolve_training_profile(cfg.infra.training_shape_id)

    if (
        needs_inference
        and not cfg.deployment.deployment_shape
        and policy_profile.deployment_shape_version
    ):
        cfg.deployment.deployment_shape = policy_profile.deployment_shape_version
    if cfg.max_seq_len is None:
        cfg.max_seq_len = policy_profile.max_supported_context_length
    if cfg.max_seq_len is None:
        raise ValueError(
            "max_seq_len is required. Set it in Config, or use a training shape "
            "(InfraConfig.training_shape_id) to auto-populate it."
        )

    # ref_profile is non-None only when a *separate* reference trainer is
    # needed: needs_reference + (full-param OR explicit ref shape).
    # LoRA + needs_reference + no explicit ref shape → shared-session path,
    # the policy trainer serves reference via base-only handle.
    ref_profile = None
    if cfg.infra.ref_training_shape_id:
        ref_profile = rlor_mgr.resolve_training_profile(cfg.infra.ref_training_shape_id)
    elif needs_reference and cfg.lora_rank == 0:
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
        ref_profile = rlor_mgr.resolve_training_profile(cfg.infra.ref_training_shape_id)

    logger.info(
        "Policy shape=%s  ref shape=%s  deployment_shape=%s",
        cfg.infra.training_shape_id,
        cfg.infra.ref_training_shape_id,
        getattr(cfg.deployment, "deployment_shape", None) if needs_inference else None,
    )

    # -- Trainer provisioning (parallel when both needed) -------------------

    _cleanup = cleanup if cleanup_on_exit else None
    role_prefix = "grpo" if needs_inference else "dpo"

    def _make_trainer(role: str, profile: Any, *, forward_only: bool, lora: int) -> Any:
        is_ref = role == "reference"
        return create_trainer_job(
            rlor_mgr,
            base_model=cfg.base_model,
            infra=cfg.infra,
            profile=profile,
            lora_rank=lora,
            max_seq_len=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            display_name=f"{role_prefix}-{role}",
            forward_only=forward_only,
            job_id=(
                getattr(cfg, "reference_job_id", None) if is_ref
                else getattr(cfg, "policy_job_id", None)
            ),
            base_url_override=(
                getattr(cfg, "reference_base_url", None) if is_ref
                else getattr(cfg, "policy_base_url", None)
            ),
            cleanup=_cleanup,
            on_status=_on_status,
        )

    if ref_profile is not None:
        _on_status("provisioning policy and reference trainers")
        with ThreadPoolExecutor(max_workers=2) as pool:
            pol_fut = pool.submit(
                _make_trainer, "policy", policy_profile,
                forward_only=False, lora=cfg.lora_rank,
            )
            ref_fut = pool.submit(
                _make_trainer, "reference", ref_profile,
                forward_only=True, lora=0,  # reference never carries LoRA adapters
            )
            errors: list[str] = []
            policy_ep = reference_ep = None
            try:
                policy_ep = pol_fut.result()
            except Exception as e:
                errors.append(f"Policy trainer: {e}")
            try:
                reference_ep = ref_fut.result()
            except Exception as e:
                errors.append(f"Reference trainer: {e}")
            if errors:
                raise RuntimeError(
                    "Trainer creation failed:\n" + "\n".join(errors)
                )
    else:
        _on_status("provisioning policy trainer")
        policy_ep = _make_trainer(
            "policy", policy_profile,
            forward_only=False, lora=cfg.lora_rank,
        )
        reference_ep = None

    policy_job_id = policy_ep.job_id
    reference_job_id = reference_ep.job_id if reference_ep is not None else None

    # -- Deployment (only when needed) --------------------------------------

    dep_info = None
    inference_model = None
    if needs_inference:
        dep_info = setup_or_reattach_deployment(
            deploy_mgr, cfg.deployment, cfg.base_model, cfg.infra,
            policy_ep.job_name,
        )
        if cleanup_on_exit and cleanup is not None:
            cleanup.deployment(cfg.deployment.deployment_id, action="scale_to_zero")
        inference_model = dep_info.inference_model if dep_info else cfg.base_model

    # -- Trainer clients ----------------------------------------------------

    closeables: list[Any] = []
    timeout = cfg.step_timeout or DEFAULT_TIMEOUT_S
    use_endpoint_override = bool(
        getattr(cfg, "policy_base_url", None) or getattr(cfg, "reference_base_url", None)
    )

    policy = ReconnectableClient(
        rlor_mgr, policy_ep.job_id, cfg.base_model,
        lora_rank=cfg.lora_rank,
        fw_api_key=api_key,
        default_timeout=timeout,
        endpoint=policy_ep if use_endpoint_override else None,
    )
    if hasattr(policy, "close"):
        closeables.append(policy)

    # Reference: separate trainer when ref_profile resolved, else
    # shared-session base handle when LoRA + needs_reference, else None.
    reference: Any | None = None
    if reference_ep is not None:
        reference = ReconnectableClient(
            rlor_mgr, reference_ep.job_id, cfg.base_model,
            lora_rank=0,                                # ref trainers never carry LoRA
            fw_api_key=api_key,
            default_timeout=timeout,
            endpoint=reference_ep if use_endpoint_override else None,
        )
        if hasattr(reference, "close"):
            closeables.append(reference)
    elif needs_reference and cfg.lora_rank > 0:
        # LoRA shared-session: policy serves reference via base-only handle on
        # the same FiretitanServiceClient. Avoids unloading the policy LoRA.
        reference = policy.create_base_reference()
        if hasattr(reference, "close"):
            closeables.append(reference)

    # -- Sampler + weight syncer + concurrency controller (inference only) --

    sampler = None
    weight_syncer = None
    concurrency_controller = None
    if needs_inference:
        tokenizer_model = cfg.deployment.tokenizer_model
        if not tokenizer_model:
            raise ValueError(
                "deployment.tokenizer_model is required for client-side tokenization."
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_model, trust_remote_code=True,
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

    # -- Boot metrics -------------------------------------------------------

    boot_metrics: dict = {
        "train/step": 0,
        "infra/total_boot_time": time.time() - boot_start,
    }
    if needs_inference and deploy_mgr.boot_time_s is not None:
        boot_metrics["infra/deploy_boot_time"] = deploy_mgr.boot_time_s

    return Infra(
        policy=policy,
        reference=reference,
        sampler=sampler,
        weight_syncer=weight_syncer,
        concurrency_controller=concurrency_controller,
        policy_profile=policy_profile,
        policy_job_id=policy_job_id,
        reference_job_id=reference_job_id,
        inference_model=inference_model,
        boot_metrics=boot_metrics,
        closeables=closeables,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_concurrency_controller(
    concurrency_config: Any,
    deploy_mgr: DeploymentManager,
    deployment_config: Any,
) -> Any | None:
    """Build a concurrency controller from a ``ConcurrencyConfig``."""
    cc = concurrency_config
    if cc.mode == "adaptive":
        gpu_count = get_deployment_gpu_count(deploy_mgr, deployment_config)
        initial_window = cc.initial_window or (_SLOTS_PER_GPU * gpu_count)
        controller = AdaptiveConcurrencyController(
            initial_window=initial_window,
            min_window=cc.min_window,
            max_window=cc.max_window,
            prefill_queue_target=cc.prefill_queue_target,
        )
        logger.info(
            "Using adaptive concurrency (initial=%d, range=%d-%d, target_pq=%.2fs)",
            initial_window, cc.min_window, cc.max_window, cc.prefill_queue_target,
        )
        return controller
    if cc.mode == "fixed":
        logger.info("Using fixed concurrency: unlimited")
        return None
    if cc.mode is None and cc.max_concurrency is not None:
        import warnings
        warnings.warn(
            "ConcurrencyConfig.max_concurrency is deprecated. "
            "Use mode='adaptive' (default) or mode='fixed' with "
            "FixedConcurrencyController instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        controller = FixedConcurrencyController(cc.max_concurrency)
        logger.info(
            "Using fixed concurrency (deprecated max_concurrency=%d)",
            cc.max_concurrency,
        )
        return controller
    raise ValueError(
        f"Unknown concurrency mode: {cc.mode!r}. Must be 'adaptive' or 'fixed'."
    )
