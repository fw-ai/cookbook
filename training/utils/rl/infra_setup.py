"""Heavy-lifting infrastructure setup for RL recipes.

The principle: recipes provide a ``Config`` and we hand back wired-up
infrastructure. Recipes never write deployment-creation, trainer-creation,
client-reconnect, sampler, or weight-syncer logic. If a knob exists,
it lives in ``Config``; if not, the right default is baked in here.

Returns an :class:`Infra` bundle of fully-constructed clients. The
caller is responsible for lifecycle (registering ``infra.closeables``
into an ``ExitStack`` and the deployment cleanup into a
``ResourceCleanup``). This keeps :func:`setup_infra` itself a pure
factory — no context-manager semantics.

Required attributes on ``cfg`` (duck-typed, structurally enforced):

* ``cfg.base_model: str``
* ``cfg.rollout_base_model: str | None``
* ``cfg.lora_rank: int``
* ``cfg.learning_rate: float``
* ``cfg.kl_beta: float``
* ``cfg.max_seq_len: int | None``
* ``cfg.step_timeout: int``
* ``cfg.policy_job_id: str | None``
* ``cfg.reference_job_id: str | None``
* ``cfg.policy_base_url: str | None``
* ``cfg.reference_base_url: str | None``
* ``cfg.infra: InfraConfig``
* ``cfg.deployment: DeployConfig``
* ``cfg.weight_sync: WeightSyncConfig``
* ``cfg.concurrency: ConcurrencyConfig``
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional
from concurrent.futures import ThreadPoolExecutor

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


# Default concurrent requests per GPU when an adaptive controller's
# initial_window is unspecified. Same default as rl_loop.py used inline.
_SLOTS_PER_GPU = 8


@dataclass
class Infra:
    """Wired-up infrastructure handed back to the recipe.

    All clients are ready to use. Callers must register :attr:`closeables`
    into their ``ExitStack`` so connections are released on shutdown.

    Attributes:
        policy: Policy trainer client.
        reference: Reference trainer client (or ``None`` when KL is disabled
            or LoRA-mode shares the policy session).
        sampler: Inference sampler bound to the deployment.
        weight_syncer: Save-and-hotload helper.
        tokenizer: HuggingFace tokenizer for client-side tokenization.
        policy_profile: Resolved policy training-shape profile.
        ref_profile: Resolved reference training-shape profile (or ``None``).
        inference_model: The base model name for inference (may differ from
            ``cfg.base_model`` when an adapter is mounted).
        dep_info: Whatever ``setup_or_reattach_deployment`` returned (may be ``None``).
        policy_job_id: Policy trainer job ID.
        reference_job_id: Reference trainer job ID — equals ``policy_job_id``
            when no separate reference trainer was provisioned.
        boot_metrics: One-shot boot metrics suitable for ``wandb_log``.
        closeables: Objects with ``.close()`` methods the caller should
            register into an ``ExitStack``.
    """

    policy: Any
    reference: Optional[Any]
    sampler: DeploymentSampler
    weight_syncer: WeightSyncer
    tokenizer: Any
    policy_profile: Any
    ref_profile: Optional[Any]
    inference_model: str
    dep_info: Optional[Any]
    policy_job_id: str
    reference_job_id: str
    boot_metrics: dict
    concurrency_controller: Optional[Any] = None
    closeables: List[Any] = field(default_factory=list)


def setup_infra(
    cfg: Any,
    *,
    rlor_mgr: TrainerJobManager,
    deploy_mgr: DeploymentManager,
    api_key: str,
    cleanup: Optional[ResourceCleanup] = None,
    cleanup_on_exit: bool = False,
    on_status: Optional[Callable[[str], None]] = None,
    auto_select_reference_when_kl: bool = True,
) -> Infra:
    """Provision deployment, trainers, sampler, and weight syncer.

    Args:
        cfg: Recipe config (duck-typed; see module docstring for required
            attributes).
        rlor_mgr: Trainer job manager.
        deploy_mgr: Deployment manager.
        api_key: Fireworks API key (forwarded to ``ReconnectableClient``).
        cleanup: Optional ``ResourceCleanup`` for deployment teardown.
        cleanup_on_exit: When True, schedule ``scale_to_zero`` on the
            deployment via ``cleanup``.
        on_status: Optional progress callback for "provisioning X" updates.
        auto_select_reference_when_kl: When True, automatically pick a
            reference shape when ``kl_beta > 0`` and ``lora_rank == 0`` and
            no explicit ``ref_training_shape_id`` is set. Disable for
            recipes that handle reference selection themselves.

    Returns:
        An :class:`Infra` bundle.

    Side effects:
        Mutates ``cfg.infra.training_shape_id``, ``cfg.infra.ref_training_shape_id``,
        ``cfg.deployment.deployment_shape``, and ``cfg.max_seq_len`` when
        these are auto-derived from the resolved training shape.
    """
    _on_status = on_status or (lambda _msg: None)

    # -- Resolve policy training shape --------------------------------------

    if not cfg.infra.training_shape_id:
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

    policy_profile = rlor_mgr.resolve_training_profile(cfg.infra.training_shape_id)

    if not cfg.deployment.deployment_shape and policy_profile.deployment_shape_version:
        cfg.deployment.deployment_shape = policy_profile.deployment_shape_version
    logger.info(
        "Policy shape=%s  deployment_shape=%s",
        cfg.infra.training_shape_id,
        cfg.deployment.deployment_shape,
    )

    if cfg.max_seq_len is None:
        cfg.max_seq_len = policy_profile.max_supported_context_length
        logger.info("max_seq_len from training shape: %d", cfg.max_seq_len)
    if cfg.max_seq_len is None:
        raise ValueError(
            "max_seq_len is required. Set it in Config, or use a training shape "
            "(InfraConfig.training_shape_id) to auto-populate it."
        )

    # -- Resolve reference training shape -----------------------------------
    # ref_profile is non-None only when a separate reference trainer is needed.
    # LoRA + kl_beta > 0 without an explicit ref shape: the policy trainer
    # serves reference logprobs via a base-only model handle (no extra GPU job).
    ref_profile = None
    if cfg.infra.ref_training_shape_id:
        ref_profile = rlor_mgr.resolve_training_profile(cfg.infra.ref_training_shape_id)
    elif (
        auto_select_reference_when_kl
        and cfg.kl_beta > 0
        and cfg.lora_rank == 0
    ):
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

    # -- Provision trainers (parallel when both needed) ---------------------

    boot_start = time.time()
    _cleanup = cleanup if cleanup_on_exit else None

    def _make_trainer(role: str, profile: Any, *, forward_only: bool = False) -> Any:
        is_ref = role == "reference"
        return create_trainer_job(
            rlor_mgr,
            base_model=cfg.base_model,
            infra=cfg.infra,
            profile=profile,
            lora_rank=cfg.lora_rank,
            max_seq_len=cfg.max_seq_len,
            learning_rate=cfg.learning_rate,
            display_name=f"grpo-{role}",
            forward_only=forward_only,
            job_id=cfg.reference_job_id if is_ref else cfg.policy_job_id,
            base_url_override=cfg.reference_base_url if is_ref else cfg.policy_base_url,
            cleanup=_cleanup,
            on_status=_on_status,
        )

    if ref_profile is not None:
        _on_status("provisioning policy and reference trainers")
        with ThreadPoolExecutor(max_workers=2) as pool:
            pol_fut = pool.submit(_make_trainer, "policy", policy_profile)
            ref_fut = pool.submit(
                _make_trainer, "reference", ref_profile, forward_only=True,
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
        policy_ep = _make_trainer("policy", policy_profile)
        reference_ep = None

    policy_job_id = policy_ep.job_id
    reference_job_id = reference_ep.job_id if reference_ep else policy_ep.job_id

    # -- Connect deployment to the trainer's hot-load bucket ----------------

    dep_info = setup_or_reattach_deployment(
        deploy_mgr, cfg.deployment, cfg.base_model, cfg.infra, policy_ep.job_name,
    )
    if cleanup_on_exit and cleanup is not None:
        cleanup.deployment(cfg.deployment.deployment_id, action="scale_to_zero")

    # -- Trainer clients ----------------------------------------------------

    closeables: list[Any] = []
    timeout = cfg.step_timeout or DEFAULT_TIMEOUT_S

    def _make_client(ep: Any) -> Any:
        c = ReconnectableClient(
            rlor_mgr,
            ep.job_id,
            cfg.base_model,
            lora_rank=cfg.lora_rank,
            fw_api_key=api_key,
            default_timeout=timeout,
            endpoint=ep if (cfg.policy_base_url or cfg.reference_base_url) else None,
        )
        if hasattr(c, "close"):
            closeables.append(c)
        return c

    policy = _make_client(policy_ep)

    if reference_ep is not None:
        reference = _make_client(reference_ep)
    elif cfg.lora_rank > 0 and cfg.kl_beta > 0:
        # Share the policy trainer's session: base-only model handle, no
        # second trainer, no second create_session (which would unload the
        # policy LoRA). See create_base_reference() in utils/client.py —
        # both clients share one FiretitanServiceClient (one session) so
        # they don't reset each other.
        reference = policy.create_base_reference()
        closeables.append(reference)
    else:
        reference = None

    # -- Tokenizer ----------------------------------------------------------

    inference_model = dep_info.inference_model if dep_info else cfg.base_model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.deployment.tokenizer_model, trust_remote_code=True,
    )

    # -- Concurrency controller --------------------------------------------

    concurrency_controller = _build_concurrency_controller(cfg, deploy_mgr)

    sampler = DeploymentSampler(
        inference_url=deploy_mgr.inference_url,
        model=inference_model,
        api_key=api_key,
        tokenizer=tokenizer,
        concurrency_controller=concurrency_controller,
    )

    # -- Weight syncer ------------------------------------------------------

    weight_syncer = WeightSyncer(
        policy_client=policy.inner,
        deploy_mgr=deploy_mgr,
        deployment_id=cfg.deployment.deployment_id,
        base_model=cfg.rollout_base_model or cfg.base_model,
        hotload_timeout=cfg.weight_sync.weight_sync_timeout,
        first_checkpoint_type=cfg.weight_sync.first_checkpoint_type,
        lora_rank=cfg.lora_rank,
    )

    # -- Boot metrics -------------------------------------------------------

    boot_metrics: dict = {
        "train/step": 0,
        "infra/total_boot_time": time.time() - boot_start,
    }
    if deploy_mgr.boot_time_s is not None:
        boot_metrics["infra/deploy_boot_time"] = deploy_mgr.boot_time_s

    return Infra(
        policy=policy,
        reference=reference,
        sampler=sampler,
        weight_syncer=weight_syncer,
        tokenizer=tokenizer,
        policy_profile=policy_profile,
        ref_profile=ref_profile,
        inference_model=inference_model,
        dep_info=dep_info,
        policy_job_id=policy_job_id,
        reference_job_id=reference_job_id,
        boot_metrics=boot_metrics,
        concurrency_controller=concurrency_controller,
        closeables=closeables,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_concurrency_controller(cfg: Any, deploy_mgr: DeploymentManager) -> Any:
    """Translate ``cfg.concurrency`` into a controller instance.

    Mirrors the inline branching that lived in rl_loop.py.
    """
    cc = cfg.concurrency
    if cc.mode == "adaptive":
        gpu_count = get_deployment_gpu_count(deploy_mgr, cfg.deployment)
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
