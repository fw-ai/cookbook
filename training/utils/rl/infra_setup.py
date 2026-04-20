"""Composable one-shot infrastructure builders for RL recipes.

Library-style: each function does one thing, takes explicit arguments,
returns a value, and never calls back into user code. Recipes compose
them directly — there is no bundled "do everything" entry point and no
lifecycle hidden behind a context manager. The recipe owns ordering,
the recipe owns cleanup.

Typical recipe usage::

    shape_id, profile = resolve_policy_profile(
        rlor_mgr,
        shape_id=cfg.infra.training_shape_id,
        base_model=cfg.base_model,
        lora_rank=cfg.lora_rank,
        max_seq_len=cfg.max_seq_len,
    )
    cfg.max_seq_len = cfg.max_seq_len or profile.max_supported_context_length

    ref_shape_id, ref_profile = resolve_reference_profile(
        rlor_mgr,
        shape_id=cfg.infra.ref_training_shape_id,
        base_model=cfg.base_model,
        lora_rank=cfg.lora_rank,
        max_seq_len=cfg.max_seq_len,
        kl_beta=cfg.kl_beta,
    )

    policy_ep, ref_ep = provision_trainer_pair(
        rlor_mgr,
        base_model=cfg.base_model,
        infra_config=cfg.infra,
        policy_profile=profile,
        ref_profile=ref_profile,
        lora_rank=cfg.lora_rank,
        max_seq_len=cfg.max_seq_len,
        learning_rate=cfg.learning_rate,
    )

    controller = make_concurrency_controller(cfg.concurrency, deploy_mgr, cfg.deployment)

The recipe then constructs ``ReconnectableClient``, ``DeploymentSampler``,
and ``WeightSyncer`` directly — those are already library-shaped.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from fireworks.training.sdk.deployment import (
    AdaptiveConcurrencyController,
    FixedConcurrencyController,
)

from training.utils import (
    auto_select_training_shape,
    create_trainer_job,
    get_deployment_gpu_count,
)

logger = logging.getLogger(__name__)

__all__ = [
    "resolve_policy_profile",
    "resolve_reference_profile",
    "provision_trainer_pair",
    "make_concurrency_controller",
]


# Default concurrent requests per GPU when AdaptiveConcurrencyController's
# initial_window is unspecified.
_SLOTS_PER_GPU = 8


def resolve_policy_profile(
    rlor_mgr: TrainerJobManager,
    *,
    shape_id: str | None,
    base_model: str,
    lora_rank: int,
    max_seq_len: int | None,
) -> tuple[str, Any]:
    """Resolve the policy training shape into a profile.

    If ``shape_id`` is ``None``, auto-select via
    :func:`auto_select_training_shape` and return the resolved id.

    Returns ``(resolved_shape_id, profile)``. The recipe is responsible
    for storing ``resolved_shape_id`` back into its config if it wants
    that side effect; this function does not mutate inputs.
    """
    if shape_id is None:
        shape_id = auto_select_training_shape(
            rlor_mgr,
            base_model=base_model,
            trainer_role="policy",
            lora_rank=lora_rank,
            max_seq_len=max_seq_len,
        )
        logger.info("Auto-selected policy training shape: %s", shape_id)
    profile = rlor_mgr.resolve_training_profile(shape_id)
    return shape_id, profile


def resolve_reference_profile(
    rlor_mgr: TrainerJobManager,
    *,
    shape_id: str | None,
    base_model: str,
    lora_rank: int,
    max_seq_len: int,
    kl_beta: float,
    auto_select_when_kl: bool = True,
) -> tuple[str | None, Any | None]:
    """Resolve the reference shape, auto-selecting when KL is enabled.

    Returns ``(shape_id, profile)``. Both are ``None`` when no
    reference trainer is needed (KL disabled, or LoRA mode where the
    policy trainer can serve reference logprobs via a base-only handle).

    No mutation of inputs.
    """
    if shape_id:
        return shape_id, rlor_mgr.resolve_training_profile(shape_id)
    if auto_select_when_kl and kl_beta > 0 and lora_rank == 0:
        shape_id = auto_select_training_shape(
            rlor_mgr,
            base_model=base_model,
            trainer_role="reference",
            lora_rank=lora_rank,
            max_seq_len=max_seq_len,
        )
        logger.info("Auto-selected reference training shape: %s", shape_id)
        return shape_id, rlor_mgr.resolve_training_profile(shape_id)
    return None, None


def provision_trainer_pair(
    rlor_mgr: TrainerJobManager,
    *,
    base_model: str,
    infra_config: Any,
    policy_profile: Any,
    ref_profile: Any | None,
    lora_rank: int,
    max_seq_len: int,
    learning_rate: float,
    policy_job_id: str | None = None,
    reference_job_id: str | None = None,
    policy_base_url: str | None = None,
    reference_base_url: str | None = None,
    cleanup: Any = None,
    on_status: Optional[Callable[[str], None]] = None,
) -> tuple[Any, Any | None]:
    """Provision policy trainer (and reference, if ``ref_profile`` set) in parallel.

    Returns ``(policy_endpoint, reference_endpoint)``. ``reference_endpoint``
    is ``None`` when ``ref_profile`` is ``None``.

    ``on_status`` is the only callback (purely advisory progress reporting);
    pass ``None`` to suppress.
    """
    _on_status = on_status or (lambda _msg: None)

    def _make_trainer(role: str, profile: Any, *, forward_only: bool = False) -> Any:
        is_ref = role == "reference"
        return create_trainer_job(
            rlor_mgr,
            base_model=base_model,
            infra=infra_config,
            profile=profile,
            lora_rank=lora_rank,
            max_seq_len=max_seq_len,
            learning_rate=learning_rate,
            display_name=f"grpo-{role}",
            forward_only=forward_only,
            job_id=reference_job_id if is_ref else policy_job_id,
            base_url_override=reference_base_url if is_ref else policy_base_url,
            cleanup=cleanup,
            on_status=_on_status,
        )

    if ref_profile is None:
        _on_status("provisioning policy trainer")
        return _make_trainer("policy", policy_profile), None

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
            raise RuntimeError("Trainer creation failed:\n" + "\n".join(errors))
        return policy_ep, reference_ep


def make_concurrency_controller(
    concurrency_config: Any,
    deploy_mgr: DeploymentManager,
    deployment_config: Any,
) -> Any | None:
    """Build a concurrency controller from a ``ConcurrencyConfig``.

    Returns ``None`` for unlimited (``mode="fixed"``), an
    :class:`AdaptiveConcurrencyController` for ``mode="adaptive"``, or
    a :class:`FixedConcurrencyController` for the deprecated
    ``max_concurrency`` path.

    Raises ``ValueError`` for unknown modes.
    """
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
