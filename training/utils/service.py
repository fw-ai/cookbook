"""Map cookbook recipe config to an SDK-managed FireTitan service client."""

from __future__ import annotations

import logging
from dataclasses import fields
from typing import Any

from fireworks.training.sdk import (
    DeploymentCleanupOnClose,
    FireworksClient,
    FiretitanServiceClient,
)
from fireworks.training.sdk.managed import FiretitanProvisioningConfig

try:
    from fireworks.training.sdk import ModelDetailsUnavailableError
except ImportError:  # SDK < this change: no typed error, so no last-resort path

    class ModelDetailsUnavailableError(RuntimeError):  # type: ignore[no-redef]
        """Placeholder never raised by an older SDK, so its RuntimeError
        propagates unchanged (today's behavior)."""

        status_code = 0


from training.utils.config import DeployConfig, TrainerConfig, WeightSyncScope

logger = logging.getLogger(__name__)


def make_weight_sync(
    policy: Any, service: Any, deployment: DeployConfig, *, extended: bool = False
):
    """Select publication once, preserving FILE names and initial-base requests."""
    if getattr(policy, "supports_rdma_weight_sync", False) is True:
        def publish(name: str, **kwargs: Any) -> Any:
            result = policy.weight_sync()
            logger.info("Weight sync completed: %s", "RDMA" if result.optimizer_version is not None else "FILE")
            return result

        return publish
    save = (
        policy.save_weights_for_sampler_ext
        if extended
        else policy.save_weights_for_sampler
    )

    def publish(name, **kwargs):
        saved = save(name, **kwargs)
        result = service.hotload_sampler_snapshot(
            saved.snapshot_name if extended else saved.path
        )
        logger.info("Weight sync completed: FILE")
        return result

    return publish


def resolve_router_replay_enabled(
    *,
    requested: bool,
    api_key: str,
    base_url: str,
    additional_headers: dict[str, str] | None,
    base_model: str,
    training_client: Any | None = None,
) -> bool:
    """Enable Router Replay only when the base model can produce routing data.

    A clear capability answer always wins; the user's ``requested`` flag is
    only honored when the model is MoE, or as a last resort when no source can
    answer:

    1. ``requested`` is False -> off.
    2. The trainer advertised ``supports_router_replay`` (a real bool) on
       ``training_client``: dense -> off even if requested (logged); MoE ->
       honor ``requested``. No model GET happens.
    3. Otherwise (older trainer / no client) probe the model's MoE flag with
       the same rule: dense -> off, MoE -> honor ``requested``.
    4. Last resort: the probe is 403/404 (the model record is not visible to
       this caller, e.g. a private early-access base model) -> honor
       ``requested`` with a warning. Any other probe failure -- control-plane
       errors or a record with no MoE flag -- still raises, so R3 is never
       silently changed on a real error.
    """
    if not requested:
        return False
    advertised = getattr(training_client, "supports_router_replay", None)
    if isinstance(advertised, bool):
        if not advertised:
            logger.info(
                "Router Replay disabled: trainer reports base model %s is dense",
                base_model,
            )
        return advertised
    try:
        with FireworksClient(
            api_key=api_key,
            base_url=base_url,
            additional_headers=additional_headers,
        ) as client:
            is_moe = client.model_is_moe(base_model)
    except ModelDetailsUnavailableError as exc:
        if exc.status_code not in (403, 404):
            raise
        logger.warning(
            "Router Replay: cannot determine whether %s is MoE (model record not "
            "visible to this caller, HTTP %d) and the trainer did not report it; "
            "honoring router_replay=True as requested. If the model is dense, "
            "sampling will reject the routing request -- rerun with "
            "--router-replay false.",
            base_model,
            exc.status_code,
        )
        return True
    if not is_moe:
        logger.info("Router Replay disabled: base model %s is dense", base_model)
    return is_moe


def _firetitan_service_kwargs(
    *,
    base_model: str,
    tokenizer_model: str | None,
    max_lora_rank: int | None,
    projection_head_dim: int | None = None,
    max_context_length: int | None,
    learning_rate: float,
    trainer: TrainerConfig,
    deployment: DeployConfig | None = None,
    hotload_timeout_s: float | None = None,
    cleanup_trainer_on_close: bool = False,
    cleanup_deployment_on_close: DeploymentCleanupOnClose | None = None,
    reference_required: bool = False,
) -> dict[str, Any]:
    """Translate cookbook user config into SDK service kwargs."""
    service_kwargs: dict[str, Any] = {
        "base_model": base_model,
        "tokenizer_model": tokenizer_model,
        "training_shape_id": trainer.training_shape_id,
        "reference_training_shape_id": trainer.reference_training_shape_id,
        "trainer_job_id": trainer.job_id,
        "reference_trainer_job_id": trainer.reference_job_id,
        "cleanup_reference_trainer_on_close": trainer.cleanup_reference_on_close,
        "reference_required": reference_required,
        "region": trainer.region,
        "max_context_length": max_context_length,
        "learning_rate": learning_rate,
        # Server-side gradient accumulation is deprecated on the Tinker/RLOR
        # path (the managed config defaults this to 1, which logs a deprecation
        # warning). Recipes express gradient accumulation as client-side control
        # flow -- N forward_backward calls per optim_step -- so leave the
        # server-side knob unset.
        "gradient_accumulation_steps": None,
        "node_count": trainer.node_count,
        "custom_image_tag": trainer.custom_image_tag,
        "extra_args": trainer.extra_args,
        "trainer_replica_count": trainer.replica_count,
        "trainer_timeout_s": trainer.timeout_s,
        "trainer_pending_timeout_s": trainer.pending_timeout_s,
        "inactivity_timeout": trainer.inactivity_timeout,
        "disable_inactivity_cleanup": trainer.disable_inactivity_cleanup,
        "purpose": trainer.purpose,
        "preemptible": trainer.preemptible,
        "managed_by": trainer.managed_by,
        "skip_validations": trainer.skip_validations,
        "use_reservation": trainer.use_reservation,
        "cleanup_trainer_on_close": cleanup_trainer_on_close,
        "create_deployment": deployment is not None,
        "hotload_timeout_s": hotload_timeout_s,
        "cleanup_deployment_on_close": cleanup_deployment_on_close,
    }
    # Projection topology is service-scoped. Omit the field entirely for actor
    # services; the SDK also canonicalizes an explicit zero to ``None``.
    if isinstance(projection_head_dim, bool) or (
        projection_head_dim is not None and not isinstance(projection_head_dim, int)
    ):
        raise ValueError("projection_head_dim must be a non-negative integer when set")
    if projection_head_dim is not None and projection_head_dim < 0:
        raise ValueError("projection_head_dim must be a non-negative integer when set")
    if projection_head_dim:
        service_kwargs["projection_head_dim"] = projection_head_dim
    # Keep the default path compatible with the cookbook's declared minimum SDK,
    # which predates reservation_target. An explicit target requires the newer SDK
    # surface and is therefore forwarded only when the caller requests it.
    if trainer.reservation_target is not None:
        service_kwargs["reservation_target"] = trainer.reservation_target
    if max_lora_rank is not None and max_lora_rank < 0:
        raise ValueError("max_lora_rank must be non-negative")
    if max_lora_rank and max_lora_rank > 0:
        service_kwargs["max_lora_rank"] = max_lora_rank
    else:
        service_kwargs["lora_rank"] = 0
    if deployment is None:
        service_kwargs["replica_count"] = 1
        return service_kwargs

    service_kwargs.update(
        {
            "deployment_shape": deployment.deployment_shape,
            "deployment_id": deployment.deployment_id,
            "deployment_extra_args": deployment.deployment_extra_args,
            "deployment_extra_values": deployment.extra_values,
            "deployment_timeout_s": deployment.deployment_timeout_s,
            "replica_count": deployment.replica_count,
            "disable_speculative_decoding": deployment.disable_speculative_decoding,
            "hot_load_transition_type": deployment.hot_load_transition_type,
        }
    )
    if (
        deployment.wait_for_trainer_before_deployment
        and deployment.weight_sync_scope != WeightSyncScope.PER_TRAINER
    ):
        raise ValueError(
            "wait_for_trainer_before_deployment requires PER_TRAINER weight sync"
        )
    if deployment.weight_sync_transport is not None:
        if deployment.weight_sync_transport != "RDMA":
            raise ValueError("weight_sync_transport must be 'RDMA' or None")
        if deployment.weight_sync_scope != WeightSyncScope.PER_TRAINER:
            raise ValueError("RDMA weight sync requires PER_TRAINER weight sync")
    supported = {f.name for f in fields(FiretitanProvisioningConfig)}
    optional_deploy = {
        "weight_sync_transport": deployment.weight_sync_transport,
        "wait_for_trainer_before_deployment": deployment.wait_for_trainer_before_deployment,
    }
    for name, value in optional_deploy.items():
        if not value:
            continue
        if name in supported:
            service_kwargs[name] = value
            continue
        raise RuntimeError(
            f"{name}={value!r} was requested, but the installed fireworks-ai "
            f"SDK's FiretitanProvisioningConfig has no {name!r} field. "
            "Upgrade the SDK or drop the option."
        )
    return service_kwargs


def build_service_client(
    *,
    api_key: str,
    base_url: str,
    inference_url: str | None = None,
    additional_headers: dict[str, str] | None,
    base_model: str,
    tokenizer_model: str | None,
    max_lora_rank: int | None,
    projection_head_dim: int | None = None,
    max_context_length: int | None,
    learning_rate: float,
    trainer: TrainerConfig,
    deployment: DeployConfig | None = None,
    hotload_timeout_s: float | None = None,
    cleanup_trainer_on_close: bool = False,
    cleanup_deployment_on_close: DeploymentCleanupOnClose | None = None,
    reference_required: bool = False,
) -> FiretitanServiceClient:
    """Create an SDK-managed service client from cookbook config."""
    service_kwargs = _firetitan_service_kwargs(
        base_model=base_model,
        tokenizer_model=tokenizer_model,
        max_lora_rank=max_lora_rank,
        projection_head_dim=projection_head_dim,
        max_context_length=max_context_length,
        learning_rate=learning_rate,
        trainer=trainer,
        deployment=deployment,
        hotload_timeout_s=hotload_timeout_s,
        cleanup_trainer_on_close=cleanup_trainer_on_close,
        cleanup_deployment_on_close=cleanup_deployment_on_close,
        reference_required=reference_required,
    )
    return FiretitanServiceClient.from_firetitan_config(
        api_key=api_key,
        base_url=base_url,
        inference_url=inference_url,
        additional_headers=additional_headers,
        **service_kwargs,
    )
