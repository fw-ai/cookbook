"""Map cookbook recipe config to an SDK-managed FireTitan service client."""

from __future__ import annotations

from typing import Any

from fireworks.training.sdk import (
    DeploymentCleanupOnClose,
    FireworksClient,
    FiretitanServiceClient,
)

from training.utils.config import DeployConfig, TrainerConfig


def resolve_router_replay_enabled(
    *,
    requested: bool,
    api_key: str,
    base_url: str,
    additional_headers: dict[str, str] | None,
    base_model: str,
) -> bool:
    """Enable Router Replay only when the base model can produce routing data."""
    if not requested:
        return False
    with FireworksClient(
        api_key=api_key,
        base_url=base_url,
        additional_headers=additional_headers,
    ) as client:
        return client.model_is_moe(base_model)


def _firetitan_service_kwargs(
    *,
    base_model: str,
    tokenizer_model: str | None,
    max_lora_rank: int | None,
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
