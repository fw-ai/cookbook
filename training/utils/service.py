"""Map cookbook recipe config to an SDK-managed FireTitan service client."""

from __future__ import annotations

from typing import Any

from fireworks.training.sdk import (
    DeploymentCleanupOnClose,
    FiretitanServiceClient,
)

from training.utils.config import DeployConfig, TrainerConfig


def _firetitan_service_kwargs(
    *,
    base_model: str,
    tokenizer_model: str | None,
    lora_rank: int | None,
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
        "lora_rank": lora_rank or 0,
        "training_shape_id": trainer.training_shape_id,
        "reference_training_shape_id": trainer.reference_training_shape_id,
        "trainer_job_id": trainer.job_id,
        "reference_trainer_job_id": trainer.reference_job_id,
        "cleanup_reference_trainer_on_close": trainer.cleanup_reference_on_close,
        "reference_required": reference_required,
        "region": trainer.region,
        "max_context_length": max_context_length,
        "learning_rate": learning_rate,
        "node_count": trainer.node_count,
        "accelerator_type": trainer.accelerator_type,
        "accelerator_count": trainer.accelerator_count,
        "custom_image_tag": trainer.custom_image_tag,
        "extra_args": trainer.extra_args,
        "trainer_replica_count": trainer.replica_count,
        "trainer_timeout_s": trainer.timeout_s,
        "purpose": trainer.purpose,
        "managed_by": trainer.managed_by,
        "skip_validations": trainer.skip_validations,
        "cleanup_trainer_on_close": cleanup_trainer_on_close,
        "create_deployment": deployment is not None,
        "hotload_timeout_s": hotload_timeout_s,
        "cleanup_deployment_on_close": cleanup_deployment_on_close,
    }
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
    lora_rank: int | None,
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
        lora_rank=lora_rank,
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
