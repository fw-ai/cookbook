"""The frozen teacher used to score student rollouts during distillation."""

from __future__ import annotations

import re
from typing import Any

from fireworks.training.sdk import CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO
from fireworks.training.sdk.deployment import DeploymentConfig, DeploymentSampler

DEPLOYMENT_ID_MAX_LENGTH = 63
DEFAULT_TEACHER_DEPLOYMENT_TIMEOUT_S = 5400


def is_base_model_resource(model: str) -> bool:
    """True for ``accounts/<a>/models/<m>``; False for deployments and deployed models."""
    return "/models/" in model and "/deployments/" not in model and "/deployedModels/" not in model


def default_teacher_deployment_id(base_model: str) -> str:
    """A stable, valid deployment id derived from the teacher base model."""
    slug = base_model.rstrip("/").split("/")[-1]
    safe_slug = re.sub(r"[^a-z0-9-]+", "-", slug.lower()).strip("-")
    return f"distillation-teacher-{safe_slug}"[:DEPLOYMENT_ID_MAX_LENGTH]


def create_teacher_sampler(
    service: Any,
    teacher_model: str,
    *,
    tokenizer: Any,
    inference_url: str,
    deployment_id: str | None = None,
    deployment_shape: str | None = None,
    replica_count: int = 1,
    timeout_s: float = DEFAULT_TEACHER_DEPLOYMENT_TIMEOUT_S,
    cleanup_on_close: bool = False,
) -> DeploymentSampler:
    """Return a sampler that scores tokens with the teacher.

    An already-served model (deployment or deployed model) is used directly.
    A base model gets its own frozen inference deployment (no hot-load), named
    ``deployment_id`` or :func:`default_teacher_deployment_id`; the deployment
    API picks a shape when ``deployment_shape`` is ``None``.
    """
    if not is_base_model_resource(teacher_model):
        return service.create_deployment_sampler_for_model(
            teacher_model,
            tokenizer=tokenizer,
            inference_url=inference_url,
        )
    return service.create_inference_deployment_sampler(
        DeploymentConfig(
            deployment_id=deployment_id or default_teacher_deployment_id(teacher_model),
            base_model=teacher_model,
            deployment_shape=deployment_shape,
            min_replica_count=replica_count,
            max_replica_count=replica_count,
            hot_load_bucket_type=None,
            enable_hot_load=False,
            for_training=True,
        ),
        timeout_s=timeout_s,
        cleanup_on_close=CLEANUP_DEPLOYMENT_ON_CLOSE_SCALE_TO_ZERO if cleanup_on_close else None,
        tokenizer=tokenizer,
    )
