#!/usr/bin/env python3
"""Create a trainer, wait for readiness, then create its hotload deployment.

This is a low-level provisioning example for cases where strict startup order
matters. Unlike the managed recipe path, it does not overlap trainer readiness
with deployment creation:

    create trainer -> wait for trainer healthz -> create deployment -> wait ready

Waiting gives the trainer first claim on available capacity, but it is not a
placement policy. Use ``--bypass-reservation`` to opt both resources out of
account reservation defaulting.

Usage:
    export FIREWORKS_API_KEY=...

    python training/examples/tools/create_trainer_then_deployment.py \
        --base-model accounts/fireworks/models/glm-5p2-fp8 \
        --training-shape accounts/fireworks/trainingShapes/glm-5p2-fp8-1000k-gb300

The resources remain running after this script exits. Delete them explicitly
when they are no longer needed.
"""

from __future__ import annotations

import argparse
import logging
import os
import uuid

from fireworks.training.sdk import (
    DeploymentConfig,
    DeploymentInfo,
    DeploymentManager,
    TrainerJobConfig,
    TrainerJobManager,
    TrainerServiceEndpoint,
)

logger = logging.getLogger(__name__)

DEFAULT_TRAINER_READY_TIMEOUT_S = 60 * 60
DEFAULT_TRAINER_PENDING_TIMEOUT_S = 48 * 60 * 60
DEFAULT_DEPLOYMENT_READY_TIMEOUT_S = 90 * 60


def create_trainer_then_deployment(
    *,
    trainer_manager: TrainerJobManager,
    deployment_manager: DeploymentManager,
    base_model: str,
    training_shape: str,
    deployment_id: str,
    trainer_job_id: str | None = None,
    lora_rank: int = 0,
    learning_rate: float = 1e-5,
    bypass_reservation: bool = False,
    trainer_ready_timeout_s: float = DEFAULT_TRAINER_READY_TIMEOUT_S,
    trainer_pending_timeout_s: float = DEFAULT_TRAINER_PENDING_TIMEOUT_S,
    deployment_ready_timeout_s: float = DEFAULT_DEPLOYMENT_READY_TIMEOUT_S,
) -> tuple[TrainerServiceEndpoint, DeploymentInfo]:
    """Provision linked resources without overlapping their startup."""
    profile = trainer_manager.resolve_training_profile(training_shape)
    logger.info("Resolved training shape: %s", profile.training_shape_version)
    logger.info("Resolved deployment shape: %s", profile.deployment_shape)

    trainer_config = TrainerJobConfig(
        base_model=base_model,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        training_shape_ref=profile.training_shape_version,
        requested_job_id=trainer_job_id,
        use_reservation=not bypass_reservation,
    )

    logger.info("[1/4] Creating trainer")
    created_trainer = trainer_manager.create(trainer_config)
    logger.info("[2/4] Waiting for trainer %s to become ready", created_trainer.job_id)
    trainer_endpoint = trainer_manager.wait_for_ready(
        created_trainer.job_id,
        job_name=created_trainer.job_name,
        timeout_s=trainer_ready_timeout_s,
        pending_timeout_s=trainer_pending_timeout_s,
    )

    # No deployment API call occurs before wait_for_ready() returns above.
    deployment_extra_values = None
    if bypass_reservation:
        deployment_extra_values = {"bypass_reservation": "true"}

    deployment_config = DeploymentConfig.from_training_profile(
        deployment_id=deployment_id,
        base_model=base_model,
        profile=profile,
        min_replica_count=1,
        max_replica_count=1,
        hot_load_trainer_job=trainer_endpoint.job_name,
        for_training=True,
        extra_values=deployment_extra_values,
    )

    logger.info("[3/4] Trainer is ready; creating deployment %s", deployment_id)
    deployment_manager.create_or_get(deployment_config)
    logger.info("[4/4] Waiting for deployment %s to become ready", deployment_id)
    deployment = deployment_manager.wait_for_ready(
        deployment_id,
        timeout_s=deployment_ready_timeout_s,
    )
    return trainer_endpoint, deployment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a ready trainer before creating its linked hotload deployment.",
    )
    parser.add_argument("--base-model", required=True, help="Full Fireworks base-model resource name.")
    parser.add_argument(
        "--training-shape",
        required=True,
        help="Full training-shape resource name, optionally including a version.",
    )
    parser.add_argument(
        "--trainer-job-id",
        default=None,
        help="Optional stable trainer job ID. The SDK generates one when omitted.",
    )
    parser.add_argument(
        "--deployment-id",
        default=None,
        help="Optional deployment ID. A unique ID is generated when omitted.",
    )
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--bypass-reservation",
        action="store_true",
        help="Set trainer use_reservation=false and deployment bypass_reservation=true.",
    )
    parser.add_argument(
        "--trainer-ready-timeout-s",
        type=float,
        default=DEFAULT_TRAINER_READY_TIMEOUT_S,
    )
    parser.add_argument(
        "--trainer-pending-timeout-s",
        type=float,
        default=DEFAULT_TRAINER_PENDING_TIMEOUT_S,
    )
    parser.add_argument(
        "--deployment-ready-timeout-s",
        type=float,
        default=DEFAULT_DEPLOYMENT_READY_TIMEOUT_S,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    deployment_id = args.deployment_id or f"training-hotload-{uuid.uuid4().hex[:8]}"

    trainer_manager = TrainerJobManager(api_key=api_key, base_url=base_url)
    deployment_manager = DeploymentManager(api_key=api_key, base_url=base_url)

    trainer, deployment = create_trainer_then_deployment(
        trainer_manager=trainer_manager,
        deployment_manager=deployment_manager,
        base_model=args.base_model,
        training_shape=args.training_shape,
        trainer_job_id=args.trainer_job_id,
        deployment_id=deployment_id,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        bypass_reservation=args.bypass_reservation,
        trainer_ready_timeout_s=args.trainer_ready_timeout_s,
        trainer_pending_timeout_s=args.trainer_pending_timeout_s,
        deployment_ready_timeout_s=args.deployment_ready_timeout_s,
    )

    logger.info("Trainer ready: %s", trainer.job_name)
    logger.info("Trainer endpoint: %s", trainer.base_url)
    logger.info("Deployment ready: %s", deployment.name)
    logger.info("Inference model: %s", deployment.inference_model)


if __name__ == "__main__":
    main()
