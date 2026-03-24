#!/usr/bin/env python3
# ruff: noqa: E402
"""Promote the latest DCP checkpoint from an existing trainer job.

This example creates a temporary trainer from a training shape, restores the
latest DCP checkpoint from a previous job, saves a promotable sampler
checkpoint, and promotes that checkpoint into a Fireworks model.

Usage:
    export FIREWORKS_API_KEY=...

    python train_promote_checkpoint.py \
        --job-id <source-trainer-job> \
        --model accounts/fireworks/models/qwen3-8b \
        --shape accounts/fireworks/trainingShapes/ts-qwen3-8b-policy
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass

from dotenv import load_dotenv

_COOKBOOK_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _COOKBOOK_ROOT not in sys.path:
    sys.path.insert(0, _COOKBOOK_ROOT)

from fireworks.training.sdk import TrainerCheckpoint, TrainerJobManager
from training.utils import (
    InfraConfig,
    ReconnectableClient,
    ResourceCleanup,
    create_trainer_job,
)
from training.utils.checkpoint_utils import get_sampler_checkpoint_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_TRAINER_TIMEOUT_S = 3600.0
DEFAULT_DCP_TIMEOUT_S = 2700
_FULL_TRAINING_SHAPE_RE = re.compile(r"^accounts/[^/]+/trainingShapes/[^/]+$")
_VERSIONED_TRAINING_SHAPE_RE = re.compile(
    r"^(accounts/[^/]+/trainingShapes/[^/]+)/versions/[^/]+$"
)


@dataclass(frozen=True)
class PromoteConfig:
    source_job_id: str
    base_model: str
    training_shape: str
    checkpoint_name: str | None
    output_model_id: str | None
    trainer_timeout_s: float
    dcp_timeout_s: int
    keep_trainer: bool


def parse_args() -> PromoteConfig:
    parser = argparse.ArgumentParser(
        description="Restore a DCP checkpoint into a temporary trainer and promote it",
    )
    parser.add_argument(
        "--job-id",
        required=True,
        help="Source trainer job that owns the DCP checkpoint",
    )
    parser.add_argument(
        "--model", required=True, help="Base model to launch in the temporary trainer"
    )
    parser.add_argument(
        "--shape",
        required=True,
        help=(
            "Training shape to use for the temporary trainer. "
            "Accepts either accounts/<account>/trainingShapes/<shape> or a bare shape ID."
        ),
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help="Specific DCP checkpoint name to restore. Defaults to the latest training checkpoint for the job.",
    )
    parser.add_argument(
        "--output-model-id",
        default=None,
        help="Promoted model ID. Defaults to an auto-generated value derived from the model and source job.",
    )
    parser.add_argument(
        "--trainer-timeout-s",
        type=float,
        default=DEFAULT_TRAINER_TIMEOUT_S,
        help="Timeout for the temporary trainer to become ready.",
    )
    parser.add_argument(
        "--dcp-timeout-s",
        type=int,
        default=DEFAULT_DCP_TIMEOUT_S,
        help="Timeout for loading the DCP checkpoint.",
    )
    parser.add_argument(
        "--keep-trainer",
        action="store_true",
        help="Keep the temporary trainer alive after promotion instead of deleting it on exit.",
    )
    args = parser.parse_args()
    return PromoteConfig(
        source_job_id=args.job_id,
        base_model=args.model,
        training_shape=args.shape,
        checkpoint_name=args.checkpoint_name,
        output_model_id=args.output_model_id,
        trainer_timeout_s=args.trainer_timeout_s,
        dcp_timeout_s=args.dcp_timeout_s,
        keep_trainer=args.keep_trainer,
    )


def _normalize_training_shape(training_shape: str, default_account: str) -> str:
    """Accept a bare shape ID or full resource and return an unversioned resource name."""
    versioned_match = _VERSIONED_TRAINING_SHAPE_RE.match(training_shape)
    if versioned_match:
        normalized = versioned_match.group(1)
        logger.info(
            "Stripping explicit training-shape version and resolving latest validated: %s",
            normalized,
        )
        return normalized
    if _FULL_TRAINING_SHAPE_RE.match(training_shape):
        return training_shape
    return f"accounts/{default_account}/trainingShapes/{training_shape}"


def _sanitize_resource_id(value: str, *, default: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or default


def _default_output_model_id(base_model: str, source_job_id: str) -> str:
    model_short = base_model.rsplit("/", 1)[-1]
    suffix = f"{source_job_id[-8:]}-{int(time.time()) % 100000}"
    return _sanitize_resource_id(
        f"{model_short}-promote-{suffix}", default="promoted-model"
    )[:63].rstrip("-")


def _default_display_name(base_model: str, source_job_id: str) -> str:
    model_short = base_model.rsplit("/", 1)[-1]
    suffix = source_job_id[-8:]
    return _sanitize_resource_id(
        f"promote-{model_short}-{suffix}", default="promote-checkpoint"
    )[:63].rstrip("-")


def _source_training_config(job: dict) -> dict:
    return job.get("trainingConfig", {}) or {}


def _source_lora_rank(job: dict) -> int:
    value = _source_training_config(job).get("loraRank", 0)
    return int(value or 0)


def _checkpoint_sort_key(checkpoint: TrainerCheckpoint) -> tuple[int, object]:
    if checkpoint.create_time:
        return (2, checkpoint.create_time)
    match = re.search(r"(\d+)$", checkpoint.checkpoint_id)
    if match:
        return (1, int(match.group(1)))
    return (0, checkpoint.checkpoint_id)


def _select_dcp_checkpoint(
    checkpoints: list[TrainerCheckpoint],
    requested_name: str | None,
) -> TrainerCheckpoint:
    if requested_name:
        for checkpoint in checkpoints:
            if (
                checkpoint.checkpoint_id == requested_name
                or checkpoint.name == requested_name
            ):
                if checkpoint.checkpoint_type not in {"training", "training_lora"}:
                    raise ValueError(
                        f"Checkpoint '{requested_name}' is a {checkpoint.checkpoint_type} checkpoint, not a DCP checkpoint"
                    )
                return checkpoint
        raise ValueError(
            f"Checkpoint '{requested_name}' was not found in the source job"
        )

    candidates = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.checkpoint_type in {"training", "training_lora"}
    ]
    if not candidates:
        raise ValueError("No training DCP checkpoints were found for the source job")
    return max(candidates, key=_checkpoint_sort_key)


def main() -> None:
    cfg = parse_args()

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)

    source_job = rlor_mgr.get(cfg.source_job_id)
    source_training = _source_training_config(source_job)
    source_base_model = source_training.get("baseModel")
    lora_rank = _source_lora_rank(source_job)

    if source_base_model and source_base_model != cfg.base_model:
        logger.warning(
            "Source job base model is %s, but the temporary trainer will use %s",
            source_base_model,
            cfg.base_model,
        )

    training_shape = _normalize_training_shape(cfg.training_shape, rlor_mgr.account_id)
    profile = rlor_mgr.resolve_training_profile(training_shape)
    checkpoint = _select_dcp_checkpoint(
        rlor_mgr.list_job_checkpoints(cfg.source_job_id),
        cfg.checkpoint_name,
    )

    output_model_id = cfg.output_model_id or _default_output_model_id(
        cfg.base_model, cfg.source_job_id
    )
    display_name = _default_display_name(cfg.base_model, cfg.source_job_id)
    sampler_name = _sanitize_resource_id(
        f"promote-{cfg.source_job_id[-8:]}-{int(time.time()) % 100000}",
        default="promote-checkpoint",
    )

    logger.info("Source job:          %s", cfg.source_job_id)
    logger.info("Base model:          %s", cfg.base_model)
    logger.info("Training shape:      %s", training_shape)
    logger.info("Resolved shape ver.: %s", profile.training_shape_version)
    logger.info("DCP checkpoint:      %s", checkpoint.checkpoint_id)
    logger.info("Source LoRA rank:    %d", lora_rank)
    logger.info("Output model ID:     %s", output_model_id)

    cleanup = ResourceCleanup(rlor_mgr)
    with cleanup:
        endpoint = create_trainer_job(
            rlor_mgr,
            base_model=cfg.base_model,
            infra=InfraConfig(trainer_timeout_s=cfg.trainer_timeout_s),
            profile=profile,
            lora_rank=lora_rank,
            display_name=display_name,
            cleanup=None if cfg.keep_trainer else cleanup,
        )

        client = ReconnectableClient(
            rlor_mgr=rlor_mgr,
            job_id=endpoint.job_id,
            base_model=cfg.base_model,
            lora_rank=lora_rank,
            fw_api_key=api_key,
            endpoint=endpoint,
        )

        checkpoint_ref = client.resolve_checkpoint_path(
            checkpoint.checkpoint_id,
            source_job_id=cfg.source_job_id,
        )
        logger.info("Loading DCP checkpoint: %s", checkpoint_ref)
        t0 = time.time()
        client.load_state_with_optimizer(checkpoint_ref, timeout=cfg.dcp_timeout_s)
        logger.info("Checkpoint loaded in %.1fs", time.time() - t0)

        save_result = client.save_weights_for_sampler_ext(
            sampler_name, checkpoint_type="base"
        )
        sampler_checkpoint_id = get_sampler_checkpoint_id(save_result)
        logger.info("Saved sampler checkpoint: %s", sampler_checkpoint_id)

        model = rlor_mgr.promote_checkpoint(
            endpoint.job_id,
            sampler_checkpoint_id,
            output_model_id,
        )
        logger.info(
            "Promoted model: %s",
            model.get(
                "name", f"accounts/{rlor_mgr.account_id}/models/{output_model_id}"
            ),
        )
        logger.info(
            "Model state=%s kind=%s",
            model.get("state", "UNKNOWN"),
            model.get("kind", "UNKNOWN"),
        )

        if cfg.keep_trainer:
            logger.info("Temporary trainer left running: %s", endpoint.job_id)


if __name__ == "__main__":
    main()
