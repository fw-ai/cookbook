#!/usr/bin/env python3
# ruff: noqa: E402
"""Promote a saved DCP checkpoint by loading it into a fresh trainer.

This example creates a temporary trainer from a training shape, loads an
existing DCP checkpoint referenced from cookbook `checkpoints.jsonl`, saves a
promotable sampler checkpoint, and promotes that checkpoint into a Fireworks
model.

When checkpoints were saved with ``base_model`` and ``training_shape`` fields
(cookbook >= v2), ``--model`` and ``--shape`` are auto-populated from the
checkpoint and can be omitted.

Usage:
    export FIREWORKS_API_KEY=...

    # Auto-detect model and shape from checkpoint metadata:
    python train_promote_checkpoint.py \
        --checkpoints-jsonl ./sft_logs/checkpoints.jsonl

    # Explicit (overrides checkpoint metadata):
    python train_promote_checkpoint.py \
        --checkpoints-jsonl ./sft_logs/checkpoints.jsonl \
        --model accounts/fireworks/models/qwen3-8b \
        --shape accounts/fireworks/trainingShapes/ts-qwen3-8b-policy

    # Specific step:
    python train_promote_checkpoint.py \
        --checkpoints-jsonl ./sft_logs/checkpoints.jsonl \
        --step 5
"""

from __future__ import annotations

import argparse
import json
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

from fireworks.training.sdk import TrainerJobManager
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
    checkpoints_jsonl: str
    step: int | None
    base_model: str | None
    training_shape: str | None
    lora_rank: int
    output_model_id: str | None
    trainer_timeout_s: float
    dcp_timeout_s: int
    keep_trainer: bool


@dataclass(frozen=True)
class ResolvedCheckpoint:
    checkpoint_ref: str
    checkpoint_name: str
    source_job_id: str | None
    base_model: str | None = None
    training_shape: str | None = None


def parse_args() -> PromoteConfig:
    parser = argparse.ArgumentParser(
        description="Load a saved DCP checkpoint into a temporary trainer and promote it",
    )
    parser.add_argument(
        "--checkpoints-jsonl",
        required=True,
        help=(
            "Path to cookbook checkpoints.jsonl. The script loads the latest "
            "`state_path`, or the one for --step when provided."
        ),
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Training step to promote. Defaults to the latest checkpoint in checkpoints.jsonl.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Base model to launch in the temporary trainer. "
            "Auto-detected from checkpoint metadata when omitted."
        ),
    )
    parser.add_argument(
        "--shape",
        default=None,
        help=(
            "Training shape to use for the temporary trainer. "
            "Auto-detected from checkpoint metadata when omitted. "
            "Accepts either accounts/<account>/trainingShapes/<shape> or a bare shape ID."
        ),
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=0,
        help="LoRA rank for the temporary trainer. Leave at 0 for full-parameter checkpoints.",
    )
    parser.add_argument(
        "--output-model-id",
        default=None,
        help="Promoted model ID. Defaults to an auto-generated value derived from the model and checkpoint.",
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
        checkpoints_jsonl=args.checkpoints_jsonl,
        step=args.step,
        base_model=args.model,
        training_shape=args.shape,
        lora_rank=args.lora_rank,
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


def _checkpoint_label(checkpoint_ref: str) -> str:
    return checkpoint_ref.rstrip("/").rsplit("/", 1)[-1]


def _validate_checkpoint_ref(checkpoint_ref: str) -> None:
    if checkpoint_ref.startswith(("cross_job://", "gs://", "/")):
        return
    raise ValueError(
        "checkpoint must be a saved `state_path` from checkpoints.jsonl "
        "(cross_job://...), a `gs://...` path, or an absolute local path"
    )


def _load_checkpoint_entries(checkpoints_jsonl: str) -> list[dict]:
    entries: list[dict] = []
    with open(checkpoints_jsonl) as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Could not parse {checkpoints_jsonl}:{line_no} as JSON"
                ) from exc
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Expected a JSON object at {checkpoints_jsonl}:{line_no}"
                )
            entries.append(entry)
    if not entries:
        raise ValueError(f"No checkpoint entries found in {checkpoints_jsonl}")
    return entries


def _resolve_checkpoint_from_jsonl(
    checkpoints_jsonl: str,
    step: int | None,
) -> ResolvedCheckpoint:
    entries = _load_checkpoint_entries(checkpoints_jsonl)
    candidates = [entry for entry in entries if entry.get("state_path")]
    if not candidates:
        raise ValueError(f"No entries with state_path found in {checkpoints_jsonl}")

    if step is not None:
        matches = [entry for entry in candidates if entry.get("step") == step]
        if not matches:
            raise ValueError(f"Step {step} was not found in {checkpoints_jsonl}")
        chosen = matches[-1]
    else:
        chosen = candidates[-1]

    checkpoint_ref = str(chosen["state_path"])
    _validate_checkpoint_ref(checkpoint_ref)
    return ResolvedCheckpoint(
        checkpoint_ref=checkpoint_ref,
        checkpoint_name=str(chosen.get("name") or _checkpoint_label(checkpoint_ref)),
        source_job_id=chosen.get("source_job_id"),
        base_model=chosen.get("base_model"),
        training_shape=chosen.get("training_shape"),
    )


def _default_output_model_id(base_model: str, checkpoint_name: str) -> str:
    model_short = base_model.rsplit("/", 1)[-1]
    suffix = f"{checkpoint_name}-{int(time.time()) % 100000}"
    return _sanitize_resource_id(
        f"{model_short}-promote-{suffix}", default="promoted-model"
    )[:63].rstrip("-")


def _default_display_name(base_model: str, checkpoint_name: str) -> str:
    model_short = base_model.rsplit("/", 1)[-1]
    return _sanitize_resource_id(
        f"promote-{model_short}-{checkpoint_name}", default="promote-checkpoint"
    )[:63].rstrip("-")


def main() -> None:
    cfg = parse_args()
    resolved_checkpoint = _resolve_checkpoint_from_jsonl(
        cfg.checkpoints_jsonl,
        cfg.step,
    )

    # Resolve base_model: CLI flag > checkpoint metadata
    base_model = cfg.base_model or resolved_checkpoint.base_model
    if not base_model:
        raise SystemExit(
            "ERROR: --model is required (checkpoint has no base_model metadata).\n"
            "Older checkpoints don't store base_model. Pass it explicitly:\n"
            "  --model accounts/fireworks/models/<model-name>"
        )

    # Resolve training_shape: CLI flag > checkpoint metadata
    training_shape_raw = cfg.training_shape or resolved_checkpoint.training_shape
    if not training_shape_raw:
        raise SystemExit(
            "ERROR: --shape is required (checkpoint has no training_shape metadata).\n"
            "Older checkpoints don't store training_shape. Pass it explicitly:\n"
            "  --shape accounts/fireworks/trainingShapes/<shape-name>"
        )

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")

    rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)

    training_shape = _normalize_training_shape(
        training_shape_raw,
        rlor_mgr.account_id,
    )
    profile = rlor_mgr.resolve_training_profile(training_shape)

    output_model_id = cfg.output_model_id or _default_output_model_id(
        base_model,
        resolved_checkpoint.checkpoint_name,
    )
    display_name = _default_display_name(
        base_model,
        resolved_checkpoint.checkpoint_name,
    )
    sampler_name = _sanitize_resource_id(
        f"promote-{resolved_checkpoint.checkpoint_name}-{int(time.time()) % 100000}",
        default="promote-checkpoint",
    )

    logger.info("Checkpoints jsonl:   %s", cfg.checkpoints_jsonl)
    if cfg.step is not None:
        logger.info("Requested step:      %d", cfg.step)
    logger.info("Checkpoint ref:      %s", resolved_checkpoint.checkpoint_ref)
    logger.info("Checkpoint name:     %s", resolved_checkpoint.checkpoint_name)
    if resolved_checkpoint.source_job_id:
        logger.info("Source job ID:       %s", resolved_checkpoint.source_job_id)
    logger.info("Base model:          %s%s", base_model, " (from checkpoint)" if not cfg.base_model else "")
    logger.info("Training shape:      %s%s", training_shape, " (from checkpoint)" if not cfg.training_shape else "")
    logger.info("Resolved shape ver.: %s", profile.training_shape_version)
    logger.info("LoRA rank:           %d", cfg.lora_rank)
    logger.info("Output model ID:     %s", output_model_id)

    cleanup = ResourceCleanup(rlor_mgr)
    with cleanup:
        endpoint = create_trainer_job(
            rlor_mgr,
            base_model=base_model,
            infra=InfraConfig(trainer_timeout_s=cfg.trainer_timeout_s),
            profile=profile,
            lora_rank=cfg.lora_rank,
            display_name=display_name,
            cleanup=None if cfg.keep_trainer else cleanup,
        )

        client = ReconnectableClient(
            rlor_mgr=rlor_mgr,
            job_id=endpoint.job_id,
            base_model=base_model,
            lora_rank=cfg.lora_rank,
            fw_api_key=api_key,
            endpoint=endpoint,
        )

        checkpoint_ref = client.resolve_checkpoint_path(
            resolved_checkpoint.checkpoint_ref
        )
        logger.info("Loading DCP checkpoint: %s", checkpoint_ref)
        t0 = time.time()
        client.load_state_with_optimizer(checkpoint_ref, timeout=cfg.dcp_timeout_s)
        logger.info("Checkpoint loaded in %.1fs", time.time() - t0)

        save_result = client.save_weights_for_sampler_ext(
            sampler_name,
            checkpoint_type="base",
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
