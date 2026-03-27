#!/usr/bin/env python3
# ruff: noqa: E402
"""Promote a sampler checkpoint to a deployable Fireworks model.

Reads ``checkpoints.jsonl`` (produced by cookbook recipes), finds the
sampler checkpoint ID and source trainer job, and calls the promotion
API.  No temporary trainer is needed — promotion is a lightweight
metadata + file-copy operation.

Usage:
    export FIREWORKS_API_KEY=...

    # Promote the latest checkpoint (auto-detects model from metadata):
    python promote_checkpoint.py \
        --checkpoints-jsonl ./sft_logs/checkpoints.jsonl

    # Promote a specific step:
    python promote_checkpoint.py \
        --checkpoints-jsonl ./sft_logs/checkpoints.jsonl \
        --step 10

    # Override base model and output model ID:
    python promote_checkpoint.py \
        --checkpoints-jsonl ./sft_logs/checkpoints.jsonl \
        --model accounts/fireworks/models/qwen3-8b \
        --output-model-id my-fine-tuned-qwen3-8b
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()


@dataclass(frozen=True)
class PromoteConfig:
    checkpoints_jsonl: str
    step: int | None
    base_model: str | None
    output_model_id: str | None


@dataclass(frozen=True)
class ResolvedCheckpoint:
    checkpoint_name: str
    sampler_path: str
    source_job_id: str
    base_model: str | None = None


def parse_args() -> PromoteConfig:
    parser = argparse.ArgumentParser(
        description="Promote a sampler checkpoint to a deployable Fireworks model",
    )
    parser.add_argument(
        "--checkpoints-jsonl",
        required=True,
        help="Path to cookbook checkpoints.jsonl.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Training step to promote. Defaults to the latest checkpoint.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Base model for metadata inheritance. "
            "Auto-detected from checkpoint metadata when omitted."
        ),
    )
    parser.add_argument(
        "--output-model-id",
        default=None,
        help="Promoted model ID. Defaults to an auto-generated value.",
    )
    args = parser.parse_args()
    return PromoteConfig(
        checkpoints_jsonl=args.checkpoints_jsonl,
        step=args.step,
        base_model=args.model,
        output_model_id=args.output_model_id,
    )


def _sanitize_resource_id(value: str, *, default: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or default


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


def _resolve_checkpoint(
    checkpoints_jsonl: str,
    step: int | None,
) -> ResolvedCheckpoint:
    entries = _load_checkpoint_entries(checkpoints_jsonl)
    candidates = [e for e in entries if e.get("sampler_path") and e.get("source_job_id")]
    if not candidates:
        raise SystemExit(
            "ERROR: No checkpoint entries with sampler_path and source_job_id found.\n"
            "Ensure checkpoints were saved with CheckpointKind.BOTH (the default in cookbook recipes)."
        )

    if step is not None:
        matches = [e for e in candidates if e.get("step") == step]
        if not matches:
            raise SystemExit(f"ERROR: Step {step} not found in {checkpoints_jsonl}")
        chosen = matches[-1]
    else:
        chosen = candidates[-1]

    return ResolvedCheckpoint(
        checkpoint_name=str(chosen.get("name", chosen["sampler_path"])),
        sampler_path=chosen["sampler_path"],
        source_job_id=chosen["source_job_id"],
        base_model=chosen.get("base_model"),
    )


def _default_output_model_id(base_model: str, checkpoint_name: str) -> str:
    model_short = base_model.rsplit("/", 1)[-1]
    suffix = f"{checkpoint_name}-{int(time.time()) % 100000}"
    return _sanitize_resource_id(
        f"{model_short}-promote-{suffix}", default="promoted-model"
    )[:63].rstrip("-")


def main() -> None:
    cfg = parse_args()
    resolved = _resolve_checkpoint(cfg.checkpoints_jsonl, cfg.step)

    base_model = cfg.base_model or resolved.base_model
    if not base_model:
        raise SystemExit(
            "ERROR: --model is required (checkpoint has no base_model metadata).\n"
            "  --model accounts/fireworks/models/<model-name>"
        )

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)

    output_model_id = cfg.output_model_id or _default_output_model_id(
        base_model, resolved.checkpoint_name,
    )

    logger.info("Checkpoint:      %s", resolved.sampler_path)
    logger.info("Source job:      %s", resolved.source_job_id)
    logger.info("Base model:      %s", base_model)
    logger.info("Output model ID: %s", output_model_id)

    model = rlor_mgr.promote_checkpoint(
        resolved.source_job_id,
        resolved.sampler_path,
        output_model_id,
        base_model=base_model,
    )

    logger.info(
        "Promoted model: %s",
        model.get("name", f"accounts/{rlor_mgr.account_id}/models/{output_model_id}"),
    )
    logger.info(
        "Model state=%s kind=%s",
        model.get("state", "UNKNOWN"),
        model.get("kind", "UNKNOWN"),
    )


if __name__ == "__main__":
    main()
