#!/usr/bin/env python3
"""Stage 2 of the distillation pipeline: distill the fine-tuned Qwen3-32B
teacher (produced by run_qwen3_32b_distill_teacher.sh) into Qwen3-8B via
on-policy reverse-KL distillation.

Usage:
    export FIREWORKS_API_KEY=...
    export TEACHER_JOB_ID=<policy_job_id from stage 1>
    python train_distill_qwen3.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from training.recipes.distillation_loop import Config, main as distill_main
from training.utils import DeployConfig, InfraConfig, WandBConfig, WeightSyncConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


DEEPMATH_DATASET = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "rl", "deepmath", "dataset.jsonl")
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--teacher-job-id",
        default=os.environ.get("TEACHER_JOB_ID"),
        help="Trainer job ID of the fine-tuned teacher. Defaults to $TEACHER_JOB_ID.",
    )
    p.add_argument(
        "--student-base-model",
        default="accounts/fireworks/models/qwen3-8b",
    )
    p.add_argument(
        "--teacher-base-model",
        default="accounts/fireworks/models/qwen3-32b",
        help="Teacher resource name -- must match the base_model the stage-1 RL job was created with.",
    )
    p.add_argument("--student-tokenizer-model", default="Qwen/Qwen3-8B")
    p.add_argument("--teacher-tokenizer-model", default="Qwen/Qwen3-32B")
    p.add_argument("--dataset-path", default=DEEPMATH_DATASET)
    p.add_argument("--max-rows", type=int, default=100)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--kl-penalty-coef", type=float, default=1.0)
    p.add_argument("--kl-discount-factor", type=float, default=0.0)
    p.add_argument("--completions-per-prompt", type=int, default=4)
    p.add_argument("--prompt-groups-per-step", type=int, default=1)
    p.add_argument("--max-completion-tokens", type=int, default=1024)
    p.add_argument("--policy-loss", default="importance_sampling")
    p.add_argument("--training-shape", default=os.environ.get("TRAINING_SHAPE", ""))
    p.add_argument("--ref-training-shape", default=os.environ.get("REF_TRAINING_SHAPE", ""))
    p.add_argument("--region", default=os.environ.get("REGION", "US_VIRGINIA_1"))
    p.add_argument("--log-path", default=f"./distill_logs/{int(time.time())}")
    p.add_argument("--output-model-id", default=None)
    p.add_argument("--skip-cleanup", action="store_true")
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "distillation-tinker"))
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    if not args.teacher_job_id:
        raise SystemExit(
            "Missing teacher job ID. Pass --teacher-job-id <id> or set "
            "TEACHER_JOB_ID. This must be the policy_job_id from the "
            "stage-1 RL run (run_qwen3_32b_distill_teacher.sh)."
        )

    infra_kwargs = {}
    if args.training_shape:
        infra_kwargs["training_shape_id"] = args.training_shape
    if args.ref_training_shape:
        infra_kwargs["ref_training_shape_id"] = args.ref_training_shape
    if args.region:
        infra_kwargs["region"] = args.region

    wandb_kwargs = {"project": args.wandb_project}
    if args.wandb_entity:
        wandb_kwargs["entity"] = args.wandb_entity

    return Config(
        log_path=args.log_path,
        base_model=args.student_base_model,
        teacher_base_model=args.teacher_base_model,
        teacher_tokenizer_model=args.teacher_tokenizer_model,
        teacher_job_id=args.teacher_job_id,
        dataset=args.dataset_path,
        learning_rate=args.learning_rate,
        kl_penalty_coef=args.kl_penalty_coef,
        kl_discount_factor=args.kl_discount_factor,
        pure_distillation=True,
        completions_per_prompt=args.completions_per_prompt,
        prompt_groups_per_step=args.prompt_groups_per_step,
        max_completion_tokens=args.max_completion_tokens,
        max_rows=args.max_rows,
        epochs=args.epochs,
        policy_loss=args.policy_loss,
        output_model_id=args.output_model_id,
        infra=InfraConfig(**infra_kwargs),
        deployment=DeployConfig(
            tokenizer_model=args.student_tokenizer_model,
            sample_timeout=600,
        ),
        weight_sync=WeightSyncConfig(
            weight_sync_interval=1,
            first_checkpoint_type="base",
            weight_sync_before_training=True,
            weight_sync_timeout=600,
        ),
        wandb=WandBConfig(**wandb_kwargs),
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    logger.info(
        "Distilling %s (teacher_job_id=%s) into %s on %d DeepMath rows",
        args.teacher_base_model, args.teacher_job_id, args.student_base_model, args.max_rows,
    )
    metrics = distill_main(config, cancel_on_exit=not args.skip_cleanup)
    logger.info("Distillation complete. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
