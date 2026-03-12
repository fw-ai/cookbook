#!/usr/bin/env python3
"""ORPO training on IFEval instruction-following preference pairs.

Run prepare_data.py first, then: python train_ifeval_orpo.py
"""

from __future__ import annotations

import argparse
import os
import logging

from dotenv import load_dotenv

import training.recipes.orpo_loop as orpo_loop
from fireworks.training.sdk import TrainerJobManager
from training.utils import InfraConfig, WandBConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

FIREWORKS_API_KEY = os.environ["FIREWORKS_API_KEY"]
FIREWORKS_ACCOUNT_ID = os.environ.get("FIREWORKS_ACCOUNT_ID", "")
FIREWORKS_BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ORPO on IFEval preference pairs"
    )
    parser.add_argument("--base-model", default="accounts/fireworks/models/qwen3-8b")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--dataset-path",
        default=os.path.join(os.path.dirname(__file__), "dataset.jsonl"),
    )
    parser.add_argument("--training-shape", default="")
    parser.add_argument("--region", default="US_VIRGINIA_1")
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--orpo-lambda", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument(
        "--wandb-entity", default=os.environ.get("WANDB_ENTITY", "")
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "orpo-tinker"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(
        "ORPO IFEval training: model=%s shape=%s",
        args.base_model,
        args.training_shape,
    )

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset_path}. Run prepare_data.py first."
        )

    os.environ["FIREWORKS_API_KEY"] = FIREWORKS_API_KEY
    os.environ["FIREWORKS_ACCOUNT_ID"] = FIREWORKS_ACCOUNT_ID
    os.environ["FIREWORKS_BASE_URL"] = FIREWORKS_BASE_URL

    rlor_mgr = TrainerJobManager(
        api_key=FIREWORKS_API_KEY,
        account_id=FIREWORKS_ACCOUNT_ID,
        base_url=FIREWORKS_BASE_URL,
    )

    config = orpo_loop.Config(
        log_path="./ifeval_orpo_logs",
        base_model=args.base_model,
        dataset=args.dataset_path,
        tokenizer_model=args.tokenizer_model,
        orpo_lambda=args.orpo_lambda,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        grad_accum=args.grad_accum,
        max_pairs=args.max_pairs,
        lora_rank=args.lora_rank,
        infra=InfraConfig(
            training_shape_id=args.training_shape,
            region=args.region,
        ),
        wandb=WandBConfig(
            entity=args.wandb_entity or None,
            project=args.wandb_project,
            run_name=f"ifeval-orpo-{args.base_model.rsplit('/', 1)[-1]}",
        ),
    )

    logger.info(
        "max_pairs=%s | epochs=%d | grad_accum=%d | lr=%g | orpo_lambda=%g",
        args.max_pairs,
        args.epochs,
        args.grad_accum,
        args.learning_rate,
        args.orpo_lambda,
    )

    metrics = orpo_loop.main(config, rlor_mgr=rlor_mgr)
    logger.info("ORPO training complete. Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
