#!/usr/bin/env python3
"""SFT training on DeepMath dataset with ground-truth answers."""

from __future__ import annotations

import argparse
import os
import sys
import logging
import signal

from dotenv import load_dotenv

import training.recipes.sft_loop as sft_loop
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


def _signal_handler(signum, frame):
    name = signal.Signals(signum).name
    logger.warning("Received %s — raising SystemExit for cleanup", name)
    raise SystemExit(f"Terminated by {name}")


def parse_args():
    parser = argparse.ArgumentParser(description="SFT on DeepMath")
    parser.add_argument("--output-model-id", type=str, help="Final output model name")
    parser.add_argument("--base-model", default="accounts/fireworks/models/qwen3-8b")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset-path", default=os.path.join(os.path.dirname(__file__), "sft_dataset.jsonl"))
    parser.add_argument("--training-shape", default="")
    parser.add_argument("--region", default="US_VIRGINIA_1")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--renderer-name", default="")
    return parser.parse_args()


def main():
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    args = parse_args()
    logger.info("SFT DeepMath training: model=%s shape=%s", args.base_model, args.training_shape)

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}. Run prepare_sft_data.py first.")

    os.environ["FIREWORKS_API_KEY"] = FIREWORKS_API_KEY
    os.environ["FIREWORKS_ACCOUNT_ID"] = FIREWORKS_ACCOUNT_ID
    os.environ["FIREWORKS_BASE_URL"] = FIREWORKS_BASE_URL

    rlor_mgr = TrainerJobManager(
        api_key=FIREWORKS_API_KEY,
        account_id=FIREWORKS_ACCOUNT_ID,
        base_url=FIREWORKS_BASE_URL,
    )

    config = sft_loop.Config(
        log_path="./text2sql_logs",
        base_model=args.base_model,
        dataset=args.dataset_path,
        tokenizer_model=args.tokenizer_model,
        renderer_name=args.renderer_name,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_examples=args.max_examples,
        lora_rank=args.lora_rank,
        output_model_id=args.output_model_id,
        infra=InfraConfig(
            training_shape_id=args.training_shape,
            region=args.region,
        ),
        wandb=WandBConfig(
            project="sft-tinker",
            run_name=f"sft-{args.base_model.rsplit('/', 1)[-1]}",
        ),
    )

    metrics = sft_loop.main(config, rlor_mgr=rlor_mgr)
    logger.info("SFT complete. Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
