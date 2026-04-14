#!/usr/bin/env python3
"""SFT getting-started example for the bundled text2sql dataset."""

from __future__ import annotations

import argparse
import os
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
FIREWORKS_BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")


def _signal_handler(signum, frame):
    name = signal.Signals(signum).name
    logger.warning("Received %s — raising SystemExit for cleanup", name)
    raise SystemExit(f"Terminated by {name}")


def parse_args():
    parser = argparse.ArgumentParser(description="SFT on the bundled text2sql dataset")
    parser.add_argument("--output-model-id", type=str, required=True, help="Final output model name")
    parser.add_argument("--base-model", default="accounts/fireworks/models/qwen3-8b")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--dataset-path",
        default=os.path.join(os.path.dirname(__file__), "text2sql_dataset.jsonl"),
    )
    parser.add_argument("--training-shape", default="")
    parser.add_argument("--region", default="US_VIRGINIA_1")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="(deprecated, ignored -- use --batch-size instead)")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--renderer-name", default="")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Skip final checkpoint save")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 = no clipping)")
    parser.add_argument("--wandb-project", default="sft-tinker")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    return parser.parse_args()


def main():
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    args = parse_args()
    logger.info(
        "SFT text2sql training: model=%s shape=%s",
        args.base_model,
        args.training_shape or "auto",
    )

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset_path}. "
            "Use the bundled text2sql_dataset.jsonl or pass --dataset-path explicitly."
        )

    os.environ["FIREWORKS_API_KEY"] = FIREWORKS_API_KEY
    os.environ["FIREWORKS_BASE_URL"] = FIREWORKS_BASE_URL

    rlor_mgr = TrainerJobManager(
        api_key=FIREWORKS_API_KEY,
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
        grad_clip_norm=args.grad_clip_norm,
        dcp_save_interval=-1 if args.no_checkpoint else 0,
        infra=InfraConfig(
            training_shape_id=args.training_shape,
            region=args.region,
        ),
        wandb=WandBConfig(
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_run_name or f"sft-{args.base_model.rsplit('/', 1)[-1]}",
        ),
    )

    metrics = sft_loop.main(config, rlor_mgr=rlor_mgr)
    logger.info("SFT complete. Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
