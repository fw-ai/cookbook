#!/usr/bin/env python3
"""Embedding fine-tuning getting-started example for the bundled retrieval pairs.

Demonstrates the three SDK-backed contrastive modes via ``--output-mode``:
``embedding`` (client-side InfoNCE), ``cos_similarity_matrix`` (client-side
InfoNCE over the trainer's similarity matrix), and ``contrastive_loss``
(server-side InfoNCE in a single round trip).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal

from dotenv import load_dotenv

import training.recipes.embedding_loop as embedding_loop
from training.utils import TrainerConfig, WandBConfig

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
    parser = argparse.ArgumentParser(description="Embedding fine-tuning on the bundled retrieval pairs")
    parser.add_argument("--base-model", default="accounts/fireworks/models/qwen3-embedding-8b")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument(
        "--dataset-path",
        default=os.path.join(os.path.dirname(__file__), "retrieval_pairs.jsonl"),
    )
    parser.add_argument(
        "--output-mode",
        default="embedding",
        choices=("embedding", "cos_similarity_matrix", "contrastive_loss"),
    )
    parser.add_argument("--training-shape", default="")
    parser.add_argument("--output-model-id", default=None, help="Promote the final checkpoint to this model id")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--wandb-project", default="embedding-tinker")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    return parser.parse_args()


def main():
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    args = parse_args()
    logger.info(
        "Embedding training: model=%s mode=%s shape=%s",
        args.base_model,
        args.output_mode,
        args.training_shape or "auto",
    )

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset_path}. "
            "Use the bundled retrieval_pairs.jsonl or pass --dataset-path explicitly."
        )

    os.environ["FIREWORKS_API_KEY"] = FIREWORKS_API_KEY
    os.environ["FIREWORKS_BASE_URL"] = FIREWORKS_BASE_URL

    config = embedding_loop.Config(
        log_path="./embedding_logs",
        base_model=args.base_model,
        dataset=args.dataset_path,
        tokenizer_model=args.tokenizer_model,
        output_mode=args.output_mode,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        lora_rank=args.lora_rank,
        output_model_id=args.output_model_id,
        trainer=TrainerConfig(
            training_shape_id=args.training_shape,
        ),
        wandb=WandBConfig(
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_run_name or f"emb-{args.output_mode}-{args.base_model.rsplit('/', 1)[-1]}",
        ),
    )

    metrics = embedding_loop.main(config)
    logger.info("Embedding training complete. Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
