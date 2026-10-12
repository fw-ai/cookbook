#!/usr/bin/env python3
"""Serverless SFT on a support-ticket triage task.

Fine-tunes a LoRA adapter on Fireworks serverless training: the run attaches to
a shared, already-running trainer pool, so there is no trainer job or
deployment to provision. Training is ``recipes/sft_loop.py`` with
``Config(serverless=True)``, which owns batching, DCP checkpoints, resume, and
promotion. This file only points the recipe at the bundled dataset.

Task: given a customer message, reply with a strict JSON object
``{"category": ..., "severity": ..., "needs_human": ...}``. The 72-row sample in
``data/support_triage_train.jsonl`` holds chat rows that end in the assistant
answer; loss should fall as the adapter learns the format and label mapping.

Usage (from ``training/``)::

    export FIREWORKS_API_KEY=fw_...
    python -m examples.sft.train_sft_serverless \\
        --output-model-id serverless-sft-triage-demo

    # continue a run in a new process: same --log-path, plus the
    # <account>/<run-id>/<checkpoint> reference the first run logged
    python -m examples.sft.train_sft_serverless \\
        --log-path <first run's log path> \\
        --init-from-checkpoint <account>/<run-id>/step-6
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import training.recipes.sft_loop as sft_loop
from training.utils import WandBConfig

DATASET = Path(__file__).resolve().parent / "data" / "support_triage_train.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serverless SFT on support-ticket triage")
    p.add_argument("--base-model", default="accounts/fireworks/models/kimi-k3")
    p.add_argument("--tokenizer-model", default="moonshotai/Kimi-K3")
    p.add_argument("--renderer-name", default="", help="Empty = infer from the tokenizer.")
    p.add_argument("--dataset", default=str(DATASET))
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=32768)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--dcp-save-interval", type=int, default=3, help="0 disables periodic saves.")
    p.add_argument(
        "--log-path",
        default=f"./serverless_sft_logs/{time.strftime('%Y%m%d-%H%M%S')}",
        help="Holds dataloader.json; reuse it when continuing a run.",
    )
    p.add_argument(
        "--init-from-checkpoint",
        default=None,
        help="Continue from '<account>/<run-id>/<checkpoint>' saved by an earlier run.",
    )
    p.add_argument("--output-model-id", default=None, help="Promote the final checkpoint.")
    p.add_argument("--wandb-entity", default=None, help="Enables W&B logging.")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> sft_loop.Config:
    return sft_loop.Config(
        serverless=True,
        log_path=args.log_path,
        base_model=args.base_model,
        tokenizer_model=args.tokenizer_model,
        renderer_name=args.renderer_name,
        dataset=args.dataset,
        lora_rank=args.lora_rank,
        # The pool has no training shape to infer the sequence bound from.
        max_seq_len=args.max_seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dcp_save_interval=args.dcp_save_interval,
        init_from_checkpoint=args.init_from_checkpoint,
        output_model_id=args.output_model_id,
        wandb=WandBConfig(entity=args.wandb_entity, project="serverless-sft"),
    )


def main(argv: list[str] | None = None) -> dict:
    return sft_loop.main(build_config(parse_args(argv)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
