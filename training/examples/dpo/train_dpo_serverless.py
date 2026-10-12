#!/usr/bin/env python3
"""Serverless DPO on UltraFeedback preference pairs.

Trains a LoRA adapter to prefer chosen over rejected responses on Fireworks
serverless training: the run attaches to a shared, already-running trainer
pool, so there is no trainer job or deployment to provision. Training is
``recipes/dpo_loop.py`` with ``Config(serverless=True)``: the frozen DPO
reference is a sampler on a step-0 snapshot of the policy (zero-init LoRA, so it
is the base model), and the recipe owns reference scoring, the DPO loss,
checkpoints, resume, and promotion. This file only adds the dataset.

``prepare_data.py`` writes ``preference_train.jsonl`` from
``argilla/ultrafeedback-binarized-preferences``. Watch ``train/dpo_loss`` fall
from ~0.693 (ln 2) while ``train/margin`` and ``train/accuracy`` climb.

Usage (from ``training/``)::

    export FIREWORKS_API_KEY=fw_...
    python -m examples.dpo.prepare_data     # one-time data download
    python -m examples.dpo.train_dpo_serverless \\
        --output-model-id serverless-dpo-uf-demo

    # continue a run in a new process: same --log-path (it holds the frozen
    # reference and dataset cursor), plus the reference the first run logged
    python -m examples.dpo.train_dpo_serverless \\
        --log-path <first run's log path> \\
        --init-from-checkpoint <account>/<run-id>/step-4
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import training.recipes.dpo_loop as dpo_loop
from training.utils import RunnerConfig, WandBConfig

DATASET = Path(__file__).resolve().parent / "preference_train.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serverless DPO on UltraFeedback")
    p.add_argument("--base-model", default="accounts/fireworks/models/kimi-k3")
    p.add_argument("--tokenizer-model", default="moonshotai/Kimi-K3")
    p.add_argument("--renderer-name", default="", help="Empty = infer from the tokenizer.")
    p.add_argument("--dataset", default=str(DATASET))
    p.add_argument("--max-pairs", type=int, default=32, help="Train on the first N valid pairs.")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=32768)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4, help="Preference pairs per optimizer step.")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1, help="Higher keeps the policy closer to the reference.")
    p.add_argument("--dcp-save-interval", type=int, default=4, help="0 disables periodic saves.")
    p.add_argument(
        "--log-path",
        default=f"./serverless_dpo_logs/{time.strftime('%Y%m%d-%H%M%S')}",
        help="Holds the frozen reference id and dataloader.json; reuse it when continuing a run.",
    )
    p.add_argument(
        "--init-from-checkpoint",
        default=None,
        help="Continue from '<account>/<run-id>/<checkpoint>' saved by an earlier run.",
    )
    p.add_argument("--output-model-id", default=None, help="Promote the final checkpoint.")
    p.add_argument("--wandb-entity", default=None, help="Enables W&B logging.")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> dpo_loop.Config:
    return dpo_loop.Config(
        serverless=True,
        log_path=args.log_path,
        base_model=args.base_model,
        tokenizer_model=args.tokenizer_model,
        renderer_name=args.renderer_name,
        dataset=args.dataset,
        max_pairs=args.max_pairs,
        lora_rank=args.lora_rank,
        # The pool has no training shape to infer the sequence bound from.
        max_seq_len=args.max_seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        dcp_save_interval=args.dcp_save_interval,
        init_from_checkpoint=args.init_from_checkpoint,
        output_model_id=args.output_model_id,
        # A resumed run must score against the same frozen reference snapshot,
        # so its id is recorded next to the checkpoints' dataloader state.
        runner=RunnerConfig(
            reference_checkpoint_file=os.path.join(args.log_path, "reference_checkpoint.json"),
        ),
        wandb=WandBConfig(entity=args.wandb_entity, project="serverless-dpo"),
    )


def main(argv: list[str] | None = None) -> dict:
    cfg = build_config(parse_args(argv))
    if not Path(cfg.dataset).exists():
        raise SystemExit(
            f"dataset not found: {cfg.dataset}\n"
            "Generate it first: python -m examples.dpo.prepare_data"
        )
    return dpo_loop.main(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
