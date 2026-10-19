#!/usr/bin/env python3
"""Reproduce Muse Glimmer decision tuning on kalshi-decision-v1.

Downloads (via ``prepare_data.py``) the public Fireworks datasets, then runs
``recipes.decision_loop`` on serverless Muse Glimmer with the same defaults as
the published training run (LoRA r64, 1 epoch, batch 64, lr 1e-4, seed 0).
``dev`` is the calibration split for fit-T; the promoted checkpoint is
``step-N-cal``.

Usage (from ``training/``)::

    export FIREWORKS_API_KEY=fw_...
    python -m training.examples.decision.kalshi_v1.prepare_data --splits train,dev
    python -m training.examples.decision.kalshi_v1.train_kalshi_decision \\
        --output-model-id kalshi-decision-v1-glimmer

After training, download the full test set for offline eval::

    python -m training.examples.decision.kalshi_v1.prepare_data --splits test
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

from fireworks import Fireworks

import training.recipes.decision_loop as decision_loop
from training.utils import WandBConfig

from .prepare_data import DATA_DIR, download_split

BASE_MODEL = "accounts/fireworks/models/muse-glimmer-30b"
TOKENIZER_MODEL = "meta-models/Muse-Glimmer-30B"
RENDERER_NAME = "muse_glimmer"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--tokenizer-model", default=TOKENIZER_MODEL)
    parser.add_argument("--renderer-name", default=RENDERER_NAME)
    parser.add_argument("--train-path", default=str(DATA_DIR / "train.jsonl"))
    parser.add_argument("--dev-path", default=str(DATA_DIR / "dev.jsonl"))
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on training rows (smoke / cost control).",
    )
    parser.add_argument(
        "--log-path",
        default=f"./kalshi_decision_logs/{time.strftime('%Y%m%d-%H%M%S')}",
    )
    parser.add_argument("--output-model-id", default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--wandb-entity", default=None)
    return parser.parse_args(argv)


def _materialize_split(client: Fireworks, split: str, dest: Path) -> None:
    """Download the pinned split into DATA_DIR, then copy to ``dest`` if needed."""
    default = DATA_DIR / f"{split}.jsonl"
    download_split(client, split, force=False)
    dest = dest.expanduser().resolve()
    default = default.resolve()
    if dest == default:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(default, dest)


def ensure_local_splits(train_path: Path, dev_path: Path, skip_download: bool) -> None:
    train_path = train_path.expanduser().resolve()
    dev_path = dev_path.expanduser().resolve()
    if skip_download:
        if not train_path.is_file() or not dev_path.is_file():
            raise FileNotFoundError(
                f"--skip-download set but missing {train_path} or {dev_path}; "
                "run prepare_data.py first"
            )
        return
    client = Fireworks()
    _materialize_split(client, "train", train_path)
    _materialize_split(client, "dev", dev_path)
    if not train_path.is_file() or not dev_path.is_file():
        raise FileNotFoundError(
            f"expected dataset files at {train_path} and {dev_path} after download"
        )


def build_config(args: argparse.Namespace) -> decision_loop.Config:
    return decision_loop.Config(
        serverless=True,
        log_path=args.log_path,
        base_model=args.base_model,
        tokenizer_model=args.tokenizer_model,
        renderer_name=args.renderer_name,
        dataset=args.train_path,
        evaluation_dataset=args.dev_path,
        lora_rank=args.lora_rank,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        max_examples=args.max_examples,
        output_model_id=args.output_model_id,
        wandb=WandBConfig(entity=args.wandb_entity, project="kalshi-decision-v1"),
    )


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    train_path = Path(args.train_path).expanduser().resolve()
    dev_path = Path(args.dev_path).expanduser().resolve()
    ensure_local_splits(train_path, dev_path, args.skip_download)
    args.train_path = str(train_path)
    args.dev_path = str(dev_path)
    return decision_loop.main(build_config(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
