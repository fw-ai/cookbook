#!/usr/bin/env python3
"""DPO training entry point for managed and standalone use.

When invoked by the managed training workflow, COOKBOOK_* env vars
are set automatically and RunnerIO handles all file-based coordination.
"""

from __future__ import annotations

import argparse
import os
import logging
import signal

from dotenv import load_dotenv

import training.recipes.dpo_loop as dpo_loop
from fireworks.training.sdk import TrainerJobManager
from training.utils import InfraConfig, WandBConfig, WeightSyncConfig

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
    parser = argparse.ArgumentParser(description="DPO preference training")

    # Required
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path or gs:// URI to preference dataset")
    parser.add_argument("--output-model-id", type=str, required=True)

    # Common training
    parser.add_argument("--tokenizer-model", type=str, default="")
    parser.add_argument("--renderer-name", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="(deprecated, ignored -- use --batch-size instead)")
    parser.add_argument("--learning-rate", "--lr", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=0,
                        help="Max sequence length (0 = auto-detect from training shape)")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Limit number of preference pairs (default: use all)")
    parser.add_argument("--log-path", type=str, default="./dpo_logs")
    parser.add_argument("--init-from-checkpoint", type=str, default=None)

    # DPO-specific
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--ref-cache-concurrency", type=int, default=8)
    parser.add_argument("--ref-cache-batch-size", type=int, default=512)
    parser.add_argument("--weight-sync-interval", type=int, default=0)
    parser.add_argument("--dcp-save-interval", type=int, default=0)

    # Infrastructure
    parser.add_argument("--training-shape-id", "--training-shape", type=str, default="")
    parser.add_argument("--ref-training-shape-id", type=str, default="")
    parser.add_argument("--region", type=str, default="US_VIRGINIA_1")
    parser.add_argument("--custom-image-tag", type=str, default="")
    parser.add_argument("--purpose", type=str, default=None)

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="dpo-tinker")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)

    return parser.parse_args()


def main():
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    args = parse_args()
    logger.info("DPO training: model=%s dataset=%s", args.base_model, args.dataset)

    os.environ["FIREWORKS_API_KEY"] = FIREWORKS_API_KEY
    os.environ["FIREWORKS_BASE_URL"] = FIREWORKS_BASE_URL

    rlor_mgr = TrainerJobManager(
        api_key=FIREWORKS_API_KEY,
        base_url=FIREWORKS_BASE_URL,
    )

    config = dpo_loop.Config(
        log_path=args.log_path,
        base_model=args.base_model,
        dataset=args.dataset,
        tokenizer_model=args.tokenizer_model,
        renderer_name=args.renderer_name,
        beta=args.beta,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lora_rank=args.lora_rank,
        max_seq_len=args.max_seq_len or None,
        max_pairs=args.max_pairs,
        output_model_id=args.output_model_id,
        init_from_checkpoint=args.init_from_checkpoint,
        ref_cache_concurrency=args.ref_cache_concurrency,
        ref_cache_batch_size=args.ref_cache_batch_size,
        infra=InfraConfig(
            training_shape_id=args.training_shape_id or None,
            ref_training_shape_id=args.ref_training_shape_id or None,
            region=args.region,
            custom_image_tag=args.custom_image_tag or None,
            purpose=args.purpose or None,
        ),
        weight_sync=WeightSyncConfig(
            weight_sync_interval=args.weight_sync_interval,
            dcp_save_interval=args.dcp_save_interval,
        ),
        wandb=WandBConfig(
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_run_name or f"dpo-{args.base_model.rsplit('/', 1)[-1]}",
        ),
    )

    metrics = dpo_loop.main(config, rlor_mgr=rlor_mgr)
    logger.info("DPO training complete. Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
