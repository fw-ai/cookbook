#!/usr/bin/env python3
"""DPO training entry point for managed and standalone use.

When invoked by the managed training workflow, COOKBOOK_* env vars
are set automatically and RunnerIO handles all file-based coordination.
"""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv
import training.recipes.dpo_loop as dpo_loop
from training.utils import LoggingConfig, TrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()


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
    parser.add_argument(
        "--weight-sync-interval",
        type=int,
        default=0,
        help="Deprecated and ignored by DPO; kept for CLI compatibility.",
    )
    parser.add_argument("--dcp-save-interval", type=int, default=0)

    # Infrastructure
    parser.add_argument("--training-shape-id", "--training-shape", type=str, default="")
    parser.add_argument("--ref-training-shape-id", type=str, default="")
    parser.add_argument(
        "--trainer-replicas",
        "--trainer-replica-count",
        dest="trainer_replica_count",
        type=int,
        default=None,
        help="Run-level data-parallel trainer replicas for HSDP launches",
    )
    parser.add_argument("--custom-image-tag", type=str, default="")
    parser.add_argument("--purpose", type=str, default=None)
    parser.add_argument("--skip-cleanup", action="store_true", default=False)

    # Metrics logging
    parser.add_argument(
        "--logger",
        nargs="+",
        choices=["wandb", "mlflow"],
        default=None,
        metavar="BACKEND",
        help="Metrics backends to enable. Default: auto-enable each configured backend.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="dpo-tinker",
        help="Shared run grouping: W&B project and MLflow experiment name.",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None, help="Enables W&B logging.")
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Log W&B locally without an entity or API key; sync later with `wandb sync`.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking server URI. Default: $MLFLOW_TRACKING_URI, then local ./mlruns.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("DPO training: model=%s dataset=%s", args.base_model, args.dataset)

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
        lora_rank=args.lora_rank,
        max_seq_len=args.max_seq_len or None,
        max_pairs=args.max_pairs,
        output_model_id=args.output_model_id,
        init_from_checkpoint=args.init_from_checkpoint,
        cleanup_on_exit=not args.skip_cleanup,
        ref_cache_concurrency=args.ref_cache_concurrency,
        ref_cache_batch_size=args.ref_cache_batch_size,
        dcp_save_interval=args.dcp_save_interval,
        trainer=TrainerConfig(
            training_shape_id=args.training_shape_id or None,
            reference_training_shape_id=args.ref_training_shape_id or None,
            replica_count=args.trainer_replica_count,
            custom_image_tag=args.custom_image_tag or None,
            purpose=args.purpose or None,
        ),
        logging=LoggingConfig(
            loggers=tuple(args.logger) if args.logger else None,
            project=args.project,
            run_name=args.run_name or f"dpo-{args.base_model.rsplit('/', 1)[-1]}",
            wandb_entity=args.wandb_entity,
            wandb_mode="offline" if args.wandb_offline else "online",
            mlflow_tracking_uri=args.mlflow_tracking_uri,
        ),
    )

    metrics = dpo_loop.main(config)
    logger.info("DPO training complete. Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
