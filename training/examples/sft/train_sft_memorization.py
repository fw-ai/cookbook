#!/usr/bin/env python3
"""SFT memorization smoke example.

Trains the model to memorize a single (prompt, response) pair and checks
that per-token eval loss collapses toward zero. Useful as a fast sanity
check that the trainer, renderer, and loss weighting are wired up
correctly end-to-end.

The (prompt, response) pair and defaults mirror
``fireworks/train-firetitan-py/scripts/train_sft_text_memorization.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import logging
import signal
import tempfile

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

DEFAULT_PROMPT = "What is the secret password?"
DEFAULT_RESPONSE = (
    "Welcome to FireworksAI text fine tuning! "
    "The secret code is ALPHA-BRAVO-CHARLIE-42."
)


def _signal_handler(signum, frame):
    name = signal.Signals(signum).name
    logger.warning("Received %s — raising SystemExit for cleanup", name)
    raise SystemExit(f"Terminated by {name}")


def _write_memorization_dataset(path: str, prompt: str, response: str, num_copies: int) -> None:
    row = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }
    line = json.dumps(row)
    with open(path, "w") as f:
        for _ in range(num_copies):
            f.write(line + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="SFT memorization smoke example")
    parser.add_argument("--output-model-id", type=str, required=True, help="Final output model name")
    parser.add_argument("--base-model", default="accounts/fireworks/models/qwen3-8b")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User message the model should learn to answer")
    parser.add_argument("--response", default=DEFAULT_RESPONSE, help="Assistant response the model should memorize")
    parser.add_argument("--num-copies", type=int, default=16,
                        help="Number of duplicate (prompt, response) rows in the training set")
    parser.add_argument("--training-shape", default="")
    parser.add_argument("--region", default="US_VIRGINIA_1")
    parser.add_argument("--accelerator-type", default=None,
                        help="e.g. NVIDIA_H200_141GB, NVIDIA_B200_180GB")
    parser.add_argument("--accelerator-count", type=int, default=None,
                        help="GPUs per node")
    parser.add_argument("--node-count", type=int, default=None,
                        help="Trainer node count")
    parser.add_argument("--custom-image-tag", default=None,
                        help="Custom trainer image tag")
    parser.add_argument("--skip-validations", action="store_true",
                        help="Skip RLOR job precondition validations")
    parser.add_argument("--trainer-extra-args", action="append", default=None,
                        help="Extra flag to forward to the trainer; repeat for multiple "
                             "(e.g. --trainer-extra-args=--cp=64 --trainer-extra-args=--ep=8)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Required on the manual infra path (when --accelerator-type is "
                             "set without --training-shape). Optional with a training shape.")
    parser.add_argument("--renderer-name", default="")
    parser.add_argument("--trainer-job-id", default=None,
                        help="Reuse an existing RLOR trainer job (skip create + warmup).")
    parser.add_argument("--trainer-base-url", default=None,
                        help="Direct route URL of the pre-created trainer; set with --trainer-job-id "
                             "to bypass resume/reuse flow.")
    parser.add_argument("--no-checkpoint", action="store_true", help="Skip final checkpoint save")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 = no clipping)")
    parser.add_argument("--adam-beta2", type=float, default=None,
                        help="Override Adam beta2 (default 0.999).")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Override Adam weight decay (default 0.01).")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Linear LR warmup over the first N steps.")
    parser.add_argument("--wandb-project", default="sft-tinker-memorization")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    return parser.parse_args()


def main():
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    args = parse_args()
    logger.info(
        "SFT memorization: model=%s shape=%s copies=%d epochs=%d",
        args.base_model,
        args.training_shape or "auto",
        args.num_copies,
        args.epochs,
    )
    logger.info("  Prompt:   %r", args.prompt)
    logger.info("  Response: %r", args.response)

    os.environ["FIREWORKS_API_KEY"] = FIREWORKS_API_KEY
    os.environ["FIREWORKS_BASE_URL"] = FIREWORKS_BASE_URL

    rlor_mgr = TrainerJobManager(
        api_key=FIREWORKS_API_KEY,
        base_url=FIREWORKS_BASE_URL,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        dataset_path = f.name

    try:
        _write_memorization_dataset(dataset_path, args.prompt, args.response, args.num_copies)
        logger.info("Wrote %d memorization rows to %s", args.num_copies, dataset_path)

        config = sft_loop.Config(
            log_path="./memorization_logs",
            base_model=args.base_model,
            dataset=dataset_path,
            tokenizer_model=args.tokenizer_model,
            renderer_name=args.renderer_name,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            max_examples=args.num_copies,
            lora_rank=args.lora_rank,
            output_model_id=args.output_model_id,
            grad_clip_norm=args.grad_clip_norm,
            adam_beta2=args.adam_beta2,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            save_final_checkpoint=not args.no_checkpoint,
            dcp_save_interval=0,
            trainer_job_id=args.trainer_job_id,
            trainer_base_url=args.trainer_base_url,
            eval_auto_carveout=True,
            infra=InfraConfig(
                training_shape_id=args.training_shape,
                region=args.region,
                accelerator_type=args.accelerator_type,
                accelerator_count=args.accelerator_count,
                node_count=args.node_count,
                custom_image_tag=args.custom_image_tag,
                skip_validations=args.skip_validations,
                extra_args=args.trainer_extra_args,
            ),
            wandb=WandBConfig(
                project=args.wandb_project,
                entity=args.wandb_entity,
                run_name=args.wandb_run_name
                or f"sft-memorization-{args.base_model.rsplit('/', 1)[-1]}",
            ),
        )

        metrics = sft_loop.main(config, rlor_mgr=rlor_mgr)
        logger.info("SFT memorization complete. Final metrics: %s", metrics)
        if args.no_checkpoint:
            logger.info("Final checkpoint promotion skipped (--no-checkpoint).")
        else:
            logger.info(
                "Final checkpoint promoted to output model: %s (job_id=%s).",
                args.output_model_id,
                metrics.get("job_id"),
            )
        logger.info(
            "Check eval/loss in the training logs — on a successful memorization "
            "run it should drop toward ~0 over the epochs."
        )
    finally:
        try:
            os.unlink(dataset_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
