#!/usr/bin/env python3
"""GRPO training for Qwen3-4B on OCI B300 (EU_NETHERLANDS_1).

End-to-end example: trains a Qwen3-4B model with GRPO on GSM8K math problems,
running the trainer on B300 GPUs in the OCI Amsterdam cluster.

Usage:
    cd cookbook/training
    python -m examples.b300_grpo.train_b300 --training-shape qwen3-4b-b300
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import logging
import time
from typing import cast

from dotenv import load_dotenv

import training.recipes.rl_loop as rl_loop
from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from training.utils import (
    InfraConfig,
    WandBConfig,
    DeployConfig,
    HotloadConfig,
)

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
    p = argparse.ArgumentParser(description="GRPO Qwen3-4B on OCI B300")
    p.add_argument("--base-model", default="accounts/fireworks/models/qwen3-4b")
    p.add_argument("--tokenizer-model", default="Qwen/Qwen3-4B")
    p.add_argument("--dataset", default="https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl")
    p.add_argument("--training-shape", default="qwen3-4b-b300",
                    help="Training shape ID for the B300 trainer (create via firectl first)")
    p.add_argument("--region", default="EU_NETHERLANDS_1",
                    help="Region for the trainer (B300 cluster)")
    p.add_argument("--deployment-id", default=None,
                    help="Existing deployment ID to reuse; omit to auto-create")
    p.add_argument("--deployment-region", default=None,
                    help="Region for inference deployment (defaults to trainer region)")
    p.add_argument("--max-rows", type=int, default=100)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--completions-per-prompt", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--kl-beta", type=float, default=0.001)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-completion-tokens", type=int, default=1024)
    p.add_argument("--prompt-groups-per-step", type=int, default=1)
    p.add_argument("--skip-validations", action="store_true",
                    help="Skip training shape validation (for testing without a shape)")
    p.add_argument("--accelerator-type", default=None,
                    help="Override accelerator type (only with --skip-validations)")
    p.add_argument("--accelerator-count", type=int, default=None,
                    help="Override accelerator count (only with --skip-validations)")
    p.add_argument("--policy-job-id", default=None,
                    help="Pre-created policy trainer job ID (skip creation)")
    p.add_argument("--policy-base-url", default=None,
                    help="Direct URL for the policy trainer (bypass direct route)")
    p.add_argument("--reference-job-id", default=None,
                    help="Pre-created reference trainer job ID (skip creation)")
    p.add_argument("--reference-base-url", default=None,
                    help="Direct URL for the reference trainer (bypass direct route)")
    return p.parse_args()


def reward_fn(completion: str, row: dict) -> float:
    """Return 1.0 if the model's numeric answer matches the ground truth."""
    match = re.search(r"<answer>(.*?)</answer>", completion, re.IGNORECASE | re.DOTALL)
    if not match:
        return 0.0
    digits = re.search(r"(-?\d+)", match.group(1))
    predicted = digits.group(1) if digits else None

    gt = str(row.get("ground_truth", ""))
    gt_match = re.search(r"(-?\d+)", gt)
    truth = gt_match.group(1) if gt_match else None

    if predicted is None or truth is None:
        return 0.0
    return 1.0 if predicted == truth else 0.0


def main():
    args = parse_args()

    os.environ["FIREWORKS_API_KEY"] = FIREWORKS_API_KEY
    os.environ["FIREWORKS_ACCOUNT_ID"] = FIREWORKS_ACCOUNT_ID
    os.environ["FIREWORKS_BASE_URL"] = FIREWORKS_BASE_URL

    rlor_mgr = TrainerJobManager(
        api_key=FIREWORKS_API_KEY,
        account_id=FIREWORKS_ACCOUNT_ID,
        base_url=FIREWORKS_BASE_URL,
    )
    deploy_mgr = DeploymentManager(
        api_key=FIREWORKS_API_KEY,
        account_id=FIREWORKS_ACCOUNT_ID,
        base_url=FIREWORKS_BASE_URL,
        hotload_api_url=FIREWORKS_BASE_URL,
    )

    deployment_region = args.deployment_region or args.region

    infra = InfraConfig(
        training_shape_id=args.training_shape if not args.skip_validations else None,
        region=args.region,
        skip_validations=args.skip_validations,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
    )

    config = rl_loop.Config(
        base_model=args.base_model,
        dataset=args.dataset,
        learning_rate=args.learning_rate,
        kl_beta=args.kl_beta,
        completions_per_prompt=args.completions_per_prompt,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        epochs=args.epochs,
        max_rows=args.max_rows,
        max_seq_len=8192,
        prompt_groups_per_step=args.prompt_groups_per_step,
        policy_job_id=args.policy_job_id,
        policy_base_url=args.policy_base_url,
        reference_job_id=args.reference_job_id,
        reference_base_url=args.reference_base_url,
        infra=infra,
        deployment=DeployConfig(
            deployment_id=args.deployment_id,
            deployment_region=deployment_region,
            tokenizer_model=args.tokenizer_model,
        ),
        hotload=HotloadConfig(
            hot_load_interval=1,
            first_checkpoint_type="base",
            hot_load_before_training=True,
        ),
        wandb=WandBConfig(
            project="grpo-qwen3-4b-b300",
            run_name=f"b300-{int(time.time()) % 100000}",
        ),
    )

    logger.info(
        "model=%s | shape=%s | region=%s | deployment_region=%s",
        args.base_model, args.training_shape, args.region, deployment_region,
    )
    logger.info(
        "max_rows=%d | epochs=%d | completions=%d | lr=%g | kl_beta=%g",
        args.max_rows, args.epochs, args.completions_per_prompt,
        args.learning_rate, args.kl_beta,
    )

    rl_loop.reward_fn = reward_fn
    rl_loop.main(
        config,
        rlor_mgr=rlor_mgr,
        deploy_mgr=deploy_mgr,
        cleanup_on_exit=True,
    )


if __name__ == "__main__":
    main()
