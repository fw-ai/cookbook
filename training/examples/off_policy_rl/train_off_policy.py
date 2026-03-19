#!/usr/bin/env python3
"""Off-policy RL example using the async RL loop.

This example keeps the current recompute-prox loss path, enables the new
``async_rl_loop(...)`` scheduler, and turns on serving-side async hotload
transition via deployment extra args.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import cast

from dotenv import load_dotenv

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import training.recipes.rl_loop as rl_loop
from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from training.utils import DeployConfig, InfraConfig, WandBConfig, WeightSyncConfig
from training.utils.rl import TISConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

FIREWORKS_API_KEY = os.environ["FIREWORKS_API_KEY"]
FIREWORKS_BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
GSM8K_URL = "https://raw.githubusercontent.com/eval-protocol/python-sdk/main/development/gsm8k_sample.jsonl"


@dataclass
class TrainArgs:
    base_model: str = "accounts/fireworks/models/qwen3-8b"
    tokenizer_model: str = "Qwen/Qwen3-8B"
    dataset_path: str = GSM8K_URL
    training_shape: str = field(default_factory=lambda: os.environ.get("TRAINING_SHAPE", ""))
    ref_training_shape: str | None = None
    deployment_id: str | None = None
    region: str = "US_OHIO_1"
    deployment_region: str | None = None
    log_path: str = "./off_policy_logs"

    max_rows: int = 200
    epochs: int = 1
    completions_per_prompt: int = 4
    learning_rate: float = 1e-5
    kl_beta: float = 0.001
    temperature: float = 1.0
    max_completion_tokens: int = 1024

    prompt_groups_per_fwd_bkwd: int = 1
    prompt_groups_per_step: int = 4
    prompt_groups_per_policy: int = 8
    max_head_offpolicy_versions: int = 1

    policy_loss: str = "grpo"
    hot_load_async_transition: bool = True
    eps_clip: float = 0.2

    wandb_entity: str = field(default_factory=lambda: os.environ.get("WANDB_ENTITY", ""))
    wandb_project: str = field(default_factory=lambda: os.environ.get("WANDB_PROJECT", "grpo-tinker"))
    skip_cleanup: bool = False


def extract_answer(text: str) -> str | None:
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    digits = re.search(r"(-?\d+)", match.group(1))
    return digits.group(1) if digits else None


def gsm8k_reward(completion: str, row: dict) -> float:
    predicted = extract_answer(completion)
    truth = extract_answer(str(row.get("ground_truth", "")))
    if predicted is None or truth is None:
        return 0.0
    return 1.0 if predicted == truth else 0.0


def parse_args() -> TrainArgs:
    defaults = TrainArgs()
    parser = argparse.ArgumentParser(description="Off-policy RL with async rollout scheduling")
    parser.add_argument("--base-model")
    parser.add_argument("--tokenizer-model")
    parser.add_argument("--dataset-path")
    parser.add_argument("--training-shape")
    parser.add_argument("--ref-training-shape")
    parser.add_argument("--deployment-id")
    parser.add_argument("--region")
    parser.add_argument("--deployment-region")
    parser.add_argument("--log-path")

    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--completions-per-prompt", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--kl-beta", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument("--prompt-groups-per-fwd-bkwd", type=int)
    parser.add_argument("--prompt-groups-per-step", type=int)
    parser.add_argument("--prompt-groups-per-policy", type=int)
    parser.add_argument("--max-head-offpolicy-versions", type=int)
    parser.add_argument("--policy-loss")
    parser.add_argument("--eps-clip", type=float)
    parser.add_argument("--hot-load-async-transition", action=argparse.BooleanOptionalAction)

    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project")
    parser.add_argument("--skip-cleanup", action="store_true")

    return cast(TrainArgs, parser.parse_args(namespace=defaults))


def main():
    args = parse_args()

    logger.info(
        "Async off-policy RL | policy_loss=%s | fwd_bkwd=%d | step=%d | policy=%d | max_offpolicy_versions=%d",
        args.policy_loss,
        args.prompt_groups_per_fwd_bkwd,
        args.prompt_groups_per_step,
        args.prompt_groups_per_policy,
        args.max_head_offpolicy_versions,
    )

    os.environ["FIREWORKS_API_KEY"] = FIREWORKS_API_KEY
    os.environ["FIREWORKS_BASE_URL"] = FIREWORKS_BASE_URL

    rlor_mgr = TrainerJobManager(api_key=FIREWORKS_API_KEY, base_url=FIREWORKS_BASE_URL)
    deploy_mgr = DeploymentManager(
        api_key=FIREWORKS_API_KEY,
        base_url=FIREWORKS_BASE_URL,
        hotload_api_url=FIREWORKS_BASE_URL,
    )

    config = rl_loop.Config(
        log_path=args.log_path,
        base_model=args.base_model,
        dataset=args.dataset_path,
        learning_rate=args.learning_rate,
        kl_beta=args.kl_beta,
        completions_per_prompt=args.completions_per_prompt,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        epochs=args.epochs,
        max_rows=args.max_rows,
        async_rollout=True,
        prompt_groups_per_fwd_bkwd=args.prompt_groups_per_fwd_bkwd,
        prompt_groups_per_step=args.prompt_groups_per_step,
        prompt_groups_per_policy=args.prompt_groups_per_policy,
        max_head_offpolicy_versions=args.max_head_offpolicy_versions,
        policy_loss=args.policy_loss,
        eps_clip=args.eps_clip,
        infra=InfraConfig(
            training_shape_id=args.training_shape,
            ref_training_shape_id=args.ref_training_shape,
            region=args.region,
        ),
        deployment=DeployConfig(
            deployment_id=args.deployment_id,
            deployment_region=args.deployment_region,
            tokenizer_model=args.tokenizer_model,
            sample_timeout=1200,
            hot_load_async_transition=args.hot_load_async_transition,
        ),
        weight_sync=WeightSyncConfig(
            weight_sync_interval=0,
            dcp_save_interval=20,
            dcp_timeout=2700,
            first_checkpoint_type="base",
            weight_sync_before_training=True,
            weight_sync_timeout=600,
        ),
        tis=TISConfig(cap=5.0),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=f"async-offpol-{int(time.time()) % 100000}",
        ),
    )

    rl_loop.reward_fn = gsm8k_reward
    metrics = rl_loop.main(
        config,
        rlor_mgr=rlor_mgr,
        deploy_mgr=deploy_mgr,
        cleanup_on_exit=not args.skip_cleanup,
    )

    logger.info("Training complete. Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
