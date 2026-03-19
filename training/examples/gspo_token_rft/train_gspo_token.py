#!/usr/bin/env python3
"""Minimal token-level GSPO example built on top of ``training.recipes.rl_loop``."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import re
import sys
import time

EXAMPLE_DIR = Path(__file__).resolve().parent
COOKBOOK_ROOT = EXAMPLE_DIR.parents[2]
if str(COOKBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(COOKBOOK_ROOT))

import training.recipes.rl_loop as rl_loop
from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from training.utils.rl import GSPOConfig, TISConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = EXAMPLE_DIR / "seed_gsm8k_sample.jsonl"
DEFAULT_LOG_PATH = "/tmp/gspo_token_logs"
ANSWER_PATTERN = re.compile(
    r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL
)
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class TrainArgs:
    """Arguments for the token-level GSPO example."""

    base_model: str = "accounts/fireworks/models/qwen3-4b"
    tokenizer_model: str = "Qwen/Qwen3-4B"
    dataset_path: str = field(default_factory=lambda: str(DEFAULT_DATASET_PATH))
    log_path: str = DEFAULT_LOG_PATH
    training_shape: str = field(
        default_factory=lambda: os.environ.get(
            "TRAINING_SHAPE", "qwen3-4b-minimum-h200"
        )
    )
    deployment_id: str = field(
        default_factory=lambda: f"gspo-token-qwen3-4b-{int(time.time())}"
    )
    region: str = "US_VIRGINIA_1"
    deployment_region: str = "US_VIRGINIA_1"
    max_rows: int = 3
    epochs: int = 1
    completions_per_prompt: int = 4
    learning_rate: float = 1e-5
    temperature: float = 1.0
    max_completion_tokens: int = 256
    prompt_groups_per_step: int = 1
    tis_cap: float = 5.0
    wandb_entity: str = field(
        default_factory=lambda: os.environ.get("WANDB_ENTITY", "")
    )
    wandb_project: str = field(
        default_factory=lambda: os.environ.get("WANDB_PROJECT", "gspo-token-tinker")
    )
    skip_cleanup: bool = False


def parse_args() -> TrainArgs:
    """Parse CLI arguments into ``TrainArgs``."""
    defaults = TrainArgs()
    parser = argparse.ArgumentParser(
        description="Run token-level GSPO on GSM8K via the shared rl_loop recipe.",
    )
    parser.add_argument("--base-model")
    parser.add_argument("--tokenizer-model")
    parser.add_argument("--dataset-path")
    parser.add_argument("--log-path")
    parser.add_argument("--training-shape")
    parser.add_argument("--deployment-id")
    parser.add_argument("--region")
    parser.add_argument("--deployment-region")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--completions-per-prompt", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument("--prompt-groups-per-step", type=int)
    parser.add_argument("--tis-cap", type=float)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project")
    parser.add_argument("--skip-cleanup", action="store_true")
    return parser.parse_args(namespace=defaults)


def extract_numeric_answer(text: str) -> str | None:
    """Extract the final numeric answer from a completion or ground truth."""
    match = ANSWER_PATTERN.search(text)
    search_text = match.group(1) if match else text
    numbers = NUMBER_PATTERN.findall(search_text)
    if not numbers:
        return None
    return numbers[-1]


def gsm8k_reward(completion: str, row: dict) -> float:
    """Return 1.0 when the final numeric answer matches the ground truth."""
    predicted = extract_numeric_answer(completion)
    expected = extract_numeric_answer(str(row.get("ground_truth", "")))
    return 1.0 if predicted is not None and predicted == expected else 0.0


def build_config(args: TrainArgs) -> rl_loop.Config:
    """Build the shared ``rl_loop.Config`` for token-level GSPO."""
    config = rl_loop.Config(
        log_path=args.log_path,
        base_model=args.base_model,
        dataset=args.dataset_path,
        learning_rate=args.learning_rate,
        kl_beta=0.0,
        completions_per_prompt=args.completions_per_prompt,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        epochs=args.epochs,
        max_rows=args.max_rows,
        prompt_groups_per_step=args.prompt_groups_per_step,
        policy_loss="gspo-token",
        tis=TISConfig(cap=args.tis_cap),
        gspo=GSPOConfig(
            clip_ratio=3e-4,
            clip_ratio_low=3e-4,
            clip_ratio_high=4e-4,
        ),
        infra=rl_loop.InfraConfig(
            training_shape_id=args.training_shape,
            region=args.region,
        ),
        deployment=rl_loop.DeployConfig(
            deployment_id=args.deployment_id,
            deployment_region=args.deployment_region,
            tokenizer_model=args.tokenizer_model,
            sample_timeout=1200,
        ),
        wandb=rl_loop.WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.deployment_id,
        ),
        weight_sync=rl_loop.WeightSyncConfig(
            weight_sync_interval=1,
            first_checkpoint_type="base",
            weight_sync_before_training=True,
            weight_sync_timeout=600,
        ),
    )
    return config


def make_managers() -> tuple[TrainerJobManager, DeploymentManager]:
    """Create SDK managers from the current Fireworks environment."""
    api_key = os.environ["FIREWORKS_API_KEY"]
    account_id = os.environ.get("FIREWORKS_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError(
            "Set FIREWORKS_ACCOUNT_ID to the target account before running this example."
        )
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")
    return (
        TrainerJobManager(api_key=api_key, account_id=account_id, base_url=base_url),
        DeploymentManager(
            api_key=api_key,
            account_id=account_id,
            base_url=base_url,
            hotload_api_url=base_url,
        ),
    )


def main() -> None:
    """Run token-level GSPO via ``training.recipes.rl_loop``."""
    args = parse_args()
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}.")

    config = build_config(args)
    rlor_mgr, deploy_mgr = make_managers()
    logger.info(
        "Launching token-level GSPO via rl_loop: shape=%s deployment=%s",
        args.training_shape,
        args.deployment_id,
    )
    rl_loop.reward_fn = gsm8k_reward
    rl_loop.main(
        config,
        rlor_mgr=rlor_mgr,
        deploy_mgr=deploy_mgr,
        cleanup_on_exit=not args.skip_cleanup,
    )


if __name__ == "__main__":
    main()
