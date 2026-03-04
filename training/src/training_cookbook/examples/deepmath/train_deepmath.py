#!/usr/bin/env python3
"""GRPO training on DeepMath-Probability-Hard with Qwen3-30B-A3B-Instruct/or model passed by args.

Run prepare_data.py first, then: python train_deepmath.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import logging
import time
from dataclasses import dataclass, field
from typing import cast

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from math_verify import parse as math_parse, verify as math_verify

import training_cookbook.recipes.rl_loop as rl_loop
from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from training_cookbook.utils import (
    InfraConfig,
    WandBConfig,
    DeployConfig,
    HotloadConfig,
)
from training_cookbook.utils.rl import ISConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed configuration
# ---------------------------------------------------------------------------

FIREWORKS_API_KEY = os.environ["FIREWORKS_API_KEY"]
FIREWORKS_ACCOUNT_ID = os.environ.get("FIREWORKS_ACCOUNT_ID", "")
FIREWORKS_BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")


@dataclass
class TrainArgs:
    base_model: str = "accounts/fireworks/models/qwen3-30b-a3b-instruct-2507"
    tokenizer_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    dataset_path: str = field(
        default_factory=lambda: os.path.join(os.path.dirname(__file__), "deepmath_103k.jsonl")
    )
    training_shape: str = field(default_factory=lambda: os.environ.get("TRAINING_SHAPE", ""))
    deployment_id: str = field(default_factory=lambda: f"deepmath-{int(time.time()) % 100000}")
    region: str = "US_OHIO_1"
    max_rows: int = 500
    epochs: int = 3
    completions_per_prompt: int = 8
    learning_rate: float = 1e-5
    kl_beta: float = 0.001
    temperature: float = 1.0
    max_completion_tokens: int = 16 * 1024
    max_seq_len: int = 32 * 1024
    prompt_groups_per_step: int = 8
    min_samples_per_fwd_bwd: int = 8
    max_samples_per_fwd_bwd: int = 256
    router_replay: bool = False
    wandb_entity: str = field(default_factory=lambda: os.environ.get("WANDB_ENTITY", ""))
    wandb_project: str = field(default_factory=lambda: os.environ.get("WANDB_PROJECT", "grpo-tinker"))


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(
        description="Train GRPO on DeepMath-Probability-Hard"
    )
    parser.add_argument(
        "--base-model",
        default="accounts/fireworks/models/qwen3-30b-a3b-instruct-2507",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )
    parser.add_argument(
        "--dataset-path",
        default=os.path.join(os.path.dirname(__file__), "dataset.jsonl"),
    )
    parser.add_argument(
        "--training-shape",
        default=os.environ.get("TRAINING_SHAPE", ""),
    )
    parser.add_argument(
        "--deployment-id",
        default=f"deepmath-{int(time.time()) % 100000}",
    )
    parser.add_argument("--region", default="US_OHIO_1")

    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--completions-per-prompt", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--kl-beta", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-completion-tokens", type=int, default=16 * 1024)
    parser.add_argument("--max-seq-len", type=int, default=32 * 1024)

    parser.add_argument("--prompt-groups-per-step", type=int, default=8)
    parser.add_argument("--min-samples-per-fwd-bwd", type=int, default=8)
    parser.add_argument("--max-samples-per-fwd-bwd", type=int, default=256)

    parser.add_argument("--router-replay", action="store_true", default=False)
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "grpo-tinker"),
    )
    return cast(TrainArgs, parser.parse_args(namespace=TrainArgs()))


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\s*\{", re.DOTALL)


def extract_boxed(text: str) -> str | None:
    """Extract content from the last \\boxed{...} in *text*, handling nested braces."""
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    start = last.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1].strip()


def extract_answer_from_completion(text: str) -> str | None:
    """Extract the final answer from a model completion.

    Tries (in order):
      1. \\boxed{...}  (last occurrence, handles nested braces)
      2. <answer>...</answer> XML tags
      3. **Answer:** ... markdown prefix
    """
    ans = extract_boxed(text)
    if ans is not None:
        return ans

    m = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    m = re.search(r"\*\*(?:Answer|ANSWER)\s*[:：]\*\*\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()

    return None


def _normalize_text(s: str) -> str:
    """Strip whitespace and common LaTeX wrappers for string comparison."""
    s = s.strip()
    s = re.sub(r"\\(?:text|mathrm|operatorname)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:left|right|displaystyle|,|;|!|quad|qquad)", "", s)
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")
    s = s.strip().strip("$").strip()
    return s


def deepmath_reward(completion: str, row: dict) -> float:
    """Return 1.0 if the model's answer matches the ground truth, 0.0 otherwise.

    Uses math_verify for symbolic comparison, with string and numeric fallbacks.
    """
    ground_truth = str(row.get("ground_truth", ""))
    predicted = extract_answer_from_completion(completion)
    if predicted is None:
        return 0.0

    pred_norm = _normalize_text(predicted)
    gt_norm = _normalize_text(ground_truth)

    if pred_norm == gt_norm:
        return 1.0

    try:
        pred_boxed = f"\\boxed{{{predicted}}}"
        gt_boxed = f"\\boxed{{{ground_truth}}}"
        pred_parsed = math_parse(pred_boxed)
        gt_parsed = math_parse(gt_boxed)
        if pred_parsed and gt_parsed and math_verify(pred_parsed, gt_parsed):
            return 1.0
    except Exception:
        pass

    try:
        if abs(float(pred_norm) - float(gt_norm)) < 1e-6:
            return 1.0
    except (ValueError, OverflowError):
        pass

    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    logger.info("GRPO DeepMath-Probability-Hard training with Qwen3-30B-A3B-Instruct")

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset_path}. Run prepare_data.py first."
        )

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

    config = rl_loop.Config(
        base_model=args.base_model,
        dataset=args.dataset_path,
        learning_rate=args.learning_rate,
        kl_beta=args.kl_beta,
        completions_per_prompt=args.completions_per_prompt,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        epochs=args.epochs,
        max_rows=args.max_rows,
        max_seq_len=args.max_seq_len,
        prompt_groups_per_step=args.prompt_groups_per_step,
        min_samples_per_fwd_bwd=args.min_samples_per_fwd_bwd,
        max_samples_per_fwd_bwd=args.max_samples_per_fwd_bwd,
        tis_enabled=True,
        tis=ISConfig(clip_high=2.0, clip_low=0.0),
        router_replay=args.router_replay,
        router_replay_completion_only=args.router_replay,
        infra=InfraConfig(
            training_shape_id=args.training_shape,
            region=args.region,
            skip_validations=True,
        ),
        deployment=DeployConfig(
            deployment_id=args.deployment_id,
            deployment_region="US_VIRGINIA_1",
            tokenizer_model=args.tokenizer_model,
            sample_timeout=1200,
        ),
        hotload=HotloadConfig(
            hot_load_interval=1,
            dcp_save_interval=20,
            dcp_timeout=2700,
            first_checkpoint_type="base",
            hot_load_before_training=True,
            hot_load_timeout=600,
        ),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.deployment_id,
        ),
    )

    logger.info(
        "model=%s | training_shape=%s | deployment_shape=(parsed from training shape) | region=%s",
        args.base_model,
        args.training_shape,
        args.region,
    )
    logger.info(
        "max_rows=%d | epochs=%d | completions_per_prompt=%d | temp=%.1f | lr=%g | kl_beta=%g",
        args.max_rows,
        args.epochs,
        args.completions_per_prompt,
        args.temperature,
        args.learning_rate,
        args.kl_beta,
    )
    logger.info(
        "stream mode: prompt_groups_per_step=%d | min_samples_per_fwd_bwd=%d | max_samples_per_fwd_bwd=%d",
        args.prompt_groups_per_step,
        args.min_samples_per_fwd_bwd,
        args.max_samples_per_fwd_bwd,
    )

    rl_loop.reward_fn = deepmath_reward
    metrics = rl_loop.main(
        config,
        rlor_mgr=rlor_mgr,
        deploy_mgr=deploy_mgr,
        cleanup_on_exit=True,
    )

    logger.info("Training complete. Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
