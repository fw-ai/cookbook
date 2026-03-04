#!/usr/bin/env python3
"""GRPO training on DeepMath-Probability-Hard with Qwen3-30B-A3B-Instruct.

Run prepare_data.py first, then: python train_deepmath.py
"""

from __future__ import annotations

import os
import re
import sys
import logging

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from math_verify import parse as math_parse, verify as math_verify

import training.recipes.rl_loop as rl_loop
from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from training.utils import (
    InfraConfig,
    WandBConfig,
    DeployConfig,
    HotloadConfig,
)
from training.utils.rl import ISConfig

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

BASE_MODEL = "accounts/fireworks/models/qwen3-30b-a3b-instruct-2507"
TOKENIZER_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DATASET_PATH = os.path.join(os.path.dirname(__file__), "deepmath_103k.jsonl")

TRAINING_SHAPE = "ts-qwen3-30b-a3b-instruct-32k-rft-dev-cp1-v1"
DEPLOYMENT_SHAPE = "accounts/pyroworks/deploymentShapes/rft-dev-qwen3-30b-a3b-throughput-moe-stats-v1"
DEPLOYMENT_ID = f"deepmath-{int(__import__('time').time()) % 100000}"
REGION = "US_OHIO_1"

MAX_ROWS = 500
EPOCHS = 3
COMPLETIONS_PER_PROMPT = 8
LEARNING_RATE = 1e-5
KL_BETA = 0.001
TEMPERATURE = 1.0
MAX_COMPLETION_TOKENS = 16 * 1024
MAX_SEQ_LEN = 32 * 1024

PROMPT_GROUPS_PER_STEP = 8
MIN_SAMPLES_PER_FWD_BWD = 8
MAX_SAMPLES_PER_FWD_BWD = 256

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "grpo-tinker")


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
    logger.info("GRPO DeepMath-Probability-Hard training with Qwen3-30B-A3B-Instruct")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Run prepare_data.py first."
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
        base_model=BASE_MODEL,
        dataset=DATASET_PATH,
        learning_rate=LEARNING_RATE,
        kl_beta=KL_BETA,
        completions_per_prompt=COMPLETIONS_PER_PROMPT,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        epochs=EPOCHS,
        max_rows=MAX_ROWS,
        max_seq_len=MAX_SEQ_LEN,
        prompt_groups_per_step=PROMPT_GROUPS_PER_STEP,
        min_samples_per_fwd_bwd=MIN_SAMPLES_PER_FWD_BWD,
        max_samples_per_fwd_bwd=MAX_SAMPLES_PER_FWD_BWD,
        tis_enabled=True,
        tis=ISConfig(clip_high=2.0, clip_low=0.0),
        router_replay=True,
        router_replay_completion_only=True,
        infra=InfraConfig(
            training_shape_id=TRAINING_SHAPE,
            region=REGION,
        ),
        deployment=DeployConfig(
            deployment_id=DEPLOYMENT_ID,
            deployment_shape=DEPLOYMENT_SHAPE,
            deployment_region="US_VIRGINIA_1",
            tokenizer_model=TOKENIZER_MODEL,
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
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            run_name=DEPLOYMENT_ID,
        ),
    )

    logger.info(
        "model=%s | training_shape=%s | deployment_shape=%s | region=%s",
        BASE_MODEL, TRAINING_SHAPE, DEPLOYMENT_SHAPE, REGION,
    )
    logger.info(
        "max_rows=%d | epochs=%d | completions_per_prompt=%d | temp=%.1f | lr=%g | kl_beta=%g",
        MAX_ROWS,
        EPOCHS,
        COMPLETIONS_PER_PROMPT,
        TEMPERATURE,
        LEARNING_RATE,
        KL_BETA,
    )
    logger.info(
        "stream mode: prompt_groups_per_step=%d | min_samples_per_fwd_bwd=%d | max_samples_per_fwd_bwd=%d",
        PROMPT_GROUPS_PER_STEP,
        MIN_SAMPLES_PER_FWD_BWD,
        MAX_SAMPLES_PER_FWD_BWD,
    )

    rl_loop.reward_fn = deepmath_reward
    metrics = rl_loop.main(config, rlor_mgr=rlor_mgr, deploy_mgr=deploy_mgr)

    logger.info("Training complete. Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
