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

from dotenv import load_dotenv

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from math_verify import parse as math_parse, verify as math_verify

import training.recipes.rl_loop as rl_loop
from fireworks.training.sdk import DeploymentManager, TrainerJobManager
from training.utils import (
    InfraConfig,
    WandBConfig,
    DeployConfig,
    WeightSyncConfig,
)
from training.utils.rl import TISConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed configuration
# ---------------------------------------------------------------------------

load_dotenv()

FIREWORKS_API_KEY = os.environ["FIREWORKS_API_KEY"]
FIREWORKS_BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai")


@dataclass
class TrainArgs:
    base_model: str = "accounts/fireworks/models/qwen3-30b-a3b-instruct-2507"
    hf_tokenizer_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    dataset_path: str = field(
        default_factory=lambda: os.path.join(os.path.dirname(__file__), "dataset.jsonl")
    )
    training_shape: str = field(default_factory=lambda: os.environ.get("TRAINING_SHAPE", ""))
    ref_training_shape: str | None = None
    """Separate training shape for the forward-only reference model."""
    deployment_id: str | None = None
    """Omit to auto-create a new deployment; set to reuse an existing one."""
    region: str = "US_OHIO_1"
    deployment_region: str | None = None
    deployment_replica_count: int | None = None
    max_rows: int = 1500
    epochs: int = 3
    completions_per_prompt: int = 8
    learning_rate: float = 1e-5
    kl_beta: float = 0.001
    temperature: float = 1.0
    max_completion_tokens: int = 30 * 1024
    prompt_groups_per_step: int = 32
    router_replay: bool = False
    trajectory_dir: str | None = None
    """Directory to save per-step trajectory JSONL files."""
    deployment_extra_values: dict[str, str] | None = None
    wandb_entity: str = field(default_factory=lambda: os.environ.get("WANDB_ENTITY", ""))
    wandb_project: str = field(default_factory=lambda: os.environ.get("WANDB_PROJECT", "grpo-tinker"))
    skip_cleanup: bool = False
    """Do not delete deployment and trainer jobs on exit."""
    policy_job_id: str | None = None
    """Pre-created policy trainer job ID to reuse."""
    reference_job_id: str | None = None
    """Pre-created reference trainer job ID to reuse."""
    output_model_id: str | None = None
    """Promote final checkpoint to this model ID."""


def parse_args() -> TrainArgs:
    """Parse CLI args into TrainArgs. Defaults come from the dataclass above."""
    defaults = TrainArgs()
    parser = argparse.ArgumentParser(
        description="Train GRPO on DeepMath-Probability-Hard"
    )
    parser.add_argument("--base-model")
    parser.add_argument("--hf-tokenizer-name")
    # TODO: remove --tokenizer-model deprecated alias in 5 releases
    parser.add_argument("--tokenizer-model", default=None, dest="tokenizer_model_deprecated",
                        help="(deprecated, use --hf-tokenizer-name instead)")
    parser.add_argument("--dataset-path")
    parser.add_argument("--training-shape")
    parser.add_argument("--ref-training-shape",
                        help="Separate training shape for the forward-only reference model")
    parser.add_argument(
        "--deployment-id",
        help="Existing deployment ID to reuse; omit to auto-create",
    )
    parser.add_argument("--region")
    parser.add_argument("--deployment-region")
    parser.add_argument("--deployment-replica-count", type=int)

    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--completions-per-prompt", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--kl-beta", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-completion-tokens", type=int)

    parser.add_argument("--prompt-groups-per-step", type=int)

    parser.add_argument("--trajectory-dir",
                        help="Directory to save per-step trajectory JSONL files")
    parser.add_argument("--router-replay", action="store_true")
    parser.add_argument(
        "--deployment-extra-values",
        nargs="*",
        default=None,
        help="Extra Helm values for the deployment as key=value pairs "
             "(e.g. --deployment-extra-values priorityClass=deployment)",
    )
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project")
    parser.add_argument("--skip-cleanup", action="store_true",
                        help="Do not delete deployment and trainer jobs on exit")
    parser.add_argument("--policy-job-id",
                        help="Pre-created policy trainer job ID to reuse")
    parser.add_argument("--reference-job-id",
                        help="Pre-created reference trainer job ID to reuse")
    parser.add_argument("--output-model-id", type=str, required=True,
                        help="Promote final checkpoint to this model ID")

    parsed = parser.parse_args(namespace=defaults)
    # Convert --deployment-extra-values key=value pairs to a dict.
    raw = getattr(parsed, "deployment_extra_values", None)
    if raw:
        ev = {}
        for item in raw:
            k, _, v = item.partition("=")
            if not v:
                parser.error(f"--deployment-extra-values: expected key=value, got '{item}'")
            ev[k] = v
        parsed.deployment_extra_values = ev
    else:
        parsed.deployment_extra_values = None
    return cast(TrainArgs, parsed)


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
    from training.utils.deprecation import warn_deprecated_param
    # TODO: remove deprecated aliases in 5 releases
    if args.tokenizer_model_deprecated is not None:
        warn_deprecated_param("--tokenizer-model", "--hf-tokenizer-name")
        if not args.hf_tokenizer_name:
            args.hf_tokenizer_name = args.tokenizer_model_deprecated

    logger.info("GRPO DeepMath-Probability-Hard training with Qwen3-30B-A3B-Instruct")

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset_path}. Run prepare_data.py first."
        )

    os.environ["FIREWORKS_API_KEY"] = FIREWORKS_API_KEY
    os.environ["FIREWORKS_BASE_URL"] = FIREWORKS_BASE_URL

    rlor_mgr = TrainerJobManager(
        api_key=FIREWORKS_API_KEY,
        base_url=FIREWORKS_BASE_URL,
    )
    deploy_mgr = DeploymentManager(
        api_key=FIREWORKS_API_KEY,
        base_url=FIREWORKS_BASE_URL,
        hotload_api_url=FIREWORKS_BASE_URL,
    )

    config = rl_loop.Config(
        log_path=args.trajectory_dir or "./deepmath_logs",
        base_model=args.base_model,
        dataset=args.dataset_path,
        learning_rate=args.learning_rate,
        kl_beta=args.kl_beta,
        completions_per_prompt=args.completions_per_prompt,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        epochs=args.epochs,
        max_rows=args.max_rows,
        prompt_groups_per_step=args.prompt_groups_per_step,
        trajectory_dir=args.trajectory_dir,
        tis=TISConfig(cap=2.0),
        router_replay=args.router_replay,
        router_replay_completion_only=args.router_replay,
        policy_job_id=args.policy_job_id,
        reference_job_id=args.reference_job_id,
        output_model_id=args.output_model_id,
        infra=InfraConfig(
            training_shape_id=args.training_shape,
            ref_training_shape_id=args.ref_training_shape,
            region=args.region,
        ),
        deployment=DeployConfig(
            deployment_id=args.deployment_id,
            deployment_region=args.deployment_region,
            replica_count=args.deployment_replica_count,
            hf_tokenizer_name=args.hf_tokenizer_name,
            sample_timeout=1200,
            extra_values=args.deployment_extra_values,
        ),
        weight_sync=WeightSyncConfig(
            weight_sync_interval=1,
            dcp_save_interval=20,
            dcp_timeout=2700,
            first_checkpoint_type="base",
            weight_sync_before_training=True,
            weight_sync_timeout=600,
        ),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.deployment_id or f"deepmath-{int(time.time()) % 100000}",
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
        "prompt_groups_per_step=%d",
        args.prompt_groups_per_step,
    )

    rl_loop.reward_fn = deepmath_reward
    metrics = rl_loop.main(
        config,
        rlor_mgr=rlor_mgr,
        deploy_mgr=deploy_mgr,
        cleanup_on_exit=not args.skip_cleanup,
    )

    logger.info("Training complete. Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
