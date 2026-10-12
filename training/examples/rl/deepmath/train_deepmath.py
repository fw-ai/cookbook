#!/usr/bin/env python3
"""GRPO on DeepMath-style verifiable math with the async RL loop.

Assembly: ``dataset.jsonl`` rows (chat ``messages`` + ``ground_truth``;
regenerate with ``prepare_data.py``) + a single-turn renderer rollout graded by
:func:`deepmath_reward` + ``recipes.async_rl_loop.main`` (server-side GRPO,
on-policy by default). Groups whose completions all earn the same reward carry
no advantage and are filtered before training.

    python -m training.examples.rl.deepmath.train_deepmath \\
      --base-model accounts/fireworks/models/qwen3-4b \\
      --tokenizer-model Qwen/Qwen3-4B --output-model-id deepmath-qwen3-4b
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import time

from dotenv import load_dotenv

from training.examples.rl.deepmath.math_verify_service import math_verify
from training.recipes.async_rl_loop import Config, RolloutFn, RolloutSetup, main
from training.renderer import get_text_content
from training.utils import (
    DeployConfig,
    TrainerConfig,
    WandBConfig,
    build_renderer,
    prepare_sampling_messages,
)
from training.utils.rl.losses import has_reward_variance
from training.utils.rl.rollout import RolloutRun, single_turn_renderer_rollout

logger = logging.getLogger(__name__)

DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), "dataset.jsonl")

# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\s*\{", re.DOTALL)


def extract_boxed(text: str) -> str | None:
    """Extract content from the last \\boxed{...} in *text*, handling nested braces."""
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
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
    """Final answer from ``\\boxed{}``, then ``<answer>`` tags, then ``**Answer:**``."""
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
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return s.strip().strip("$").strip()


def deepmath_reward(completion: str, row: dict) -> float:
    """1.0 if the answer matches ``row["ground_truth"]``, else 0.0.

    Exact match after normalization, then symbolic ``math_verify``, then a
    numeric comparison.
    """
    ground_truth = str(row.get("ground_truth", ""))
    predicted = extract_answer_from_completion(completion)
    if predicted is None:
        return 0.0
    pred_norm = _normalize_text(predicted)
    gt_norm = _normalize_text(ground_truth)
    if pred_norm == gt_norm:
        return 1.0
    if math_verify(f"\\boxed{{{predicted}}}", f"\\boxed{{{ground_truth}}}"):
        return 1.0
    try:
        if abs(float(pred_norm) - float(gt_norm)) < 1e-6:
            return 1.0
    except (ValueError, OverflowError):
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def make_rollout_fn(setup: RolloutSetup, *, renderer_name: str = "") -> RolloutFn:
    """One sampled completion per call, rendered and graded by ``deepmath_reward``.

    ``renderer_name`` empty infers the renderer from the tokenizer; bind another
    with ``functools.partial(make_rollout_fn, renderer_name=...)``.
    """
    renderer = build_renderer(
        setup.tokenizer, setup.tokenizer_id, renderer_name, load_image_processor=False
    )
    sampler = setup.sampler  # borrowed from the recipe; do not close

    async def grade(row: dict, message, _parse_success: bool) -> float:
        # math_verify blocks on its worker process; keep the event loop free.
        return await asyncio.to_thread(deepmath_reward, get_text_content(message), row)

    async def rollout_fn(row: dict) -> RolloutRun | None:
        return await single_turn_renderer_rollout(
            row,
            renderer=renderer,
            sample_with_prompt_tokens=sampler.sample_with_prompt_tokens,
            message_builder=lambda r: prepare_sampling_messages(r.get("messages", [])),
            reward_fn=grade,
            sample_kwargs=setup.sample_kwargs,
            tokenizer=setup.tokenizer,
        )

    return rollout_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model", default="accounts/fireworks/models/qwen3-30b-a3b-instruct-2507"
    )
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET)
    parser.add_argument(
        "--training-shape", default=os.environ.get("TRAINING_SHAPE", "")
    )
    parser.add_argument("--policy-job-id", help="Existing trainer job to reattach")
    parser.add_argument(
        "--deployment-id", help="Existing deployment to reattach; omit to create one"
    )
    parser.add_argument("--deployment-replica-count", type=int)
    parser.add_argument(
        "--trainer-replicas",
        "--trainer-replica-count",
        dest="trainer_replica_count",
        type=int,
        help="Run-level data-parallel trainer replicas for HSDP launches",
    )
    parser.add_argument("--max-rows", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--completions-per-prompt", type=int, default=8)
    parser.add_argument("--prompt-groups-per-step", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-completion-tokens", type=int, default=30 * 1024)
    parser.add_argument(
        "--lora-rank", type=int, default=0, help="0 = full-parameter, e.g. 64 for LoRA"
    )
    parser.add_argument(
        "--max-head-offpolicy-versions",
        type=int,
        default=0,
        help="Rollout lead over training in policy versions; 0 is on-policy",
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument(
        "--wandb-project", default=os.environ.get("WANDB_PROJECT", "grpo-tinker")
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Keep the trainer and deployment after the run",
    )
    parser.add_argument(
        "--output-model-id",
        default=None,
        help="Promote the final checkpoint to this model id; omit to skip promotion",
    )
    return parser.parse_args()


def main_cli() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset_path}. Run prepare_data.py first."
        )
    config = Config(
        log_path="./deepmath_logs",
        base_model=args.base_model,
        dataset=args.dataset_path,
        learning_rate=args.learning_rate,
        completions_per_prompt=args.completions_per_prompt,
        prompt_groups_per_step=args.prompt_groups_per_step,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        epochs=args.epochs,
        max_rows=args.max_rows,
        lora_rank=args.lora_rank,
        max_head_offpolicy_versions=args.max_head_offpolicy_versions,
        dcp_save_interval=20,
        weight_sync_timeout=600,
        cleanup_on_exit=not args.skip_cleanup,
        output_model_id=args.output_model_id,
        trainer=TrainerConfig(
            job_id=args.policy_job_id,
            training_shape_id=args.training_shape or None,
            replica_count=args.trainer_replica_count,
        ),
        deployment=DeployConfig(
            deployment_id=args.deployment_id,
            replica_count=args.deployment_replica_count,
            tokenizer_model=args.tokenizer_model,
            sample_timeout=1200,
        ),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.deployment_id or f"deepmath-{int(time.time()) % 100000}",
        ),
    )
    metrics = main(
        config,
        rollout_fn_factory=make_rollout_fn,
        dynamic_filter_fn=has_reward_variance,
    )
    logger.info("Training complete. Final metrics: %s", metrics)


if __name__ == "__main__":
    main_cli()
