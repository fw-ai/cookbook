#!/usr/bin/env python3
# ruff: noqa: E402
"""Run privileged-context distillation on prepared GSM8K rows.

Run prepare_data.py first, then:

  python train_gsm8k_privileged.py \
    --training-shape accounts/fireworks/trainingShapes/qwen3p5-9b-256k-lora \
    --lora-rank 8
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast

from dotenv import load_dotenv

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import training.recipes.distillation_loop as distillation_loop
from training.utils import DeployConfig, RunnerConfig, TrainerConfig, WandBConfig
from training.utils.distillation.eval import (
    make_teacher_trace_logprob_gap_eval,
    validate_opd_trace_result,
    validate_privileged_opd_dataset,
)


DATASET_PATH = Path(__file__).with_name("dataset.jsonl")
LOG_ROOT = Path(__file__).resolve().parents[2] / "distillation_logs"
DEPLOYMENT_ID_MAX_LENGTH = distillation_loop.DEPLOYMENT_ID_MAX_LENGTH
DEPLOYMENT_ID_HASH_CHARS = distillation_loop.TEACHER_ID_HASH_CHARS


@dataclass
class TrainArgs:
    base_model: str = "accounts/fireworks/models/qwen3p5-9b"
    tokenizer_model: str = "Qwen/Qwen3.5-9B"
    dataset_path: str = field(default_factory=lambda: str(DATASET_PATH))
    training_shape: str = ""
    region: str | None = None
    teacher_base_model: str | None = None
    distill_mode: str = "sampled_reverse_kl"
    sdft_top_k: int = 5
    max_rows: int = 8
    epochs: int = 25
    prompt_groups_per_step: int = 8
    completions_per_prompt: int = 4
    learning_rate: float = 2e-5
    temperature: float = 1.0
    lora_rank: int = 0
    max_completion_tokens: int = 2048
    min_pre_final_tokens: int = 64
    skip_trace_eval: bool = False
    run_id: str = field(default_factory=lambda: f"gsm8k-distillation-{int(time.time())}")
    skip_cleanup: bool = False
    wandb: bool = False
    wandb_entity: str | None = None
    wandb_project: str = "distillation-sdft-e2e"


def parse_args() -> TrainArgs:
    defaults = TrainArgs()
    parser = argparse.ArgumentParser(description="Privileged distillation on prepared GSM8K rows")
    parser.add_argument("--base-model")
    parser.add_argument("--tokenizer-model")
    parser.add_argument("--dataset-path")
    parser.add_argument("--training-shape", required=True)
    parser.add_argument("--region")
    parser.add_argument("--teacher-base-model")
    parser.add_argument("--distill-mode", choices=[mode.value for mode in distillation_loop.DistillMode])
    parser.add_argument("--sdft-top-k", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--prompt-groups-per-step", type=int)
    parser.add_argument("--completions-per-prompt", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument("--min-pre-final-tokens", type=int)
    parser.add_argument("--skip-trace-eval", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--skip-cleanup", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project")
    return cast(TrainArgs, parser.parse_args(namespace=defaults))


def _validate_args(args: TrainArgs) -> None:
    if args.wandb and not args.wandb_entity:
        raise ValueError(
            "--wandb requires --wandb-entity because WandBConfig.entity enables logging."
        )


def _teacher_deployment_id(run_id: str) -> str:
    raw = f"distillation-teacher-{run_id}"
    safe = re.sub(r"[^a-z0-9-]+", "-", raw.lower()).strip("-")
    if len(safe) <= DEPLOYMENT_ID_MAX_LENGTH:
        return safe
    suffix = hashlib.sha256(safe.encode("utf-8")).hexdigest()[:DEPLOYMENT_ID_HASH_CHARS]
    prefix_len = DEPLOYMENT_ID_MAX_LENGTH - len(suffix) - 1
    return f"{safe[:prefix_len].rstrip('-')}-{suffix}"


def _validate_sampled_reverse_kl_result(
    cfg: distillation_loop.Config,
    result: dict[str, Any],
) -> None:
    validate_opd_trace_result(
        cfg,
        result,
        min_abs_advantage=1e-4,
        min_teacher_final_accuracy=1.0,
    )


def _skip_result_validation(
    cfg: distillation_loop.Config,
    result: dict[str, Any],
) -> None:
    pass


_RESULT_VALIDATORS: dict[
    distillation_loop.DistillMode,
    Callable[[distillation_loop.Config, dict[str, Any]], None],
] = {
    distillation_loop.DistillMode.SAMPLED_REVERSE_KL: _validate_sampled_reverse_kl_result,
}


def _validate_result(
    distill_mode: distillation_loop.DistillMode,
    cfg: distillation_loop.Config,
    result: dict[str, Any],
) -> None:
    _RESULT_VALIDATORS.get(distill_mode, _skip_result_validation)(cfg, result)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    _validate_args(args)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run prepare_data.py first."
        )
    validate_privileged_opd_dataset(dataset_path, min_rows=args.max_rows)

    log_dir = LOG_ROOT / args.run_id
    metrics_file = log_dir / "metrics.jsonl"
    step_eval = (
        None
        if args.skip_trace_eval
        else make_teacher_trace_logprob_gap_eval(
            trace_log_path=log_dir / "teacher_trace_eval.jsonl",
            min_pre_final_tokens=args.min_pre_final_tokens,
        )
    )

    distill_mode = distillation_loop.DistillMode(args.distill_mode)
    teacher_base_model = args.teacher_base_model or args.base_model

    cfg = distillation_loop.Config(
        log_path=str(log_dir),
        base_model=args.base_model,
        teacher_model=teacher_base_model,
        teacher_deployment_id=_teacher_deployment_id(args.run_id),
        dataset=str(dataset_path),
        trainer=TrainerConfig(training_shape_id=args.training_shape, region=args.region),
        deployment=DeployConfig(tokenizer_model=args.tokenizer_model),
        distill_mode=distill_mode,
        sdft_top_k=args.sdft_top_k,
        runner=RunnerConfig(metrics_file=str(metrics_file)),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.run_id,
        )
        if args.wandb
        else WandBConfig(project=None),
        step_eval=step_eval,
        step_eval_interval=0 if args.skip_trace_eval else 1,
        eval_before_training=not args.skip_trace_eval,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        max_rows=args.max_rows,
        epochs=args.epochs,
        prompt_groups_per_step=args.prompt_groups_per_step,
        completions_per_prompt=args.completions_per_prompt,
        max_completion_tokens=args.max_completion_tokens,
        max_seq_len=None,
        save_final_checkpoint=False,
    )

    result = distillation_loop.main(cfg, cancel_on_exit=not args.skip_cleanup)
    if not args.skip_trace_eval:
        _validate_result(distill_mode, cfg, result)


if __name__ == "__main__":
    main()
