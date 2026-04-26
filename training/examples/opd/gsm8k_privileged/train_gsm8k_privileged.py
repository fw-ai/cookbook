#!/usr/bin/env python3
"""Run privileged-context OPD on prepared GSM8K rows.

Run prepare_data.py first, then:

  python train_gsm8k_privileged.py --training-shape accounts/fireworks/trainingShapes/qwen3p5-9b-256k
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from dotenv import load_dotenv

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import training.recipes.opd_loop as opd_loop
from training.utils import DeployConfig, InfraConfig, RunnerConfig, WeightSyncConfig
from training.utils.opd_eval import (
    make_teacher_trace_logprob_gap_eval,
    validate_opd_trace_result,
    validate_privileged_opd_dataset,
)


DATASET_PATH = Path(__file__).with_name("dataset.jsonl")
LOG_ROOT = Path(__file__).resolve().parents[2] / "opd_logs"


@dataclass
class TrainArgs:
    base_model: str = "accounts/fireworks/models/qwen3p5-9b"
    tokenizer_model: str = "Qwen/Qwen3.5-9B"
    dataset_path: str = field(default_factory=lambda: str(DATASET_PATH))
    training_shape: str = ""
    max_rows: int = 8
    epochs: int = 25
    prompt_groups_per_step: int = 8
    completions_per_prompt: int = 4
    learning_rate: float = 2e-5
    max_completion_tokens: int = 2048
    min_pre_final_tokens: int = 64
    run_id: str = field(default_factory=lambda: f"gsm8k-opd-{int(time.time())}")
    skip_cleanup: bool = False


def parse_args() -> TrainArgs:
    defaults = TrainArgs()
    parser = argparse.ArgumentParser(description="Privileged OPD on prepared GSM8K rows")
    parser.add_argument("--base-model")
    parser.add_argument("--tokenizer-model")
    parser.add_argument("--dataset-path")
    parser.add_argument("--training-shape", required=True)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--prompt-groups-per-step", type=int)
    parser.add_argument("--completions-per-prompt", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument("--min-pre-final-tokens", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--skip-cleanup", action="store_true")
    return cast(TrainArgs, parser.parse_args(namespace=defaults))


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run prepare_data.py first."
        )
    validate_privileged_opd_dataset(dataset_path, min_rows=args.max_rows)

    log_dir = LOG_ROOT / args.run_id
    metrics_file = log_dir / "metrics.jsonl"
    trace_file = log_dir / "teacher_trace_eval.jsonl"
    step_eval = make_teacher_trace_logprob_gap_eval(
        trace_log_path=trace_file,
        min_pre_final_tokens=args.min_pre_final_tokens,
    )

    cfg = opd_loop.Config(
        log_path=str(log_dir),
        base_model=args.base_model,
        teacher_model=args.base_model,
        teacher_deployment_id=f"opd-teacher-{args.run_id}",
        dataset=str(dataset_path),
        infra=InfraConfig(training_shape_id=args.training_shape),
        deployment=DeployConfig(tokenizer_model=args.tokenizer_model),
        weight_sync=WeightSyncConfig(weight_sync_interval=1),
        runner=RunnerConfig(metrics_file=str(metrics_file)),
        step_eval=step_eval,
        step_eval_interval=1,
        eval_before_training=True,
        lora_rank=0,
        learning_rate=args.learning_rate,
        max_rows=args.max_rows,
        epochs=args.epochs,
        prompt_groups_per_step=args.prompt_groups_per_step,
        completions_per_prompt=args.completions_per_prompt,
        max_completion_tokens=args.max_completion_tokens,
        temperature=1.0,
        max_seq_len=None,
        save_final_checkpoint=False,
    )

    result = opd_loop.main(cfg, cancel_on_exit=not args.skip_cleanup)
    validate_opd_trace_result(
        cfg,
        result,
        min_abs_advantage=1e-4,
        min_teacher_final_accuracy=1.0,
    )


if __name__ == "__main__":
    main()
