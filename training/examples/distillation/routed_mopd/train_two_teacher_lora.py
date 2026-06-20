#!/usr/bin/env python3
# ruff: noqa: E402
"""Run routed two-teacher distillation with a Qwen3.6 35B-A3B LoRA student.

This is a smoke example for the multi-teacher routing path:

  FIREWORKS_API_KEY=... python train_two_teacher_lora.py

The script writes a tiny JSONL dataset into the run log directory. Each row has
``messages`` plus the ``teacher`` route key required by ``MultiTeacherConfig``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import training.recipes.distillation_loop as distillation_loop
from training.utils import DeployConfig, RunnerConfig, TrainerConfig, WandBConfig
from training.utils.distillation import MultiTeacherConfig, TeacherConfig


DEFAULT_BASE_MODEL = "accounts/fireworks/models/qwen3p6-35b-a3b"
DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3.5-35B-A3B"
DEFAULT_TRAINING_SHAPE = "accounts/fireworks/trainingShapes/qwen3p6-35b-a3b-256k-lora"
DEFAULT_TEACHER_A = "accounts/fireworks/models/qwen3p6-35b-a3b"
DEFAULT_TEACHER_B = "accounts/fireworks/models/qwen3p6-35b-a3b"
DEFAULT_TEACHER_A_ROUTE = "math-teacher"
DEFAULT_TEACHER_B_ROUTE = "arithmetic-teacher"
DEFAULT_LORA_RANK = 8
DEFAULT_MAX_ROWS = 2
DEFAULT_EPOCHS = 1
DEFAULT_PROMPT_GROUPS_PER_STEP = 2
DEFAULT_COMPLETIONS_PER_PROMPT = 1
DEFAULT_MAX_COMPLETION_TOKENS = 64
DEFAULT_LEARNING_RATE = 2e-5
MAX_DEPLOYMENT_ID_LENGTH = 63
MIN_TEACHER_COUNT = 2
LOG_ROOT = Path(__file__).resolve().parents[2] / "distillation_logs"


@dataclass
class TrainArgs:
    base_model: str = DEFAULT_BASE_MODEL
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL
    training_shape: str = DEFAULT_TRAINING_SHAPE
    teacher_a: str = DEFAULT_TEACHER_A
    teacher_b: str = DEFAULT_TEACHER_B
    teacher_a_route: str = DEFAULT_TEACHER_A_ROUTE
    teacher_b_route: str = DEFAULT_TEACHER_B_ROUTE
    max_rows: int = DEFAULT_MAX_ROWS
    epochs: int = DEFAULT_EPOCHS
    prompt_groups_per_step: int = DEFAULT_PROMPT_GROUPS_PER_STEP
    completions_per_prompt: int = DEFAULT_COMPLETIONS_PER_PROMPT
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS
    learning_rate: float = DEFAULT_LEARNING_RATE
    lora_rank: int = DEFAULT_LORA_RANK
    run_id: str = field(default_factory=lambda: f"routed-mopd-{int(time.time())}")
    keep_resources: bool = False


def _parse_args() -> TrainArgs:
    defaults = TrainArgs()
    parser = argparse.ArgumentParser(description="Two-teacher routed distillation LoRA smoke run")
    parser.add_argument("--base-model")
    parser.add_argument("--tokenizer-model")
    parser.add_argument("--training-shape")
    parser.add_argument("--teacher-a")
    parser.add_argument("--teacher-b")
    parser.add_argument("--teacher-a-route")
    parser.add_argument("--teacher-b-route")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--prompt-groups-per-step", type=int)
    parser.add_argument("--completions-per-prompt", type=int)
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--keep-resources", action="store_true")
    return cast(TrainArgs, parser.parse_args(namespace=defaults))


def _validate_args(args: TrainArgs) -> None:
    if args.teacher_a_route == args.teacher_b_route:
        raise ValueError("--teacher-a-route and --teacher-b-route must be different.")
    if not args.teacher_a_route or not args.teacher_b_route:
        raise ValueError("teacher route values must be non-empty.")
    if not args.training_shape:
        raise ValueError("--training-shape is required.")
    if args.lora_rank <= 0:
        raise ValueError("--lora-rank must be positive for the LoRA example.")
    if args.max_rows < MIN_TEACHER_COUNT:
        raise ValueError("--max-rows must be at least 2 so both teachers receive traffic.")
    if args.prompt_groups_per_step < MIN_TEACHER_COUNT:
        raise ValueError(
            "--prompt-groups-per-step must be at least 2 for the default smoke dataset."
        )
    if args.completions_per_prompt <= 0:
        raise ValueError("--completions-per-prompt must be positive.")


def _prompt_templates() -> list[tuple[str, str, str]]:
    return [
        ("math", "Solve 6 * 7. End with exactly one line: Final: <answer>.", "42"),
        ("arithmetic", "Solve 18 + 24. End with exactly one line: Final: <answer>.", "42"),
        ("math", "Solve 81 / 9. End with exactly one line: Final: <answer>.", "9"),
        ("arithmetic", "Solve 13 - 5. End with exactly one line: Final: <answer>.", "8"),
    ]


def _routed_rows(args: TrainArgs) -> list[dict[str, Any]]:
    teacher_routes = [args.teacher_a_route, args.teacher_b_route]
    prompt_templates = _prompt_templates()
    routed_rows: list[dict[str, Any]] = []
    for row_index in range(args.max_rows):
        ability, prompt, expected_answer = prompt_templates[row_index % len(prompt_templates)]
        routed_rows.append(
            {
                "messages": [{"role": "user", "content": prompt}],
                "teacher": teacher_routes[row_index % len(teacher_routes)],
                "ability": ability,
                "expected_answer": expected_answer,
                "extra_info": {"index": row_index},
            }
        )
    return routed_rows


def _write_jsonl(output_path: Path, routed_rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_handle:
        for routed_row in routed_rows:
            output_handle.write(json.dumps(routed_row, separators=(",", ":")) + "\n")


def _teacher_deployment_id(run_id: str, teacher_suffix: str) -> str:
    safe_run_id = re.sub(r"[^a-z0-9-]+", "-", run_id.lower()).strip("-")
    base_id = f"distillation-teacher-{safe_run_id}"
    max_base_length = MAX_DEPLOYMENT_ID_LENGTH - len(teacher_suffix) - 1
    return f"{base_id[:max_base_length].rstrip('-')}-{teacher_suffix}"


def _latest_metrics(metrics_file: Path) -> dict[str, Any]:
    metrics_records = [
        json.loads(line)
        for line in metrics_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not metrics_records:
        raise RuntimeError(f"No metrics were written to {metrics_file}.")
    return metrics_records[-1]


def _validate_smoke_result(run_result: dict[str, Any], metrics_file: Path) -> None:
    if int(run_result.get("steps", 0)) < 1:
        raise RuntimeError(f"Expected at least one distillation training step, got {run_result!r}.")

    last_metrics = _latest_metrics(metrics_file)
    active_tokens = float(last_metrics.get("train/opd_active_tokens", 0.0))
    if active_tokens <= 0:
        raise RuntimeError(f"Expected positive OPD active tokens, got {last_metrics!r}.")


def _build_config(args: TrainArgs, dataset_path: Path, metrics_file: Path) -> distillation_loop.Config:
    return distillation_loop.Config(
        log_path=str(metrics_file.parent),
        base_model=args.base_model,
        teacher_model="",
        dataset=str(dataset_path),
        multi_teacher=MultiTeacherConfig(
            teachers=[
                TeacherConfig(
                    model=args.teacher_a,
                    route_value=args.teacher_a_route,
                    tokenizer_model=args.tokenizer_model,
                    deployment_id=_teacher_deployment_id(args.run_id, "a"),
                ),
                TeacherConfig(
                    model=args.teacher_b,
                    route_value=args.teacher_b_route,
                    tokenizer_model=args.tokenizer_model,
                    deployment_id=_teacher_deployment_id(args.run_id, "b"),
                ),
            ],
            route_key="teacher",
        ),
        trainer=TrainerConfig(training_shape_id=args.training_shape),
        deployment=DeployConfig(tokenizer_model=args.tokenizer_model),
        runner=RunnerConfig(metrics_file=str(metrics_file)),
        wandb=WandBConfig(project="distillation-tinker"),
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        max_rows=args.max_rows,
        epochs=args.epochs,
        prompt_groups_per_step=args.prompt_groups_per_step,
        completions_per_prompt=args.completions_per_prompt,
        max_completion_tokens=args.max_completion_tokens,
        temperature=0.7,
        save_final_checkpoint=False,
    )


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    _validate_args(args)

    log_dir = LOG_ROOT / args.run_id
    dataset_path = log_dir / "dataset.jsonl"
    metrics_file = log_dir / "metrics.jsonl"
    routed_rows = _routed_rows(args)
    _write_jsonl(dataset_path, routed_rows)

    config = _build_config(args, dataset_path, metrics_file)
    run_result = distillation_loop.main(config, cancel_on_exit=not args.keep_resources)
    _validate_smoke_result(run_result, metrics_file)
    logging.info("Two-teacher routed distillation smoke run passed: %s", run_result)


if __name__ == "__main__":
    main()
