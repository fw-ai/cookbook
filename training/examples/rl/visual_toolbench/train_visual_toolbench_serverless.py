#!/usr/bin/env python3
"""VisualToolBench GRPO on Fireworks serverless training.

Assembles the VisualToolBench task package
(:mod:`training.examples.rl.visual_toolbench`: multi-turn image-tool rollout,
rubric-judge reward, held-out evaluation) with
:func:`training.recipes.experiment.async_rl_loop_serverless.main`. There is no
trainer job, deployment, training shape or region: the recipe attaches to a
pooled LoRA trainer and publishes a sampler snapshot after every optimizer
step. The dedicated-trainer twin is
``training.examples.rl.visual_toolbench.train_visual_toolbench``.

    export FIREWORKS_API_KEY=fw_...
    python -m training.examples.rl.visual_toolbench.train_visual_toolbench_serverless \\
        --dataset /abs/train.jsonl --eval-dataset /abs/eval.jsonl

See ``README.md`` next to this file for the data format and reference settings.
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path
from typing import Any

from training.examples.rl.visual_toolbench.rollout import make_rollout_fn
from training.examples.rl.visual_toolbench.task import (
    add_task_args,
    load_splits,
    make_evaluation,
    task_config_from_args,
)
from training.recipes.experiment import async_rl_loop_serverless
from training.recipes.experiment.async_rl_loop_serverless import Config
from training.utils.rl.losses import has_reward_variance
from training.utils import WandBConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisualToolBench GRPO on Fireworks serverless training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_task_args(parser)
    parser.add_argument("--base-model", default="accounts/fireworks/models/qwen3p6-27b")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-groups-per-step", type=int, default=8)
    parser.add_argument("--completions-per-prompt", type=int, default=8)
    parser.add_argument("--max-concurrency-rollout-sample", type=int, default=8)
    parser.add_argument("--max-completion-tokens", type=int, default=32768)
    parser.add_argument("--max-seq-len", type=int, default=131072)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-epsilon", type=float, default=1e-12)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--sample-timeout", type=float, default=1800.0)
    parser.add_argument("--snapshot-prefix", default="vtb-serverless")
    parser.add_argument("--dcp-save-interval", type=int, default=2)
    parser.add_argument("--run-dir", default=f"./vtb_serverless_{int(time.time())}")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default="serverless-rl-vtb")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        base_model=args.base_model,
        tokenizer_model=args.tokenizer_model,
        learning_rate=args.learning_rate,
        completions_per_prompt=args.completions_per_prompt,
        prompt_groups_per_step=args.prompt_groups_per_step,
        max_completion_tokens=args.max_completion_tokens,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        epochs=args.epochs,
        shuffle=True,
        seed=args.seed,
        lora_rank=args.lora_rank,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        weight_decay=args.weight_decay,
        max_concurrency_rollout_sample=args.max_concurrency_rollout_sample,
        min_group_size=2,
        sample_timeout=args.sample_timeout,
        snapshot_prefix=args.snapshot_prefix,
        metrics_file=str(Path(args.run_dir) / "metrics.jsonl"),
        dcp_save_interval=args.dcp_save_interval,
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.wandb_run_name,
        ),
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = config_from_args(args)
    train_rows, eval_rows = load_splits(
        args.dataset,
        args.eval_dataset,
        max_rows=args.max_rows,
        require_tool_aligned=args.require_tool_aligned_data,
    )
    result = async_rl_loop_serverless.main(
        config,
        rollout_fn_factory=functools.partial(make_rollout_fn, task=task_config_from_args(args)),
        dynamic_filter_fn=has_reward_variance,
        evaluation_fn=(
            make_evaluation(
                eval_rows,
                samples_per_row=args.eval_group_size,
                max_concurrency=config.max_concurrency_rollout_sample,
            )
            if eval_rows
            else None
        ),
        evaluation_interval=args.evaluation_interval,
        rows=train_rows,
    )
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
