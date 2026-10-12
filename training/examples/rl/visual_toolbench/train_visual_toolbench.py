#!/usr/bin/env python3
"""VisualToolBench GRPO on a dedicated trainer + inference deployment.

Assembles this package's task code (``rollout.make_rollout_fn``,
``task.TaskConfig``, ``task.make_evaluation``) with
:func:`training.recipes.async_rl_loop.main`. The recipe creates or reattaches
the trainer job and hot-load deployment, schedules rollouts asynchronously,
trains with server-side GRPO, syncs weights, logs metrics and saves
checkpoints. MoE policies replay sampled expert routes (R3) automatically.
The serverless twin is ``training.examples.rl.visual_toolbench.train_visual_toolbench_serverless``.

    export FIREWORKS_API_KEY=fw_...
    python -m training.examples.rl.visual_toolbench.train_visual_toolbench \\
        --dataset /abs/train.jsonl --eval-dataset /abs/eval.jsonl \\
        --training-shape-id <training-shape> --log-path ./vtb_dedicated

Reuse an existing trainer and deployment with ``--job-id`` and
``--deployment-id`` (add ``--no-cleanup-on-exit`` to leave them running).
"""

from __future__ import annotations

import argparse
import functools
import logging
from typing import Any

from training.examples.rl.visual_toolbench.rollout import make_rollout_fn
from training.examples.rl.visual_toolbench.task import (
    add_task_args,
    load_splits,
    make_evaluation,
    task_config_from_args,
)
from training.recipes import async_rl_loop
from training.recipes.async_rl_loop import Config
from training.utils.rl.losses import has_reward_variance
from training.utils import DeployConfig, TrainerConfig, WandBConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisualToolBench GRPO on a dedicated trainer + deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_task_args(parser)
    parser.add_argument("--base-model", default="accounts/fireworks/models/qwen3p6-27b")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--log-path", default="./vtb_dedicated", help="Checkpoint bookkeeping directory.")
    parser.add_argument("--training-shape-id", default=None)
    parser.add_argument("--job-id", default=None, help="Existing trainer job to reattach to.")
    parser.add_argument("--deployment-id", default=None, help="Existing hot-load deployment to reuse.")
    parser.add_argument("--deployment-shape", default=None)
    parser.add_argument("--cleanup-on-exit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-groups-per-step", type=int, default=8)
    parser.add_argument("--completions-per-prompt", type=int, default=8)
    parser.add_argument("--max-concurrency-rollout-sample", type=int, default=64)
    parser.add_argument("--max-completion-tokens", type=int, default=32768)
    parser.add_argument("--max-seq-len", type=int, default=None, help="Default: the trainer's context length.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=0, help="0 trains all parameters.")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--sample-timeout", type=int, default=1800)
    parser.add_argument("--dcp-save-interval", type=int, default=0)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default="rl-vtb")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        log_path=args.log_path,
        base_model=args.base_model,
        learning_rate=args.learning_rate,
        completions_per_prompt=args.completions_per_prompt,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        epochs=args.epochs,
        shuffle=True,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        lora_rank=args.lora_rank,
        prompt_groups_per_step=args.prompt_groups_per_step,
        max_concurrency_rollout_sample=args.max_concurrency_rollout_sample,
        min_group_size=2,
        trainer=TrainerConfig(job_id=args.job_id, training_shape_id=args.training_shape_id),
        deployment=DeployConfig(
            deployment_id=args.deployment_id,
            deployment_shape=args.deployment_shape,
            tokenizer_model=args.tokenizer_model,
            sample_timeout=args.sample_timeout,
        ),
        dcp_save_interval=args.dcp_save_interval,
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.wandb_run_name,
        ),
        cleanup_on_exit=args.cleanup_on_exit,
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    config = config_from_args(args)
    train_rows, eval_rows = load_splits(
        args.dataset,
        args.eval_dataset,
        max_rows=args.max_rows,
        require_tool_aligned=args.require_tool_aligned_data,
    )
    return async_rl_loop.main(
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(main())
