#!/usr/bin/env python3
"""GRPO on DeepMath with Fireworks serverless training.

Same task code as ``train_deepmath.py`` (``make_rollout_fn`` graded by
:func:`deepmath_reward`), run by
:func:`training.recipes.experiment.async_rl_loop_serverless.main` instead of
the dedicated recipe. There is no trainer job, deployment, training shape or
region: the recipe attaches to a pooled LoRA trainer and publishes a sampler
snapshot after every optimizer step. MoE base models replay the sampled expert
routes (R3) automatically.

    export FIREWORKS_API_KEY=fw_...
    python -m training.examples.rl.deepmath.train_deepmath_serverless \\
        --base-model accounts/fireworks/models/kimi-k3 --tokenizer-model moonshotai/Kimi-K3

``--run-dir`` receives ``metrics.jsonl`` and ``result.json`` (steps, rows
consumed and checkpoint paths). Continue in a new process with
``--init-from-checkpoint``, ``--step-offset`` and ``--resolved-rows-offset``
from that file.
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training.examples.rl.deepmath.train_deepmath import DEFAULT_DATASET, make_rollout_fn
from training.recipes.experiment import async_rl_loop_serverless
from training.recipes.experiment.async_rl_loop_serverless import Config
from training.utils import WandBConfig, load_jsonl_dataset
from training.utils.rl.heldout import HeldoutRuns, make_heldout_evaluation
from training.utils.rl.losses import has_reward_variance


def deepmath_eval_metrics(_rows: Sequence[Mapping[str, Any]], runs: HeldoutRuns) -> dict[str, float]:
    """Held-out accuracy; dropped samples count as wrong."""
    expected = sum(len(row_runs) for row_runs in runs)
    rewards = [run.segments[0].reward for row_runs in runs for run in row_runs if run is not None]
    return {
        "eval/accuracy": sum(rewards) / expected if expected else 0.0,
        "eval/samples": len(rewards),
        "eval/return_ratio": len(rewards) / expected if expected else 0.0,
    }


def train(
    config: Config,
    *,
    renderer_name: str = "",
    eval_rows: int = 16,
    evaluation_interval: int = 5,
) -> dict[str, Any]:
    """Hold out the first ``eval_rows`` rows, train on the next ``config.max_rows``."""
    if not config.dataset:
        raise ValueError("config.dataset is required")
    rows = load_jsonl_dataset(config.dataset)
    held_out, train_rows = rows[:eval_rows], rows[eval_rows : eval_rows + config.max_rows]
    if not train_rows:
        raise ValueError(f"{config.dataset} has no rows left for training after {eval_rows} eval rows")
    evaluation_fn = None
    if held_out:
        evaluation_fn = make_heldout_evaluation(
            held_out,
            samples_per_row=config.completions_per_prompt,
            metrics_fn=deepmath_eval_metrics,
            max_concurrency=config.max_concurrency_rollout_sample,
        )
    return async_rl_loop_serverless.main(
        config,
        rollout_fn_factory=functools.partial(make_rollout_fn, renderer_name=renderer_name),
        dynamic_filter_fn=has_reward_variance,
        evaluation_fn=evaluation_fn,
        evaluation_interval=evaluation_interval,
        rows=train_rows,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serverless GRPO on DeepMath",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-model", default="accounts/fireworks/models/kimi-k3")
    parser.add_argument("--tokenizer-model", default="moonshotai/Kimi-K3")
    parser.add_argument("--renderer-name", default="", help="Empty resolves it from the tokenizer.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--eval-rows", type=int, default=16, help="Leading rows held out for eval (0 disables).")
    parser.add_argument("--max-rows", type=int, default=184, help="Training prompts per epoch.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--prompt-groups-per-step", type=int, default=16)
    parser.add_argument("--completions-per-prompt", type=int, default=8)
    parser.add_argument("--max-concurrency-rollout-sample", type=int, default=64)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--sample-timeout", type=float, default=1800.0)
    parser.add_argument("--snapshot-prefix", default="deepmath")
    parser.add_argument("--dcp-save-interval", type=int, default=0)
    parser.add_argument("--init-from-checkpoint", default=None, help="Training-state path from result.json.")
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--resolved-rows-offset", type=int, default=0)
    parser.add_argument("--run-dir", default=f"./deepmath_serverless_{int(time.time())}")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default="serverless-rl-deepmath")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        base_model=args.base_model,
        tokenizer_model=args.tokenizer_model,
        dataset=args.dataset,
        learning_rate=args.learning_rate,
        completions_per_prompt=args.completions_per_prompt,
        prompt_groups_per_step=args.prompt_groups_per_step,
        max_completion_tokens=args.max_completion_tokens,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        epochs=args.epochs,
        # Keep file order so the row cursor stays stable on resume.
        shuffle=False,
        max_rows=args.max_rows,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        weight_decay=0.0,
        max_concurrency_rollout_sample=args.max_concurrency_rollout_sample,
        min_group_size=2,
        sample_timeout=args.sample_timeout,
        snapshot_prefix=args.snapshot_prefix,
        metrics_file=str(Path(args.run_dir) / "metrics.jsonl"),
        init_from_checkpoint=args.init_from_checkpoint,
        step_offset=args.step_offset,
        resolved_rows_offset=args.resolved_rows_offset,
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
    result = train(
        config_from_args(args),
        renderer_name=args.renderer_name,
        eval_rows=args.eval_rows,
        evaluation_interval=args.evaluation_interval,
    )
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
