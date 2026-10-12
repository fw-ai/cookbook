#!/usr/bin/env python3
"""Serverless GRPO on Countdown: task code plus the serverless async-RL recipe.

Deprecated: archived example, covered by ``rl/deepmath/train_deepmath_serverless.py``. See ``examples/archive/README.md``.

Countdown (TinyZero): combine the given numbers with ``+ - * /`` to hit a
target. This file only supplies the task:

* ``make_rollout_fn`` renders the chat prompt, samples one completion and
  scores it with :func:`countdown_rewards.composite_reward`;
* ``countdown_eval_metrics`` scores a fixed held-out slice of the dataset;
* ``train`` hands both to
  :func:`training.recipes.experiment.async_rl_loop_serverless.main`.

The recipe owns everything else: the serverless training session (no trainer
job, no deployment), per-step sampler snapshots, async rollout scheduling,
server-side GRPO, metrics/W&B, and checkpoints. MoE base models replay the
sampled expert routes (R3) automatically. ``make_rollout_fn`` runs unchanged on
a dedicated trainer + deployment through :func:`training.recipes.async_rl_loop.main`.

    python -m training.examples.archive.countdown.prepare_data      # once
    export FIREWORKS_API_KEY=fw_...
    python -m training.examples.archive.countdown.countdown_rl \\
        --base-model accounts/fireworks/models/kimi-k3 --tokenizer-model moonshotai/Kimi-K3

``--run-dir`` receives ``metrics.jsonl``, ``result.json`` (steps, rows consumed
and checkpoint paths for ``--init-from-checkpoint`` resume) and
``reward_curve.png``.
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training.examples.archive.countdown.countdown_rewards import composite_reward
from training.examples.archive.countdown.prepare_data import DEFAULT_DATASET
from training.recipes.async_rl_loop import RolloutFn, RolloutSetup
from training.recipes.experiment import async_rl_loop_serverless
from training.recipes.experiment.async_rl_loop_serverless import Config
from training.renderer import get_text_content
from training.utils import WandBConfig, load_jsonl_dataset
from training.utils.rl.heldout import HeldoutRuns, make_heldout_evaluation
from training.utils.rl.losses import has_reward_variance
from training.utils.rl.rollout import single_turn_renderer_rollout
from training.utils.supervised import build_renderer


def make_rollout_fn(setup: RolloutSetup, *, renderer_name: str = "") -> RolloutFn:
    """One Countdown completion per call, sampled from the recipe's sampler."""
    renderer = build_renderer(setup.tokenizer, setup.tokenizer_id, renderer_name)

    def reward(row: dict, message: Any, _parsed: bool) -> float:
        return composite_reward(get_text_content(message), row["ground_truth"])

    async def rollout_fn(row: dict):
        return await single_turn_renderer_rollout(
            row,
            renderer=renderer,
            sample_with_prompt_tokens=setup.sampler.sample_with_prompt_tokens,
            message_builder=lambda r: r["messages"],
            reward_fn=reward,
            sample_kwargs=setup.sample_kwargs,
            tokenizer=setup.tokenizer,
        )

    return rollout_fn


def countdown_eval_metrics(_rows: Sequence[Mapping[str, Any]], runs: HeldoutRuns) -> dict[str, float]:
    """Held-out reward; ``eval/accuracy`` counts dropped samples as wrong."""
    expected = sum(len(row_runs) for row_runs in runs)
    rewards = [run.segments[0].reward for row_runs in runs for run in row_runs if run is not None]
    return {
        "eval/raw_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "eval/accuracy": sum(reward >= 1.0 for reward in rewards) / expected,
        "eval/samples": len(rewards),
        "eval/return_ratio": len(rewards) / expected,
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
            metrics_fn=countdown_eval_metrics,
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


def plot_reward_curve(metrics_file: Path, output: Path) -> None:
    """Plot training and held-out reward from the recipe's metrics JSONL."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = [json.loads(line) for line in metrics_file.open() if line.strip()]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in (("rollout/raw_reward", "train"), ("eval/raw_reward", "held-out")):
        points = [(record["step"], record[key]) for record in records if key in record]
        if points:
            ax.plot(*zip(*points), marker="o", label=label)
    ax.set(xlabel="optimizer step", ylabel="reward", ylim=(0.0, 1.0))
    ax.legend()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serverless GRPO on the Countdown task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-model", default="accounts/fireworks/models/kimi-k3")
    parser.add_argument("--tokenizer-model", default="moonshotai/Kimi-K3")
    parser.add_argument("--renderer-name", default="", help="Empty resolves it from the tokenizer.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--eval-rows", type=int, default=16, help="Leading rows held out for eval (0 disables).")
    parser.add_argument("--max-rows", type=int, default=320, help="Training prompts per epoch.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--prompt-groups-per-step", type=int, default=16)
    parser.add_argument("--completions-per-prompt", type=int, default=8)
    parser.add_argument("--max-concurrency-rollout-sample", type=int, default=64)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--sample-timeout", type=float, default=1800.0)
    parser.add_argument("--snapshot-prefix", default="countdown")
    parser.add_argument("--dcp-save-interval", type=int, default=0)
    parser.add_argument("--init-from-checkpoint", default=None, help="Training-state path from result.json.")
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--resolved-rows-offset", type=int, default=0)
    parser.add_argument("--run-dir", default=f"./countdown_serverless_{int(time.time())}")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default="serverless-rl-countdown")
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
        # prepare_data already shuffled the file; keeping its order makes the
        # row cursor stable when --max-rows grows on resume.
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
    try:
        plot_reward_curve(run_dir / "metrics.jsonl", run_dir / "reward_curve.png")
    except ImportError:
        print("matplotlib not installed; skipping reward_curve.png")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
