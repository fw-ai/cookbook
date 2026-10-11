"""VisualToolBench task settings, data loading and held-out evaluation.

Both entry points assemble the same pieces with a different recipe:

* ``train_visual_toolbench.py`` -> :mod:`training.recipes.async_rl_loop`
  (dedicated trainer job + hot-load deployment);
* ``examples/rl/visual_toolbench/train_visual_toolbench_serverless.py`` ->
  :mod:`training.recipes.experiment.async_rl_loop_serverless` (pooled
  serverless training).

Each passes ``functools.partial(rollout.make_rollout_fn, task=TaskConfig(...))``
as the rollout factory, ``training.utils.rl.losses.has_reward_variance`` as the
group filter and ``make_evaluation(...)`` as ``evaluation_fn``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

from training.examples.rl.visual_toolbench.reward import (
    DEFAULT_CRITICAL_REWARD_WEIGHT,
    DEFAULT_JUDGE_MAX_CONCURRENCY,
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_TIMEOUT_S,
)
from training.utils import load_jsonl_dataset
from training.utils.rl.heldout import (
    HeldoutEvaluationFn,
    HeldoutRuns,
    make_heldout_evaluation,
)

DEFAULT_RENDERER_NAME = "qwen3_6_disable_thinking_interleaved"
DEFAULT_SYSTEM_PROMPT = (
    "You are a visual analysis assistant. The user's images may hide the "
    "detail you need at full resolution: crop into the relevant region, zoom, "
    "rotate, or adjust brightness/contrast/sharpness with the provided tools, "
    "inspect the transformed image, and only then answer. When you are "
    "confident, give the final answer directly without calling a tool."
)


@dataclass
class TaskConfig:
    """Rollout, judge and evaluation-sampling settings; ``make_rollout_fn`` closes over it."""

    renderer_name: str = DEFAULT_RENDERER_NAME
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_turns: int = 6
    max_workspace_images: int = 6
    max_prompt_tokens: int = 57344
    tool_image_dim: int = 1024
    filter_truncated_rollouts: bool = True
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS
    judge_max_concurrency: int = DEFAULT_JUDGE_MAX_CONCURRENCY
    judge_timeout_s: float = DEFAULT_JUDGE_TIMEOUT_S
    critical_reward_weight: float = DEFAULT_CRITICAL_REWARD_WEIGHT
    # Held-out evaluation uses the benchmark's sampling. Training rollouts keep
    # the recipe's temperature / top_p=1 / top_k=0 so sampler and trainer
    # score the same distribution.
    eval_temperature: float = 1.0
    eval_top_p: float = 0.95
    eval_top_k: int = 20
    eval_max_completion_tokens: int = 26666

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {self.max_turns}")

    def eval_sample_kwargs(self) -> dict[str, Any]:
        """Sampling overrides for rollouts called with ``evaluation=True``."""
        return {
            "temperature": self.eval_temperature,
            "top_p": self.eval_top_p,
            "top_k": self.eval_top_k,
            "max_tokens": self.eval_max_completion_tokens,
        }


def add_task_args(parser: argparse.ArgumentParser) -> None:
    """Data, evaluation and ``TaskConfig`` flags shared by both entry points."""
    parser.add_argument("--dataset", required=True, help="Aligned training JSONL.")
    parser.add_argument("--eval-dataset", default=None, help="Held-out JSONL; omit to skip evaluation.")
    parser.add_argument("--max-rows", type=int, default=None, help="Use only the first N training rows.")
    parser.add_argument("--eval-group-size", type=int, default=1, help="Samples per held-out row.")
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument(
        "--require-tool-aligned-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject rows without tool_alignment.eligible=true.",
    )
    for spec in fields(TaskConfig):
        flag = "--" + spec.name.replace("_", "-")
        if isinstance(spec.default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=spec.default)
        else:
            parser.add_argument(flag, type=type(spec.default), default=spec.default)


def task_config_from_args(args: argparse.Namespace) -> TaskConfig:
    return TaskConfig(**{spec.name: getattr(args, spec.name) for spec in fields(TaskConfig)})


def load_rows(path: str, *, require_tool_aligned: bool = True) -> list[dict[str, Any]]:
    """Load task rows; by default every row must be marked tool-aligned."""
    rows = load_jsonl_dataset(path)
    for row in rows:
        alignment = row.get("tool_alignment")
        if require_tool_aligned and not (
            isinstance(alignment, dict) and alignment.get("eligible") is True
        ):
            raise ValueError(
                f"{path}: row {row.get('id', '<unknown>')} is not tool-aligned; set "
                "tool_alignment.eligible=true only after verifying the row uses "
                "the supported image tools"
            )
    return rows


def load_splits(
    dataset: str,
    eval_dataset: str | None,
    *,
    max_rows: int | None = None,
    require_tool_aligned: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Training rows and disjoint held-out rows (empty without ``eval_dataset``)."""
    train_rows = load_rows(dataset, require_tool_aligned=require_tool_aligned)[:max_rows]
    eval_rows = (
        load_rows(eval_dataset, require_tool_aligned=require_tool_aligned)
        if eval_dataset
        else []
    )
    if not train_rows:
        raise ValueError(f"training dataset is empty: {dataset}")
    if eval_dataset and not eval_rows:
        raise ValueError(f"eval dataset is empty: {eval_dataset}")
    overlap = {str(row.get("id", "")) for row in train_rows} & {
        str(row.get("id", "")) for row in eval_rows
    }
    if overlap:
        raise ValueError(
            f"train/eval datasets overlap on {len(overlap)} row ids; "
            f"first overlap: {sorted(overlap)[0]}"
        )
    return train_rows, eval_rows


def evaluation_metrics(
    rows: Sequence[Mapping[str, Any]], runs: HeldoutRuns
) -> dict[str, float]:
    """Held-out metrics. A dropped episode scores zero, so every evaluation
    divides by the same number of expected episodes."""
    expected = sum(len(row_runs) for row_runs in runs)
    returned = [run for row_runs in runs for run in row_runs if run is not None]

    def official(run) -> float:
        return float((run.metadata or {}).get("mean_official_score", 0.0))

    metrics: dict[str, float] = {
        "eval/reward": sum(run.segments[0].reward for run in returned) / expected,
        "eval/official_score": sum(official(run) for run in returned) / expected,
        "eval/critical_fraction": sum(
            float((run.metadata or {}).get("mean_critical_fraction", 0.0))
            for run in returned
        )
        / expected,
        "eval/judge_pass": sum(
            bool((run.metadata or {}).get("judge_passed")) for run in returned
        )
        / expected,
        "eval/mean_tool_calls": (
            sum(int((run.metadata or {}).get("num_tool_calls", 0)) for run in returned)
            / len(returned)
            if returned
            else 0.0
        ),
        "eval/rows": len(rows),
        "eval/returned_episodes": len(returned),
        "eval/return_ratio": len(returned) / expected,
    }
    for field, prefix in (("category", "category"), ("eval_focus", "focus")):
        totals: dict[str, tuple[float, int]] = {}
        for row, row_runs in zip(rows, runs):
            name = str(row.get(field) or "missing")
            score, count = totals.get(name, (0.0, 0))
            totals[name] = (
                score + sum(official(run) for run in row_runs if run is not None),
                count + len(row_runs),
            )
        for name, (score, count) in totals.items():
            metrics[f"eval/{prefix}/{name}/official_score"] = score / count
    return metrics


def make_evaluation(
    rows: Sequence[Mapping[str, Any]],
    *,
    samples_per_row: int,
    max_concurrency: int | None,
) -> HeldoutEvaluationFn:
    """Recipe ``evaluation_fn`` over the held-out rows."""
    return make_heldout_evaluation(
        rows,
        samples_per_row=samples_per_row,
        metrics_fn=evaluation_metrics,
        max_concurrency=max_concurrency,
    )
