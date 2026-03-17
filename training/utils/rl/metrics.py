"""Reusable metric helpers for RL cookbook loops."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import tinker

from training.utils.rl.losses import PromptGroup

logger = logging.getLogger(__name__)

_SKIP_REMOTE_KEYS = {"step_id", "step"}


def median(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def datum_target_len(datum: tinker.Datum) -> int:
    """Best-effort extraction of target-token length from a training datum."""
    try:
        target = datum.loss_fn_inputs.get("target_tokens")
        shape = getattr(target, "shape", None)
        if isinstance(shape, (list, tuple)) and shape:
            return int(shape[0])
        data = getattr(target, "data", None)
        if data is not None:
            return len(data)
    except Exception:
        pass
    return 0


def total_target_tokens(prompt_groups: Sequence[PromptGroup]) -> int:
    return sum(
        datum_target_len(datum)
        for pg in prompt_groups
        for datum in pg.data
    )


def add_response_length_stats(metrics: dict[str, Any], completion_lens: Sequence[int]) -> None:
    if not completion_lens:
        return
    total = sum(completion_lens)
    metrics["rollout/response_len/mean"] = total / len(completion_lens)
    metrics["rollout/response_len/median"] = median(completion_lens)
    metrics["rollout/response_len/max"] = max(completion_lens)
    metrics["rollout/response_len/min"] = min(completion_lens)


def add_train_perf_metrics(metrics: dict[str, Any], *, total_model_tokens: int) -> None:
    if total_model_tokens <= 0:
        return

    ref_time = metrics.get("perf/ref_forward_time", 0.0)
    if ref_time > 0:
        metrics["perf/ref_tokens_per_s"] = total_model_tokens / ref_time

    train_time = metrics.get("perf/fwd_bwd_time", 0.0)
    if train_time > 0:
        metrics["perf/train_tokens_per_s"] = total_model_tokens / train_time

    step_time = metrics.get("perf/step_time", 0.0)
    weight_sync_time = metrics.get("perf/weight_sync_time", 0.0)
    if step_time > 0 and weight_sync_time > 0:
        metrics["perf/weight_sync_ratio"] = weight_sync_time / step_time


def build_loop_metrics(
    *,
    train_step: int,
    sample_fails: int,
    staleness_steps: Sequence[int] | None = None,
) -> dict[str, Any]:
    loop_metrics: dict[str, Any] = {
        "train/step": train_step,
        "rollout/sample_fail_count": sample_fails,
    }

    if staleness_steps:
        loop_metrics["version/sample_staleness_avg"] = sum(staleness_steps) / len(staleness_steps)
        loop_metrics["version/sample_staleness_max"] = max(staleness_steps)

    return loop_metrics


def compute_step_metrics(
    *,
    prompt_groups: Sequence[PromptGroup],
    fwd_bwd_results: Sequence,
    optim_result: Any,
    n_accum: int,
    timing_metrics: dict[str, Any],
    rollout_stats: Any | None = None,
    completions_per_prompt: int = 1,
) -> dict[str, Any]:
    """Compute all per-step wandb metrics from prompt groups and remote results.

    Consolidates reward/accuracy/entropy/truncation/filter/batch stats
    into a single metrics dict.  Called by ``finish_step`` in the recipe.

    ``rollout_stats`` is a :class:`~training.utils.rl.train.RolloutStats`
    instance from the training loop.
    """
    metrics = dict(timing_metrics)

    total_model_tokens = total_target_tokens(prompt_groups)
    metrics["train/effective_grad_accum_steps"] = n_accum
    add_train_perf_metrics(metrics, total_model_tokens=total_model_tokens)

    if optim_result and hasattr(optim_result, "metrics") and optim_result.metrics:
        for k, v in optim_result.metrics.items():
            if k not in _SKIP_REMOTE_KEYS:
                metrics[f"train/{k}"] = v

    last_fwd_bwd = fwd_bwd_results[-1] if fwd_bwd_results else None
    if last_fwd_bwd is not None:
        for k, v in last_fwd_bwd.metrics.items():
            if k not in _SKIP_REMOTE_KEYS:
                metrics[f"train/{k}"] = v

    all_rewards: list[float] = []
    all_comp_lens: list[int] = []
    all_truncated: list[bool] = []
    correct_count = 0
    total_samples = 0

    for pg in prompt_groups:
        all_rewards.extend(pg.rewards)
        all_comp_lens.extend(pg.completion_lens)
        all_truncated.extend(pg.truncated)
        correct_count += sum(1 for r in pg.rewards if r > 0.5)
        total_samples += len(pg.rewards)

    metrics["rollout/samples_completed"] = total_samples
    if total_samples > 0:
        metrics["rollout/reward"] = sum(all_rewards) / total_samples
        metrics["rollout/accuracy"] = correct_count / total_samples
    add_response_length_stats(metrics, all_comp_lens)
    if all_truncated:
        metrics["rollout/truncated_ratio"] = sum(all_truncated) / len(all_truncated)

    entropy_vals: list[float] = []
    for pg in prompt_groups:
        for inf_lp in pg.inf_logprobs:
            resp_start = max(0, pg.prompt_len - 1)
            resp_lp = inf_lp[resp_start:] if len(inf_lp) > resp_start else []
            if resp_lp:
                entropy_vals.append(-sum(resp_lp) / len(resp_lp))
    if entropy_vals:
        metrics["rollout/entropy"] = sum(entropy_vals) / len(entropy_vals)

    rs = rollout_stats
    if rs is not None:
        filter_ratio = rs.filtered / max(1, rs.accepted + rs.filtered)
        metrics["rollout/valid_prompt_groups"] = rs.accepted
        metrics["rollout/total_sampled"] = rs.sampled
        metrics["rollout/filter_reject_ratio"] = filter_ratio
        metrics["rollout/sample_fail_count"] = rs.failed
        metrics["rollout/fwd_bwd_count"] = n_accum

        metrics["perf/sample_wait_time"] = rs.wait_time
        if rs.wall_time > 0:
            wait_ratio = rs.wait_time / rs.wall_time
            metrics["perf/wait_time_ratio"] = wait_ratio
            metrics["perf/overlap_ratio"] = 1.0 - wait_ratio

        if rs.wall_time > 0:
            metrics["perf/step_wall_time"] = rs.wall_time
            metrics["perf/rollout_samples_per_s"] = total_samples / rs.wall_time
            metrics["perf/rollout_tokens_per_s"] = sum(all_comp_lens) / rs.wall_time

        if rs.rewards:
            metrics["rollout/raw_reward"] = sum(rs.rewards) / len(rs.rewards)

    return metrics
