"""Reusable metric helpers for RL cookbook loops."""

from __future__ import annotations

from typing import Any, Sequence

import tinker

from training.utils.rl.losses import PromptGroup

_SKIP_REMOTE_KEYS = {"step_id", "step", "response_tokens", "total_tokens"}
_SUM_REMOTE_KEYS = {"active_tokens", "total_resp_tokens"}
_LOOP_STAT_PASSTHROUGH_KEYS = (
    "async/version_offset_mean",
    "async/version_offset_max",
    "async/version_offset_min",
    "async/in_flight_samples_mean",
    "async/admission_capacity_samples_mean",
    "async/staleness_capacity_samples_mean",
    "async/concurrency_capacity_samples_mean",
    "async/realized_training_chunks",
    "async/trained_against_version",
    "perf/trainer_wait_for_chunk_time",
)

_CANONICAL_OPTIMIZER_METRICS = ("grad_norm", "grad_norm_rms", "lr")


def datum_target_len(datum: tinker.Datum) -> int:
    """Length of the target-token tensor on a training datum (0 if missing)."""
    target = datum.loss_fn_inputs.get("target_tokens")
    shape = getattr(target, "shape", None)
    if isinstance(shape, (list, tuple)) and shape:
        return int(shape[0])
    data = getattr(target, "data", None)
    if data is not None:
        return len(data)
    return 0


def total_target_tokens(prompt_groups: Sequence[PromptGroup]) -> int:
    return sum(
        datum_target_len(datum)
        for pg in prompt_groups
        for datum in pg.data
    )
def add_optimizer_metrics(metrics: dict[str, Any], optim_result: Any) -> None:
    """Keep one customer-facing value for each useful optimizer diagnostic."""

    raw = getattr(optim_result, "metrics", None) if optim_result else None
    if not raw:
        return
    for key in _CANONICAL_OPTIMIZER_METRICS:
        value = raw.get(key, raw.get(f"{key}:last"))
        if value is not None:
            metrics[f"train/{key}"] = value

    # Post-clip norm is useful only when clipping changed the gradient. The
    # trainer reports it even when clipping is disabled, where it duplicates
    # the canonical pre-clip norm exactly.
    post_clip = raw.get("grad_norm_post_clip", raw.get("grad_norm_post_clip:last"))
    grad_norm = metrics.get("train/grad_norm")
    if post_clip is not None and post_clip != grad_norm:
        metrics["train/grad_norm_post_clip"] = post_clip


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


def build_accumulated_async_loop_stats(
    *,
    latest_loop_stats: dict[str, Any] | None,
    trainer_wait_for_sampler_time: float,
    sampler_wait_for_trainer_time: float,
    train_wall_time: float,
) -> dict[str, Any] | None:
    """Merge per-chunk scheduler stats into one optimizer-step stats payload."""
    if latest_loop_stats is None:
        return None

    loop_stats = dict(latest_loop_stats)
    loop_stats["trainer_wait_for_sampler_time"] = trainer_wait_for_sampler_time
    loop_stats["sampler_wait_for_trainer_time"] = sampler_wait_for_trainer_time
    loop_stats["train_wall_time"] = train_wall_time
    loop_stats["scheduler_step_wall_time"] = trainer_wait_for_sampler_time + train_wall_time
    return loop_stats


def compute_step_metrics(
    *,
    prompt_groups: Sequence[PromptGroup],
    fwd_bwd_results: Sequence,
    optim_result: Any,
    n_accum: int,
    timing_metrics: dict[str, Any],
    loop_stats: dict | None = None,
) -> dict[str, Any]:
    """Compute all per-step wandb metrics from prompt groups and remote results.

    Consolidates rollout quality, optimizer telemetry, and scheduler timing
    into one record per optimizer step.
    """
    metrics = dict(timing_metrics)

    total_model_tokens = total_target_tokens(prompt_groups)
    metrics["train/target_tokens"] = total_model_tokens
    if not loop_stats or "async/realized_training_chunks" not in loop_stats:
        metrics["train/effective_accumulation_steps"] = n_accum
    add_train_perf_metrics(metrics, total_model_tokens=total_model_tokens)

    add_optimizer_metrics(metrics, optim_result)

    # Reduce every physical forward/backward result to one optimizer-step
    # record. Counts add; scalar diagnostics average only over chunks that
    # actually reported them, so optional metrics are never diluted by zero.
    if fwd_bwd_results:
        accum: dict[str, float] = {}
        counts: dict[str, int] = {}
        reducers: dict[str, str] = {}
        for result in fwd_bwd_results:
            for k, v in result.metrics.items():
                base, separator, reducer = k.partition(":")
                if base in _SKIP_REMOTE_KEYS or reducer == "last":
                    continue
                if separator and reducer not in {"sum", "min", "max"}:
                    continue
                reducers[k] = reducer
                if reducer == "min":
                    accum[k] = min(accum.get(k, v), v)
                elif reducer == "max":
                    accum[k] = max(accum.get(k, v), v)
                else:
                    accum[k] = accum.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1
        for k, v in accum.items():
            base = k.partition(":")[0]
            metrics[f"train/{k}"] = (
                v
                if base in _SUM_REMOTE_KEYS or reducers[k] in {"sum", "min", "max"}
                else v / counts[k]
            )
        active_tokens = metrics.get("train/active_tokens")
        response_tokens = metrics.get("train/total_resp_tokens")
        if active_tokens is not None and response_tokens:
            metrics["train/mask_ratio"] = active_tokens / response_tokens

    all_rewards: list[float] = []

    for pg in prompt_groups:
        all_rewards.extend(pg.rewards)
    filtered_samples = len(all_rewards)
    completion_tokens = sum(
        completion_len
        for pg in prompt_groups
        for completion_len in pg.completion_lens
    )

    if loop_stats:
        raw_rewards = loop_stats["all_raw_rewards"]
        raw_samples = len(raw_rewards)
        if raw_samples:
            metrics["rollout/raw_reward"] = sum(raw_rewards) / raw_samples
            metrics["rollout/filtered_reward"] = (
                sum(all_rewards) / filtered_samples if filtered_samples else 0.0
            )
            metrics["rollout/raw_samples"] = raw_samples
            metrics["rollout/filtered_samples"] = filtered_samples
            metrics["rollout/filter_ratio"] = 1.0 - filtered_samples / raw_samples
        for key in _LOOP_STAT_PASSTHROUGH_KEYS:
            if key in loop_stats:
                metrics[key] = loop_stats[key]

        rollout_wall_time = float(loop_stats.get("rollout_batch_wall_time", 0.0))
        if rollout_wall_time > 0:
            metrics["perf/rollout_batch_wall_time"] = rollout_wall_time

        train_wall_time = float(loop_stats.get("train_wall_time", 0.0))
        if train_wall_time > 0:
            metrics["perf/train_step_wall_time"] = train_wall_time

        trainer_wait_for_sampler_time = float(loop_stats.get("trainer_wait_for_sampler_time", 0.0))
        if trainer_wait_for_sampler_time > 0:
            metrics["perf/trainer_wait_for_sampler_time"] = trainer_wait_for_sampler_time
        sampler_wait_for_trainer_time = float(
            loop_stats.get("sampler_wait_for_trainer_time", 0.0)
        )
        if sampler_wait_for_trainer_time > 0:
            metrics["perf/sampler_wait_for_trainer_time"] = sampler_wait_for_trainer_time

        scheduler_step_wall_time = float(loop_stats.get("scheduler_step_wall_time", 0.0))
        if scheduler_step_wall_time <= 0:
            scheduler_step_wall_time = (
                rollout_wall_time + trainer_wait_for_sampler_time + train_wall_time
            )
        if scheduler_step_wall_time > 0:
            metrics["perf/scheduler_step_wall_time"] = scheduler_step_wall_time
            metrics["perf/step_samples_per_s"] = filtered_samples / scheduler_step_wall_time
            metrics["perf/step_tokens_per_s"] = completion_tokens / scheduler_step_wall_time
            if rollout_wall_time > 0:
                metrics["perf/rollout_batch_wall_ratio"] = rollout_wall_time / scheduler_step_wall_time
            trainer_wait_for_chunk_time = float(
                loop_stats.get("perf/trainer_wait_for_chunk_time", 0.0)
            )
            trainer_idle_time = (
                trainer_wait_for_sampler_time + trainer_wait_for_chunk_time
            )
            metrics["perf/trainer_idle_ratio"] = min(
                1.0,
                trainer_idle_time / scheduler_step_wall_time,
            )

        if rollout_wall_time > 0:
            metrics["perf/rollout_batch_samples_per_s"] = filtered_samples / rollout_wall_time
            metrics["perf/rollout_batch_tokens_per_s"] = completion_tokens / rollout_wall_time

    return metrics
