"""Reusable metric helpers for RL cookbook loops."""

from __future__ import annotations

import math
from collections.abc import Mapping
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
    "perf/step_time",
    "perf/train_time",
    "perf/train_wait_time",
    "perf/wait_time_ratio",
    "perf/train_chunk_wait_time",
)

_CANONICAL_OPTIMIZER_METRICS = ("grad_norm", "grad_norm_rms", "lr")
_TITO_METRIC_ROOT = "tito/"
_TITO_DEBUG_METRIC_ROOT = "tito/debug/"
_DISTRIBUTION_SUFFIXES = ("_count", "_sum", "_mean", "_min", "_max")

_TITO_PUBLIC_DISTRIBUTIONS = {
    "tito/turn/request_wall_seconds": "tito/turn/runtime_seconds",
    "tito/turn/prompt_tokens": "tito/turn/input_tokens",
    "tito/turn/completion_tokens": "tito/turn/output_tokens",
}


def _finite_metric(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"TITO sidecar metric {name!r} must be finite numeric data")
    return float(value)


def merge_tito_sidecar_metrics(
    metrics: dict[str, Any],
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    """Merge trajectory summaries without averaging per-trajectory means."""
    if not summaries:
        return
    for value in summaries:
        invalid = sorted(
            str(name)
            for name in value
            if not isinstance(name, str) or not name.startswith(_TITO_METRIC_ROOT)
        )
        if invalid:
            raise ValueError(
                "TITO sidecar metrics must use only the canonical root: "
                + ", ".join(invalid)
            )

    distribution_bases = {
        name[: -len(suffix)]
        for summary in summaries
        for name in summary
        for suffix in _DISTRIBUTION_SUFFIXES
        if name.endswith(suffix)
    }
    distribution_keys = {
        f"{base}{suffix}"
        for base in distribution_bases
        for suffix in _DISTRIBUTION_SUFFIXES
    }

    counter_totals: dict[str, float] = {}
    for summary in summaries:
        for name, value in summary.items():
            if name in distribution_keys:
                continue
            counter_totals[name] = counter_totals.get(name, 0.0) + _finite_metric(
                value,
                name=name,
            )
    metrics.update(counter_totals)

    for base in sorted(distribution_bases):
        base_keys = {f"{base}{suffix}" for suffix in _DISTRIBUTION_SUFFIXES}
        count = 0.0
        total = 0.0
        minimum: float | None = None
        maximum: float | None = None
        for summary in summaries:
            count_name = f"{base}_count"
            sum_name = f"{base}_sum"
            if base_keys.isdisjoint(summary):
                continue
            if count_name not in summary or sum_name not in summary:
                raise ValueError(f"TITO distribution {base!r} lacks count or sum")
            item_count = _finite_metric(summary[count_name], name=count_name)
            item_sum = _finite_metric(summary[sum_name], name=sum_name)
            if item_count < 0 or not item_count.is_integer():
                raise ValueError(f"TITO distribution {base!r} has invalid count")
            count += item_count
            total += item_sum
            if item_count == 0:
                continue
            min_name = f"{base}_min"
            max_name = f"{base}_max"
            if min_name not in summary or max_name not in summary:
                raise ValueError(f"TITO distribution {base!r} lacks min or max")
            item_min = _finite_metric(summary[min_name], name=min_name)
            item_max = _finite_metric(summary[max_name], name=max_name)
            minimum = item_min if minimum is None else min(minimum, item_min)
            maximum = item_max if maximum is None else max(maximum, item_max)
        metrics[f"{base}_count"] = count
        metrics[f"{base}_sum"] = total
        if count:
            metrics[f"{base}_mean"] = total / count
            metrics[f"{base}_min"] = minimum
            metrics[f"{base}_max"] = maximum


def publish_tito_sidecar_metrics(
    metrics: dict[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    *,
    debug_enabled: bool = False,
) -> None:
    """Publish a compact TITO dashboard, with full internals only in debug mode."""
    merged: dict[str, Any] = {}
    merge_tito_sidecar_metrics(merged, summaries)
    if not merged:
        return

    turn_count = float(merged.get("tito/turn/completion_tokens_count", 0.0))
    metrics["tito/turn/count"] = turn_count
    for source, destination in _TITO_PUBLIC_DISTRIBUTIONS.items():
        count = merged.get(f"{source}_count")
        if count is None:
            continue
        for suffix in ("_mean", "_min", "_max"):
            value = merged.get(f"{source}{suffix}")
            if value is not None:
                metrics[f"{destination}{suffix}"] = value

    trajectory_count = float(merged.get("tito/trajectory/policy_turns_count", 0.0))
    metrics["tito/trajectory/count"] = trajectory_count
    new_segments = float(merged.get("tito/lineage/new_segment", 0.0))
    splits = max(0.0, new_segments - trajectory_count)
    eligible_boundaries = max(0.0, turn_count - trajectory_count)
    metrics["tito/lineage/splits"] = splits
    metrics["tito/lineage/split_ratio"] = (
        splits / eligible_boundaries if eligible_boundaries else 0.0
    )
    metrics["tito/lineage/realigns"] = float(merged.get("tito/lineage/realign", 0.0))
    metrics["tito/parser/model_malformed"] = float(
        merged.get("tito/parser/model_malformed", 0.0)
    )
    metrics["tito/calls/errors"] = float(merged.get("tito/calls/failed", 0.0)) + float(
        merged.get("tito/calls/rejected", 0.0)
    )
    metrics["tito/calls/upstream_retries"] = float(
        merged.get("tito/calls/upstream_retry_attempts", 0.0)
    )

    if debug_enabled:
        for name, value in merged.items():
            detail_name = name.removeprefix(_TITO_METRIC_ROOT)
            if detail_name.startswith("debug/"):
                detail_name = f"artifact/{detail_name.removeprefix('debug/')}"
            metrics[f"{_TITO_DEBUG_METRIC_ROOT}{detail_name}"] = value


def add_tito_sidecar_metrics(
    metrics: dict[str, Any],
    prompt_groups: Sequence[PromptGroup],
) -> None:
    summaries: list[Mapping[str, Any]] = []
    debug_values: set[bool] = set()
    for group in prompt_groups:
        for metadata in group.run_metadata:
            value = metadata.get("tito_metrics")
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise ValueError("run_metadata.tito_metrics must be a mapping")
            summaries.append(value)
            debug_values.add(bool(metadata.get("tito_debug_enabled", False)))
    if len(debug_values) > 1:
        raise ValueError("a batch cannot mix TITO debug modes")
    publish_tito_sidecar_metrics(
        metrics,
        summaries,
        debug_enabled=debug_values == {True},
    )


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
    return sum(datum_target_len(datum) for pg in prompt_groups for datum in pg.data)


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
    add_tito_sidecar_metrics(metrics, prompt_groups)
    filtered_samples = len(all_rewards)
    completion_tokens = sum(
        completion_len for pg in prompt_groups for completion_len in pg.completion_lens
    )
    trainable_tokens = [
        float(metadata["trainable_tokens"])
        for pg in prompt_groups
        for metadata in pg.run_metadata
        if isinstance(metadata.get("trainable_tokens"), (int, float))
        and math.isfinite(float(metadata["trainable_tokens"]))
    ]
    if trainable_tokens:
        metrics.update(
            {
                "rollout/trainable_tokens_mean": sum(trainable_tokens)
                / len(trainable_tokens),
                "rollout/trainable_tokens_max": max(trainable_tokens),
                "rollout/trainable_tokens_min": min(trainable_tokens),
            }
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

        if "async/realized_training_chunks" in loop_stats:
            step_time = float(loop_stats.get("perf/step_time", 0.0))
            if step_time > 0:
                metrics["perf/step_samples_per_s"] = filtered_samples / step_time
                metrics["perf/step_tokens_per_s"] = completion_tokens / step_time
        else:
            rollout_wall_time = float(loop_stats.get("rollout_batch_wall_time", 0.0))
            if rollout_wall_time > 0:
                metrics["perf/rollout_batch_wall_time"] = rollout_wall_time

            train_wall_time = float(loop_stats.get("train_wall_time", 0.0))
            if train_wall_time > 0:
                metrics["perf/train_step_wall_time"] = train_wall_time

            trainer_wait_for_sampler_time = float(
                loop_stats.get("trainer_wait_for_sampler_time", 0.0)
            )
            if trainer_wait_for_sampler_time > 0:
                metrics["perf/trainer_wait_for_sampler_time"] = (
                    trainer_wait_for_sampler_time
                )
            sampler_wait_for_trainer_time = float(
                loop_stats.get("sampler_wait_for_trainer_time", 0.0)
            )
            if sampler_wait_for_trainer_time > 0:
                metrics["perf/sampler_wait_for_trainer_time"] = (
                    sampler_wait_for_trainer_time
                )

            scheduler_step_wall_time = float(
                loop_stats.get("scheduler_step_wall_time", 0.0)
            )
            if scheduler_step_wall_time <= 0:
                scheduler_step_wall_time = (
                    rollout_wall_time + trainer_wait_for_sampler_time + train_wall_time
                )
            if scheduler_step_wall_time > 0:
                metrics["perf/scheduler_step_wall_time"] = scheduler_step_wall_time
                metrics["perf/step_samples_per_s"] = (
                    filtered_samples / scheduler_step_wall_time
                )
                metrics["perf/step_tokens_per_s"] = (
                    completion_tokens / scheduler_step_wall_time
                )
                if rollout_wall_time > 0:
                    metrics["perf/rollout_batch_wall_ratio"] = (
                        rollout_wall_time / scheduler_step_wall_time
                    )

            if rollout_wall_time > 0:
                metrics["perf/rollout_batch_samples_per_s"] = (
                    filtered_samples / rollout_wall_time
                )
                metrics["perf/rollout_batch_tokens_per_s"] = (
                    completion_tokens / rollout_wall_time
                )

    return metrics
