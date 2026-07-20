"""Reusable metric helpers for RL cookbook loops."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import tinker

from training.utils.rl.losses import PromptGroup

logger = logging.getLogger(__name__)

_SKIP_REMOTE_KEYS = {"step_id", "step"}
_LOOP_STAT_PASSTHROUGH_KEYS = (
    "async/version_offset_mean",
    "async/version_offset_max",
    "async/version_offset_min",
    "async/in_flight",
    "async/in_flight_at_train_start",
    "async/in_flight_done_at_train_start",
    "async/running_samples",
    "async/accepted_samples",
    "async/staleness_capacity_at_step",
    "async/concurrency_capacity_at_step",
    "async/max_in_flight_during_train",
    "async/completion_refills_during_train",
    "perf/trainer_wait_for_chunk_time",
    "pipeline/chunk_idx",
    "pipeline/chunk_prompt_groups",
    "pipeline/prompt_groups_accumulated",
    "pipeline/prompt_groups_per_step",
    "pipeline/prompt_groups_per_chunk",
    "pipeline/chunks_per_step",
    "async/max_ready_training_chunks_during_train",
    "pipeline/configured_chunks_per_step",
    "pipeline/requested_chunks_per_step",
    "batch/optimizer_prompt_groups",
)


def median(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


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


def datum_loss_mask(datum: tinker.Datum) -> list[float] | None:
    """Per-position loss mask for a datum (length == target_tokens), or None.

    The adapter writes the mask under ``"weights"`` (legacy datums use
    ``"loss_mask"``).  Returned values are floats; callers compare against
    a positive threshold to find active positions.
    """
    td = datum.loss_fn_inputs.get("weights") or datum.loss_fn_inputs.get("loss_mask")
    if td is None:
        return None
    return list(getattr(td, "data", []) or [])


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


def build_accumulated_async_loop_stats(
    *,
    prompt_groups: Sequence[PromptGroup],
    latest_loop_stats: dict[str, Any] | None,
    trainer_wait_for_sampler_time: float,
    sampler_wait_for_trainer_time: float,
    train_wall_time: float,
) -> dict[str, Any] | None:
    """Merge per-chunk scheduler stats into one optimizer-step stats payload."""
    if latest_loop_stats is None:
        return None

    loop_stats = dict(latest_loop_stats)
    loop_stats["valid_prompt_groups"] = len(prompt_groups)
    loop_stats["all_raw_rewards"] = [
        reward for pg in prompt_groups for reward in pg.rewards
    ]
    loop_stats["trainer_wait_for_sampler_time"] = trainer_wait_for_sampler_time
    loop_stats["sampler_wait_for_trainer_time"] = sampler_wait_for_trainer_time
    loop_stats["train_wall_time"] = train_wall_time
    loop_stats["scheduler_step_wall_time"] = trainer_wait_for_sampler_time + train_wall_time
    return loop_stats


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
    loop_stats: dict | None = None,
    completions_per_prompt: int = 1,
) -> dict[str, Any]:
    """Compute all per-step wandb metrics from prompt groups and remote results.

    Consolidates reward/accuracy/entropy/truncation/filter/batch stats
    into a single metrics dict.  Called by ``finish_step`` in the recipe.
    """
    metrics = dict(timing_metrics)

    total_model_tokens = total_target_tokens(prompt_groups)
    metrics["train/target_tokens"] = total_model_tokens
    metrics["train/effective_accumulation_steps"] = n_accum
    add_train_perf_metrics(metrics, total_model_tokens=total_model_tokens)

    if optim_result and hasattr(optim_result, "metrics") and optim_result.metrics:
        for k, v in optim_result.metrics.items():
            if k not in _SKIP_REMOTE_KEYS:
                metrics[f"train/{k}"] = v

    # Mean across inner minibatches (last-only would hide that early
    # minibatches don't clip yet while later ones do).
    if fwd_bwd_results:
        accum: dict[str, float] = {}
        for result in fwd_bwd_results:
            for k, v in result.metrics.items():
                if k in _SKIP_REMOTE_KEYS:
                    continue
                accum[k] = accum.get(k, 0.0) + v
        n = len(fwd_bwd_results)
        for k, v in accum.items():
            metrics[f"train/{k}"] = v / n

        # Per-step KLD diagnostics on a clean one-point-per-optimizer-step
        # axis.  In async pipeline mode each optimizer step accumulates a
        # variable number of fwd/bwd chunks (the scheduler flushes ready
        # prompt groups instead of waiting to fill a chunk), and the
        # per-chunk ``train/inference_kld`` lands many noisy partial-batch
        # points on the ``train/step`` axis.  These ``kld/*`` keys are not
        # ``train/``-prefixed so they survive the recipe's ``train/`` wandb
        # filter and let us separate a real per-step drift (mean rising)
        # from a single-chunk spike artifact (max >> mean).
        kld_chunk_vals = [
            r.metrics["inference_kld"]
            for r in fwd_bwd_results
            if getattr(r, "metrics", None) and "inference_kld" in r.metrics
        ]
        if kld_chunk_vals:
            metrics["kld/inference_step_mean"] = sum(kld_chunk_vals) / len(kld_chunk_vals)
            metrics["kld/inference_step_max"] = max(kld_chunk_vals)
            metrics["kld/inference_step_min"] = min(kld_chunk_vals)
            metrics["kld/chunks_per_step"] = len(kld_chunk_vals)

        # Signed-bias (k1) bracketing + a directionality ratio to tell a real
        # numerics regression apart from rollout-entropy noise on the same
        # ``kld/*`` (train-filter-surviving) axis. k1 keeps sign, so min<0<max
        # is informative (a single-chunk directional spike shows as |min|/|max|
        # approaching the diff, while symmetric noise straddles zero).
        k1_chunk_vals = [
            r.metrics["inference_k1"]
            for r in fwd_bwd_results
            if getattr(r, "metrics", None) and "inference_k1" in r.metrics
        ]
        diff_chunk_vals = [
            r.metrics["inference_diff"]
            for r in fwd_bwd_results
            if getattr(r, "metrics", None) and "inference_diff" in r.metrics
        ]
        if k1_chunk_vals:
            k1_mean = sum(k1_chunk_vals) / len(k1_chunk_vals)
            metrics["kld/inference_k1_step_mean"] = k1_mean
            metrics["kld/inference_k1_step_max"] = max(k1_chunk_vals)
            metrics["kld/inference_k1_step_min"] = min(k1_chunk_vals)
            # Directionality ratio in [0, 1]: |mean signed diff| / mean |diff|.
            #   ~1 => logprob gap is one-directional  => systematic numerics drift.
            #   ~0 => gap cancels (symmetric)         => rollout entropy / outlier
            #         tokens, i.e. a large kld/inference_step_* here is NOISE, not
            #         a numerics regression.
            if diff_chunk_vals:
                diff_mean = sum(diff_chunk_vals) / len(diff_chunk_vals)
                if diff_mean > 0:
                    metrics["kld/inference_directionality"] = abs(k1_mean) / diff_mean

    all_rewards: list[float] = []
    all_comp_lens: list[int] = []
    all_truncated: list[bool] = []
    correct_count = 0
    total_samples = 0
    total_trained_datums = 0

    for pg in prompt_groups:
        all_rewards.extend(pg.rewards)
        all_comp_lens.extend(pg.completion_lens)
        all_truncated.extend(pg.truncated)
        correct_count += sum(1 for r in pg.rewards if r > 0.5)
        total_samples += len(pg.rewards)
        total_trained_datums += len(pg.data)

    metrics["rollout/samples_completed"] = total_samples
    metrics["rollout/trained_datums"] = total_trained_datums
    if total_samples > 0:
        metrics["rollout/reward"] = sum(all_rewards) / total_samples
        metrics["rollout/accuracy"] = correct_count / total_samples
    add_response_length_stats(metrics, all_comp_lens)
    if all_truncated:
        metrics["rollout/truncated_ratio"] = sum(all_truncated) / len(all_truncated)

    # Entropy is a mean of -logprob over loss_mask>0 positions only.
    # Multi-turn rollouts emit loss_mask=0 / logprobs=0.0 on bridge/user/tool
    # tokens; including them biases entropy toward 0.
    entropy_vals: list[float] = []
    for pg in prompt_groups:
        per_sample = pg.prompt_lens if pg.prompt_lens is not None else None
        for i, inf_lp in enumerate(pg.inf_logprobs):
            sample_prompt_len = (
                per_sample[i] if per_sample is not None and i < len(per_sample)
                else pg.prompt_len
            )
            resp_start = max(0, sample_prompt_len - 1)
            resp_lp = inf_lp[resp_start:] if len(inf_lp) > resp_start else []
            if not resp_lp:
                continue
            mask = datum_loss_mask(pg.data[i]) if i < len(pg.data) else None
            if mask is not None:
                resp_mask = mask[resp_start : resp_start + len(resp_lp)]
                active = [lp for lp, m in zip(resp_lp, resp_mask) if m > 0.5]
            else:
                active = list(resp_lp)
            if active:
                entropy_vals.append(-sum(active) / len(active))
    if entropy_vals:
        metrics["rollout/entropy"] = sum(entropy_vals) / len(entropy_vals)

    if loop_stats:
        valid = loop_stats["valid_prompt_groups"]
        filtered = loop_stats["filter_drops"]
        filter_ratio = filtered / max(1, valid + filtered)
        metrics["rollout/valid_prompt_groups"] = valid
        metrics["rollout/total_sampled"] = loop_stats["total_sampled"]
        metrics["rollout/filter_accept_ratio"] = 1.0 - filter_ratio
        metrics["rollout/filter_reject_ratio"] = filter_ratio
        metrics["rollout/sample_fail_count"] = loop_stats["sample_fails"]
        metrics["rollout/fwd_bwd_count"] = n_accum
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
            metrics["perf/step_samples_per_s"] = total_samples / scheduler_step_wall_time
            metrics["perf/step_tokens_per_s"] = sum(all_comp_lens) / scheduler_step_wall_time
            if rollout_wall_time > 0:
                metrics["perf/rollout_batch_wall_ratio"] = rollout_wall_time / scheduler_step_wall_time
            if trainer_wait_for_sampler_time > 0:
                metrics["perf/trainer_idle_ratio"] = trainer_wait_for_sampler_time / scheduler_step_wall_time
            if train_wall_time > 0:
                metrics["perf/train_step_wall_ratio"] = train_wall_time / scheduler_step_wall_time

        if rollout_wall_time > 0:
            metrics["perf/rollout_batch_samples_per_s"] = total_samples / rollout_wall_time
            metrics["perf/rollout_batch_tokens_per_s"] = sum(all_comp_lens) / rollout_wall_time

        raw_rewards = loop_stats["all_raw_rewards"]
        if raw_rewards:
            metrics["rollout/raw_samples_completed"] = len(raw_rewards)
            metrics["rollout/raw_reward"] = sum(raw_rewards) / len(raw_rewards)
            metrics["rollout/raw_accuracy"] = (
                sum(1 for reward in raw_rewards if reward > 0.5) / len(raw_rewards)
            )

    return metrics
