"""Step-metric helpers shared by training loops.

Each helper takes objects a loop already holds and returns a flat
``dict[str, float]``. A loop builds one dict per optimizer step, merges the
helpers it wants, and logs that dict once::

    with elapsed_timer("fwd_bwd"):
        fwd_bwd = policy.forward_backward(data, "cross_entropy")
    with elapsed_timer("optim_step"):
        optim = policy.optim_step(adam_params)
    metrics = flush_timing()  # perf/{phase}_time
    metrics.update(forward_backward_metrics([fwd_bwd]))
    metrics.update(optimizer_metrics(optim))
    metrics.update(sampler_metrics(sampler.drain_metrics()))
    log_metrics(metrics, step=step)

Helpers are pure: they never log, never call a trainer or deployment (the
loop drains the sampler itself), and never read loop configuration. A helper with nothing to report returns ``{}``.

Prefixes:

``perf/``
    Client wall time per phase (``perf/{phase}_time``, from
    :mod:`training.utils.timer`), the trainer's own execution time for the
    same phases (``perf/trainer_{phase}_time``) and throughput.
``train/``
    Values the trainer or a client loss reported: loss, token counts,
    gradient norms and learning rate.
``sampler/`` (``eval/sampler/`` for the sync loop's evaluation traffic)
    Inference timings the deployment returned with each sampling request.

RL rollout statistics (``rollout/*``) live in
:mod:`training.utils.rl.metrics`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "forward_backward_metrics",
    "optimizer_metrics",
    "sampler_metrics",
    "weight_sync_metrics",
]

# Bookkeeping keys the trainer or SDK attach to every result; not diagnostics.
_SKIPPED_RESULT_KEYS = frozenset({"step_id", "step", "response_tokens", "total_tokens"})
# Plain (unsuffixed) counts that add across chunks instead of averaging.
_SUMMED_RESULT_KEYS = frozenset({"active_tokens", "total_resp_tokens"})
# Trainer-reported execution time, mapped next to the client wall time.
_TRAINER_TIME_KEYS = {
    "trainer/forward_backward_duration_s": "perf/trainer_fwd_bwd_time",
    "trainer/optim_step_duration_s": "perf/trainer_optim_step_time",
}
_TRAINER_BUSY_KEY = "trainer_busy_walltime_pct"

_OPTIMIZER_KEYS = (
    "grad_norm_pre_norm",
    "grad_norm",
    "grad_norm_rms",
    "grad_norm_attn",
    "grad_norm_mlp",
    "grad_norm_embed",
    "grad_norm_norm",
    "grad_norm_lora",
    "grad_norm_other",
    "lr",
)

# ServerMetrics attribute -> metric base name (seconds unless noted).
_SAMPLER_DURATIONS = {
    "client_latency": "client_latency_s",
    "server_processing_time": "server_time_s",
    "server_ttft": "ttft_s",
    "prefill_queue_duration": "prefill_queue_s",
    "generation_queue_duration": "generation_queue_s",
    "prefill_duration": "prefill_s",
    "generation_duration": "generation_s",
    "num_concurrent_requests": "concurrent_requests",
}


def _finite(value: Any) -> float | None:
    # Booleans are numeric evidence here (e.g. ``dp_sharded_counts:min``).
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def forward_backward_metrics(results: Sequence[Any]) -> dict[str, float]:
    """Reduce the ``metrics`` of one step's forward/backward results.

    Keys follow Tinker's ``name:reduction`` convention. ``:sum``, ``:min`` and
    ``:max`` reduce across chunks as named; unsuffixed values average over the
    chunks that reported them, so an optional diagnostic is never diluted by
    chunks without it. ``:last`` values describe one call and are dropped,
    except the trainer's busy wall-time percentage.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    reducers: dict[str, str] = {}
    trainer_time: dict[str, float] = {}
    trainer_busy: float | None = None
    for result in results:
        for key, raw in (getattr(result, "metrics", None) or {}).items():
            value = _finite(raw)
            if value is None:
                continue
            base, separator, reducer = key.partition(":")
            if base in _TRAINER_TIME_KEYS:
                name = _TRAINER_TIME_KEYS[base]
                trainer_time[name] = trainer_time.get(name, 0.0) + value
                continue
            if base == _TRAINER_BUSY_KEY:
                trainer_busy = value
                continue
            if base in _SKIPPED_RESULT_KEYS or reducer == "last":
                continue
            if separator and reducer not in {"sum", "min", "max"}:
                continue
            reducers[key] = reducer
            if reducer == "min":
                totals[key] = min(totals.get(key, value), value)
            elif reducer == "max":
                totals[key] = max(totals.get(key, value), value)
            else:
                totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1

    metrics: dict[str, float] = {}
    for key, total in totals.items():
        additive = key.partition(":")[0] in _SUMMED_RESULT_KEYS or reducers[key]
        metrics[f"train/{key}"] = total if additive else total / counts[key]
    active = metrics.get("train/active_tokens")
    response = metrics.get("train/total_resp_tokens")
    if active is not None and response:
        metrics["train/mask_ratio"] = active / response
    metrics.update(trainer_time)
    if trainer_busy is not None:
        metrics["perf/trainer_busy_walltime_pct"] = trainer_busy
    return metrics


def optimizer_metrics(result: Any) -> dict[str, float]:
    """One value for each optimizer diagnostic the trainer returned."""
    raw = getattr(result, "metrics", None) if result is not None else None
    if not raw:
        return {}

    def value(key: str) -> Any:
        return raw.get(key, raw.get(f"{key}:last"))

    metrics: dict[str, float] = {}
    for key in _OPTIMIZER_KEYS:
        if value(key) is not None:
            metrics[f"train/{key}"] = value(key)
    # The trainer reports the post-clip norm even without clipping, where it
    # duplicates the pre-clip norm; keep it only when clipping changed it.
    post_clip = value("grad_norm_post_clip")
    if post_clip is not None and post_clip != metrics.get("train/grad_norm"):
        metrics["train/grad_norm_post_clip"] = post_clip
    duration = _finite(raw.get("trainer/optim_step_duration_s:sum"))
    if duration is not None:
        metrics["perf/trainer_optim_step_time"] = duration
    return metrics


def sampler_metrics(records: Sequence[Any], *, prefix: str = "sampler") -> dict[str, float]:
    """Summarize one step's sampler request records under ``{prefix}/*``.

    Pass ``sampler.drain_metrics()``: the deployment returns per-request
    timings in response headers and the SDK keeps one record per completed or
    failed attempt, with its 1-based ``retry_attempt``, until drained. Draining
    stays in the loop so each record is reported once.

    An attempt is an error when it set ``error`` (non-retryable HTTP errors,
    truncated streams), failed with HTTP 4xx/5xx, or failed below HTTP. SDKs
    released before per-attempt records did not record terminal failures or
    number successful retries, so those undercount ``errors`` and ``retries``.

    ``{prefix}/overhead_s_mean`` is client latency minus server processing
    time: transport, gateway and client-side queueing.
    """
    records = list(records or [])
    if not records:
        return {}

    metrics: dict[str, float] = {f"{prefix}/requests": float(len(records))}
    metrics[f"{prefix}/errors"] = float(
        sum(
            getattr(r, "error", None) is not None
            or bool(getattr(r, "transport_error", False))
            or (getattr(r, "http_status_code", None) or 0) >= 400
            for r in records
        )
    )
    metrics[f"{prefix}/retries"] = float(
        sum((getattr(r, "retry_attempt", None) or 1) > 1 for r in records)
    )
    for attribute, name in _SAMPLER_DURATIONS.items():
        values = [
            v
            for r in records
            if (v := _finite(getattr(r, attribute, None))) is not None
        ]
        if values:
            metrics[f"{prefix}/{name}_mean"] = sum(values) / len(values)
            metrics[f"{prefix}/{name}_max"] = max(values)

    overheads = [
        latency - server
        for r in records
        if (latency := _finite(getattr(r, "client_latency", None))) is not None
        and (server := _finite(getattr(r, "server_processing_time", None))) is not None
    ]
    if overheads:
        metrics[f"{prefix}/overhead_s_mean"] = sum(overheads) / len(overheads)

    prompt_tokens = cached_tokens = 0
    for r in records:
        prompt = _finite(getattr(r, "prompt_tokens", None))
        cached = _finite(getattr(r, "cached_prompt_tokens", None))
        if prompt is not None and cached is not None:
            prompt_tokens += prompt
            cached_tokens += cached
    if prompt_tokens > 0:
        metrics[f"{prefix}/prompt_cache_hit_ratio"] = cached_tokens / prompt_tokens

    generated = generation_time = 0.0
    for r in records:
        tokens = _finite(getattr(r, "completion_tokens", None))
        seconds = _finite(getattr(r, "generation_duration", None))
        if tokens is not None and seconds is not None:
            generated += tokens
            generation_time += seconds
    if generation_time > 0:
        metrics[f"{prefix}/generation_tokens_per_s"] = generated / generation_time
    return metrics


def weight_sync_metrics(result: Any) -> dict[str, float]:
    """Phase times a weight sync reported, under ``perf/weight_sync_*_time``.

    RDMA publication returns ``capture_s`` and ``hotload_s``; file-based
    publication returns nothing, and the loop's own ``perf/weight_sync_time``
    remains the only measurement.
    """
    raw = getattr(result, "metrics", None)
    if not isinstance(raw, Mapping):
        return {}
    metrics: dict[str, float] = {}
    for key, raw_value in raw.items():
        value = _finite(raw_value)
        if value is not None and key.endswith("_s"):
            metrics[f"perf/weight_sync_{key[: -len('_s')]}_time"] = value
    return metrics
