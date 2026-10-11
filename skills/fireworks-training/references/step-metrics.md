# Step metrics in your own loop

The cookbook recipes are reference loops. When you fork one or write your own,
build each optimizer step's metrics from the same helpers the recipes use. Each
helper takes objects the loop already holds and returns a flat dictionary. Log
one dictionary per step:

```python
from training.utils import log_metrics
from training.utils.metrics import (
    forward_backward_metrics,
    optimizer_metrics,
    sampler_metrics,
    weight_sync_metrics,
)
from training.utils.rl.metrics import rollout_metrics
from training.utils.timer import elapsed_timer, flush_timing

with elapsed_timer("fwd_bwd"):
    fwd_bwd = policy.forward_backward_custom(data, loss_fn)
with elapsed_timer("optim_step"):
    optim = policy.optim_step(adam_params, emit_grad_norm_metrics=True)
with elapsed_timer("weight_sync"):
    synced = publish_weights(f"step-{step}")

metrics = flush_timing()  # perf/fwd_bwd_time, perf/optim_step_time, ...
metrics.update(rollout_metrics(prompt_groups, raw_rewards=raw_rewards))
metrics.update(forward_backward_metrics([fwd_bwd]))
metrics.update(optimizer_metrics(optim))
metrics.update(sampler_metrics(sampler.drain_metrics()))
metrics.update(weight_sync_metrics(synced))
metrics["train/step"] = step
log_metrics(metrics, step=step)
```

The helpers are pure: they do not log, call the trainer or deployment, or read
loop configuration; the loop drains the sampler itself. A helper with nothing to report returns `{}`, so a missing
series means the source did not report it, never zero.

`training.utils.rl.metrics.compute_step_metrics` composes the trainer and
rollout helpers. The RL recipes add `sync_step_perf_metrics(loop_stats, groups)`
or `async_step_perf_metrics(loop_stats, groups)` for their wall-time split, then
the sampler and weight-sync helpers.

## Helpers

| Helper | Input | Metrics |
|---|---|---|
| `flush_timing()` | `elapsed_timer(name)` blocks since the last flush | `perf/{name}_time`: client wall seconds |
| `forward_backward_metrics(results)` | One step's forward/backward results, one per chunk | `train/*` loss, policy-objective and custom-loss diagnostics, `perf/trainer_fwd_bwd_time`, `perf/trainer_busy_walltime_pct` |
| `optimizer_metrics(result)` | The `optim_step` result | `train/grad_norm*`, `train/lr`, `perf/trainer_optim_step_time` |
| `sampler_metrics(records, prefix="sampler")` | `sampler.drain_metrics()`: request records since the last drain | `sampler/*` (`eval/sampler/*` for evaluation) |
| `weight_sync_metrics(result)` | The weight-publication result | `perf/weight_sync_{phase}_time` |
| `rollout_metrics(groups, raw_rewards=...)` | The step's `PromptGroup`s and every sampled reward before filtering | `rollout/*` |

### Trainer results

Tinker result keys use `name:reduction`. `:sum`, `:min` and `:max` reduce
across chunks as named and keep their suffix (`train/loss:sum`). Unsuffixed
diagnostics average over the chunks that reported them. `:last` values describe
one call and are dropped, except `trainer_busy_walltime_pct`.

When the trainer reports its own execution time, compare it with the client
wall time of the same phase:

| Client | Trainer | Difference |
|---|---|---|
| `perf/fwd_bwd_time` | `perf/trainer_fwd_bwd_time` | Transport, request queueing and client-side work |
| `perf/optim_step_time` | `perf/trainer_optim_step_time` | Same, for the optimizer step |

`perf/trainer_busy_walltime_pct` is the trainer's recent busy share of wall
time. A low value with a long `perf/step_time` means the trainer waits for the
client, usually for rollouts.

### Policy objectives

Each objective module in `training.utils.rl.algorithm` computes ratio
diagnostics from the logprobs a server-side policy loss returns. Merge them
into the result before `forward_backward_metrics` adds the `train/` prefix:

```python
result = policy.forward_backward(datums, objective.LOSS_FN, loss_fn_config=objective.loss_fn_config(cfg))
result.metrics.update(objective.metrics(batch, result, cfg))      # trust region
result.metrics.update(inference_drift_metrics(batch, result))     # sampler/trainer drift
```

| Metric | Meaning |
|---|---|
| `train/{prefix}_ratio_mean` | Policy/anchor probability ratio over trained tokens, averaged per datum; GSPO uses the sequence ratio |
| `train/{prefix}_clip_frac` | Share of ratios outside the objective's trust region; for DPPO, tokens past the divergence threshold. Absent for DRO and importance sampling |
| `train/{prefix}_kl` | k3 estimate of the policy/anchor KL, averaged per datum |
| `train/active_tokens`, `train/total_resp_tokens`, `train/mask_ratio` | Trained response tokens, all response tokens, and their ratio |
| `train/inference_k1`, `train/inference_k3` | k1/k3 drift from raw serving logprobs to the trainer policy |
| `train/raw_inference_logprob_coverage` | Share of trained tokens compared for drift; `0.0` when serving returned no raw logprobs |
| `train/anchor_k1`, `train/anchor_k3` | Rollout-to-anchor drift when a fork recomputes the anchor (`anchor_drift_metrics`) |
| `train/tis_weight_mean`, `train/tis_clip_frac` | Mean truncated importance weight and share cut at the cap (`apply_tis`) |

The prefix is `ppo` for GRPO, then `gspo`, `dapo`, `cispo`, `dro`, `dppo` and
`is` (importance sampling). Ratios are against `batch.anchor_logprobs` (the
rollout logprobs unless a fork recomputes them). A client-side loss
(`make_loss_fn`) reports the same `{prefix}_*` names, plus extras such as
`train/mean_loss` and `train/ref_kl` with reference KL. With sampling-support
replay, skip `inference_drift_metrics`: serving logprobs are full-vocabulary.

### Sampler

The deployment returns per-request timings with each completion. Call
`sampler_metrics(sampler.drain_metrics())` once per step; it reports every
request since the previous drain. `rl_loop` evaluates between steps and logs
those requests as `eval/sampler/*`; in `async_rl_loop` evaluation overlaps
training, so `sampler/*` includes evaluation requests.

| Metric | Meaning |
|---|---|
| `sampler/requests` | Request attempts, including retries and failed attempts |
| `sampler/errors` | Failed attempts: HTTP 4xx/5xx, truncated streams, transport errors |
| `sampler/retries` | Attempts after the first for a logical request |
| `sampler/client_latency_s_{mean,max}` | Client-measured seconds per completed attempt |
| `sampler/server_time_s_{mean,max}` | Server processing seconds |
| `sampler/overhead_s_mean` | Client latency minus server time: transport and gateway |
| `sampler/ttft_s_{mean,max}` | Server time to first token |
| `sampler/prefill_queue_s_{mean,max}` | Wait before prefill. High values mean the deployment is saturated |
| `sampler/generation_queue_s_{mean,max}` | Wait before decoding |
| `sampler/prefill_s_*`, `sampler/generation_s_*` | Prefill and decode seconds |
| `sampler/concurrent_requests_{mean,max}` | Requests the replica was serving |
| `sampler/prompt_cache_hit_ratio` | Cached prompt tokens / prompt tokens |
| `sampler/generation_tokens_per_s` | Generated tokens / decode seconds, per request stream |

The SDK keeps one `ServerMetrics` record per completed or failed attempt
(bounded; oldest dropped first) until `drain_metrics()`. Each record has its
1-based `retry_attempt`, and a failed attempt sets `error`. Draining does not
clear the recent history the SDK's sampling-timeout diagnostics report.
Client latency is timed from the completion request, excluding the R3 format
probe.

Fields a deployment or SDK version does not return are absent. Client latency,
generated tokens and complete error/retry counts need an SDK with per-attempt
records; older SDKs undercount `errors` and `retries`. Custom rollouts that do
not use the recipe's sampler report no `sampler/*` metrics.

### Rollouts

| Metric | Meaning |
|---|---|
| `rollout/raw_reward` | Mean reward before group filtering |
| `rollout/filtered_reward` | Mean reward of the groups trained |
| `rollout/raw_samples`, `rollout/filtered_samples` | Logical rollouts before and after filtering |
| `rollout/filter_ratio` | `1 - filtered_samples / raw_samples` |
| `rollout/completion_tokens_{mean,max}` | Completion length of trained sequences |
| `rollout/truncated_ratio` | Trained sequences that hit the completion-token limit |
| `rollout/trainable_tokens_{mean,max,min}` | Trainable tokens per trajectory, when the rollout reports them |

A `rollout/truncated_ratio` near 1 means almost every completion hit
`max_completion_tokens`; rewards then measure truncation, not the task.

## Run identity

`log_run_resources(service)` logs the policy trainer, reference trainer and
deployment ids and adds them to the W&B run config. The recipes call it once
after the first weight sync. Include these ids when you report a performance
problem: Fireworks uses them to find the trainer and deployment server metrics
for the same run.
