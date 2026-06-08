# RL: sampler timeout diagnostics

Use this reference when SDK rollout sampling raises
`DeploymentSamplerTimeoutError`, or when users report repeated timeout-like
sampler failures such as HTTP 408, HTTP 504, `ReadTimeout`,
`ConnectTimeout`, `WriteTimeout`, or `PoolTimeout`.

The SDK error is a diagnostic summary, not a root-cause verdict. A
timeout-like failure means the sampler exhausted its retry budget after one of
those transient classes. It does **not** prove that RL rollout concurrency is
too high, that the deployment is underprovisioned, or that the gateway is at
fault.

## What to read from the error

The SDK includes facts it can know at the `DeploymentSampler` boundary:

- `raw_error`: the last timeout-like failure class/status.
- `model`: the deployment/model string used for sampling.
- `prompt_tokens` and `max_tokens`: request shape.
- `http_timeout`: client timeout passed to the sampler.
- `sampler_concurrency_window`: current SDK concurrency-controller window, if
  the sampler has one.
- `recent_prefill_queue_p95`, `recent_generation_queue_p95`,
  `recent_client_ttft_p95`, `recent_concurrent_requests_max`: recent serving
  metrics, when successful responses provided them before the failure.

Treat missing metrics as missing evidence. A final timeout can happen before a
deployment returns any metrics.

## Triage order

1. **Confirm the request shape.** Large `prompt_tokens + max_tokens` increases
   the chance of hitting client, gateway, or server-side time limits. If the
   timeout happens only on long rows, first reduce `max_completion_tokens` or
   filter/truncate pathological prompts.
2. **Check queue and TTFT evidence.** High recent queue p95 or high TTFT
   suggests sampler pressure. For RL rollouts, reduce the actual fan-out knob
   in use (`max_concurrency_rollout_sample` in `async_rl_loop.py`, or the
   sync-loop concurrency controller), reduce `max_completion_tokens`, or add
   sampler capacity.
3. **If queue/TTFT evidence is not high or is absent, do not claim capacity as
   the cause.** Investigate gateway timeout limits, network stability, hotload
   readiness, request size, or an overly short `DeployConfig.sample_timeout`.
4. **Compare trainer/sampler waits for async RL.** If
   `perf/trainer_wait_for_sampler_time` is high and sampler queue metrics are
   high, the sampler is the slow side. If `perf/sampler_wait_for_trainer_time`
   is high, the trainer or off-policy gate is more likely limiting progress.
   See [`async-rl.md#diagnosing-waits`](async-rl.md#diagnosing-waits).

## What not to infer

- Do not infer RL context from a recipe name, stack trace, or HTTP status
  alone.
- Do not add `timeout_diagnostic_context` to cookbook recipes by default.
  Cookbook recipes should pass only real sampler request kwargs.
- Do not tell users to increase sampler capacity unless queue/TTFT metrics or
  other hard evidence point there.

## When explicit context is acceptable

The SDK accepts an optional `timeout_diagnostic_context` kwarg for callers that
own the sampling boundary and have explicit, caller-side facts they want echoed
back in the final diagnostic. Use it only in custom rollout code when the
values are facts, not guesses; the SDK strips this key before sending the
inference request.

Good examples:

```python
sample_kwargs = {
    "max_tokens": cfg.max_completion_tokens,
    "http_timeout": cfg.deployment.sample_timeout,
    "timeout_diagnostic_context": {
        "workload": "my_rollout_worker",
        "request_group": group_id,
        "concurrency_limit": rollout_semaphore_limit,
    },
}
```

Avoid cookbook-wide or recipe-wide markers that imply a root cause before the
metrics support it.

## Related

- [`async-rl.md`](async-rl.md): async RL overlap, waits, and off-policy gate.
- [`concurrency.md`](concurrency.md): sync RL sampler concurrency control.
- [`hotload.md`](hotload.md): sampler hotload and deployment readiness.
