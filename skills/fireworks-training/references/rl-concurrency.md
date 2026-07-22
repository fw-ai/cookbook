# RL: sampling concurrency control

When the recipe fans out rollout requests to the sampler deployment, it needs to cap how many are in flight so the deployment doesn't queue up. This is controlled by `cfg.concurrency: ConcurrencyConfig`, which picks one of two SDK controllers at startup.

## Use adaptive (the default). That's the recommendation.

`cfg.concurrency.mode = "adaptive"` (default) → `AdaptiveConcurrencyController`.

It watches the deployment's `prefill_queue_duration` header on every response and resizes the concurrency window via AIMD to keep queue latency near `prefill_queue_target` (0.5 s default). The window grows under light load, shrinks under pressure. This is the right choice for almost every RL run — leave it alone.

Knobs you rarely need to touch:

| Field | Default | Use when |
|---|---|---|
| `initial_window` | `8 * replica_count` | Bootstrap at a different starting point (e.g. you know the deployment handles more) |
| `min_window` | `1` | Never needs raising; raising hurts back-pressure behaviour |
| `max_window` | `256` | You've profiled the deployment and know a higher ceiling is safe |
| `prefill_queue_target` | `0.5` s | Tighten on a latency-critical deployment, loosen to squeeze more throughput |
| `rollout_adjustment_interval` | `32` | Set to `N > 0` to resize after every N completed rollout requests, including within an RL step; set to `0` for step-boundary-only adjustment |

By default (`rollout_adjustment_interval=32`), the controller adjusts within
each step after every 32 completed requests. At the step boundary,
`step_completed()` adjusts any remaining requests and starts a fresh interval
for the next step. Set the interval to `0` to average the prefill queue over the
whole rollout batch and adjust only at the step boundary.

## When to use `fixed`

`cfg.concurrency.mode = "fixed"` + a `FixedConcurrencyController(max_concurrency)` — a static semaphore. Use **only** when you are implementing your own concurrency control outside the recipe (e.g. driving requests from your own scheduler and you want the recipe to be a pass-through). The recipe will not scale the window for you.

Leaving `mode=None` with a `max_concurrency` set is the deprecated backward-compat path — it emits a warning and constructs a fixed controller. Don't write new code against it.

## Why not just "no concurrency limit"?

Unbounded fan-out is not supported by the recipe path. If you truly want no SDK-side limit (you're handling pacing entirely yourself), pass a `FixedConcurrencyController` with a very high ceiling — but this is almost always wrong for RL because it pins the deployment into head-of-line blocking under any noticeable load.

## Related

- Weight sync cadence (when the sampler sees new weights) → [`rl-hotload.md`](rl-hotload.md)
- `prefill_queue_target` is the same signal the deployment exposes to telemetry; if you're tuning it, watch the deployment's server metrics for prefill queue time in the same range.
- Controller implementations: `fireworks.training.sdk.deployment.AdaptiveConcurrencyController` and `FixedConcurrencyController` (installed under `src/fireworks/training/sdk/deployment.py`).
