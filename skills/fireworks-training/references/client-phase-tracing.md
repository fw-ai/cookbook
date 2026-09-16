# Client phase tracing

Use client phase traces to see when rollout, trainer, evaluation, weight-update,
and checkpoint work happened relative to each other. The output is a local
Chrome trace JSON file that opens directly in [Perfetto](https://ui.perfetto.dev).
It complements `metrics.jsonl`: scalar metrics answer "how much," while the
trace timeline answers "when" and "what overlapped."

## Export a trace

Set one environment variable before running a cookbook recipe:

```bash
export COOKBOOK_TRACE_FILE="$PWD/client-phase-trace.json"
python -m training.recipes.rl_loop ...
```

To run one real serverless Countdown RL optimizer step and trace the actual
weight snapshot, rollout, forward/backward, optimizer, and checkpoint calls:

```bash
export FIREWORKS_API_KEY=fw_...
export HF_TRUST_REMOTE_CODE=1
python -m training.examples.tools.client_phase_trace_demo
```

This is a paid training run against the shared serverless pool. It uses the
bundled sample dataset, disables evaluation and promotion, runs one bounded
optimizer step, and writes both the training artifacts and
`client-phase-trace.json` under `./client-phase-trace-demo/`.

The synchronous and asynchronous RL recipes emit their existing coarse phases:

- synchronous rollout batch collection;
- reference and old-policy forward passes;
- forward/backward and optimizer operations;
- sampler weight publication;
- evaluation and evaluation joins;
- checkpoint saves.

The trace is flushed when the recipe calls `wandb_finish()` and again at normal
process exit. It remains local unless the client separately configures the
optional OpenTelemetry bridge.

Open the result:

1. Go to [ui.perfetto.dev](https://ui.perfetto.dev).
2. Select **Open trace file**.
3. Choose `client-phase-trace.json`.

Each thread or asyncio task gets a separate lane. Nested spans carry
`parent_span_id` in their arguments. Select a span to inspect its category,
duration, attributes, and error type.

## Trace custom harness phases

The cookbook cannot infer what a customer harness does inside one rollout.
Instrument those boundaries explicitly:

```python
from training.utils import phase_span


async def run_coding_task(row):
    with phase_span(
        "environment_setup",
        category="harness",
        attributes={"row_id": row["id"]},
    ):
        environment = await create_environment(row)

    with phase_span("agent_loop", category="harness") as span:
        result = await run_agent(environment)
        if span is not None:
            span.set_attribute("turns", result.turns)

    with phase_span("score", category="evaluation"):
        return await score_result(result)
```

The regular context manager is safe around `await`. Concurrent asyncio tasks
and worker threads appear on distinct Perfetto lanes.

Use bounded identifiers and numeric summaries as attributes. Do not attach
prompts, model responses, environment variables, API keys, or other sensitive
payloads.

## Optional OpenTelemetry mirror

Perfetto export has no extra dependency. To mirror the same spans to
OpenTelemetry, install and configure an OpenTelemetry tracer provider in the
client process, then enable the bridge:

```bash
export COOKBOOK_TRACE_FILE="$PWD/client-phase-trace.json"
export COOKBOOK_OTEL_ENABLED=1
```

The cookbook lazily calls `opentelemetry.trace.get_tracer(...)`. It does not
install an SDK, choose an exporter, or configure an OTLP endpoint. Use your
existing OpenTelemetry bootstrap. Applications that already own a tracer may
pass it to `configure_phase_tracing(..., otel_tracer=tracer)`.

If `opentelemetry-api` is absent, Perfetto tracing continues and a warning is
logged.

## Artifact contract

The file is written atomically as a versioned object with `schema_version`,
`trace_origin_unix_ns`, `dropped_events`, and `traceEvents` fields.

Span events use Chrome complete-event records (`ph: "X"`) with microsecond
`ts` and `dur` values. The default in-memory limit is 100,000 completed spans;
additional spans are counted in `dropped_events`. At most 1,024 execution lanes
receive metadata; further thread/task contexts share overflow lane `0`.

## Limits

- These are client wall-clock spans, not GPU utilization or kernel traces.
- Async rollout calls are not traced automatically; instrument the customer
  rollout function when that detail is needed.
- The trace clock is process-local. Use the OpenTelemetry mirror when traces
  must join another process or service.
- A hard kill can lose the unflushed tail. Normal completion and exceptions
  that unwind Python contexts retain completed spans.
- Built-in instrumentation stays phase-level. Per-rollout and per-tool spans
  require explicit harness instrumentation to control trace size and sensitive
  attributes.
