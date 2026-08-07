# Managed RFT remote tracing

*Source of truth: live [Remote Environment Setup](https://docs.fireworks.ai/fine-tuning/connect-environments.md) and [Eval Protocol](https://evalprotocol.io/introduction).*

Use this reference when implementing a managed RFT remote environment, wiring Fireworks tracing, or debugging a reward-to-rollout join. For custom Training API agent trajectories, use `references/rl-agentic.md`.

## Why tracing matters

Agent training needs the complete trajectory, including model calls, tools, environment state, and terminal reward. Tracing supports credit assignment, replay, and diagnosis without forcing the evaluator to infer hidden steps from the final answer.

## Correlation metadata

Carry the identifiers provided by the `/init` request through every log and trace:

- `invocation_id`
- `experiment_id`
- `rollout_id`
- `run_id`
- `row_id`

Do not invent replacements or drop them when spawning a child process.

## Required pieces

1. Use the supplied `model_base_url` unchanged for model calls. It contains the routing and correlation context required by the trainer.
2. Attach `FireworksTracingHttpHandler` and `RolloutIdFilter`, or propagate `EP_ROLLOUT_ID` to child processes.
3. Emit a structured terminal status with `Status.rollout_finished()` or `Status.rollout_error(message)`.

The trainer polls by `rollout_id`, joins status and trace data, and then computes or records the reward.

## Minimal server pattern

```python
import logging

from eval_protocol import (
    FireworksTracingHttpHandler,
    InitRequest,
    RolloutIdFilter,
    Status,
)

logging.getLogger().addHandler(FireworksTracingHttpHandler())


@app.post("/init")
def init(request: InitRequest):
    logger = logging.getLogger(f"eval.{request.metadata.rollout_id}")
    logger.addFilter(RolloutIdFilter(request.metadata.rollout_id))
    try:
        # Use request.model_base_url and request.api_key for policy calls.
        # Run the environment and compute the reviewed reward.
        logger.info("done", extra={"status": Status.rollout_finished()})
    except Exception as exc:
        logger.error(
            "failed",
            extra={"status": Status.rollout_error(str(exc))},
        )
```

Treat exception text as potentially sensitive. Redact credentials, signed URLs, raw customer data, and private environment values before logging.

## Capture

- Input identifiers, deterministic seeds, and retrieval-context summaries
- Model messages, parameters, token counts, and response identifiers
- Tool input and output summaries without credentials
- Per-step and terminal rewards with documented weights
- Errors, timeouts, and artifacts required for verification

## Operational rules

- Keep stable field names for automated filters.
- Emit heartbeats for long rollouts.
- Always finalize success or failure.
- Normalize rewards to the reviewed range.
- Pin code and dependency versions when reproducibility matters.
- Never substitute a reconstructed base URL for `model_base_url`.

## Join failures

When a rollout is missing from reward or trace views:

1. Confirm the same `rollout_id` appears in `/init`, logger filters, child processes, and terminal status.
2. Confirm the policy client used the supplied `model_base_url`.
3. Confirm a terminal status was emitted exactly once.
4. Check for process exits before buffered logs were flushed.
5. Inspect redacted traces before retrying; do not convert a missing trace into reward zero.

## Related

- Managed RFT launch and monitoring: `managed-rft-operations.md`
- Evaluator authoring: `preference-data-and-evaluators.md`
- Custom Training API agent trajectories: `rl-agentic.md`
