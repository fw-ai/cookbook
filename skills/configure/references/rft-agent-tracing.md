# RFT agent tracing (remote environments)

*Source of truth: live [Remote Environment Setup](https://docs.fireworks.ai/fine-tuning/connect-environments.md), [eval-protocol](https://evalprotocol.io/introduction).*

Use when implementing `/init` for remote RFT rollouts, wiring `tracing.fireworks.ai`, or debugging reward join failures.

## Why tracing matters

RL for agents needs the full trajectory (tool calls, state, intermediate decisions), not just the final answer. Tracing enables credit assignment, replay, and debugging.

## Correlation metadata

From `/init` `metadata`, tag every log and trace:

- `invocation_id`, `experiment_id`, `rollout_id`, `run_id`, `row_id`

## Three pieces

1. **`model_base_url`** from trainer — OpenAI-compatible URL on `https://tracing.fireworks.ai` with embedded correlation IDs. **Always use as-is** for model calls.
2. **`FireworksTracingHttpHandler`** on your logger + `RolloutIdFilter` (or set `EP_ROLLOUT_ID` per child process).
3. **Structured completion** — log `Status.rollout_finished()` or `Status.rollout_error(message)` when the rollout ends.

Trainer polls logs by `rollout_id`, then loads traces and joins for scoring.

## Minimal remote server pattern

```python
import logging
from eval_protocol import InitRequest, Status, FireworksTracingHttpHandler, RolloutIdFilter

logging.getLogger().addHandler(FireworksTracingHttpHandler())

@app.post("/init")
def init(request: InitRequest):
    logger = logging.getLogger(f"eval.{request.metadata.rollout_id}")
    logger.addFilter(RolloutIdFilter(request.metadata.rollout_id))
    # client = LLMClient(base_url=request.model_base_url, api_key=request.api_key)
    try:
        # ... rollout ...
        logger.info("done", extra={"status": Status.rollout_finished()})
    except Exception as e:
        logger.error("fail", extra={"status": Status.rollout_error(str(e))})
```

## Capture in traces (redact secrets)

- Inputs, seeds, retrieval context
- Model calls: messages, params, token counts
- Tool I/O summaries (not raw credentials)
- Per-step and terminal rewards with weights
- Errors, timeouts, artifacts needed for verification

## Best practices

- Deterministic seeds and version pins where possible
- Normalize rewards to a documented range (e.g. 0–1)
- Heartbeats on long rollouts; always finalize with success or failure
- Stable field names for automated filters

## RemoteRolloutProcessor loop

1. Remote server logs `Status.rollout_finished()` / `rollout_error()`
2. Trainer polls Fireworks Tracing by `rollout_id`
3. Status + trace joined → reward computed

## Related

- Environment HTTP contract: live connect-environments docs
- Evaluator authoring: `preference-data-and-evaluators.md`
- Managed RFT launch: `managed-rft-operations.md`
