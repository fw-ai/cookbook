# RL: dynamic rollout filter

`rl_loop.py` passes `dynamic_filter_fn=should_accept` into `run_rl_loop` (see `recipes/rl_loop.py`, search `should_accept`). The default predicate rejects **zero-variance** groups — groups where every response in the group got the same reward:

```python
def should_accept(pg: PromptGroup) -> bool:
    # Reject groups where all rewards are identical (zero-variance).
    return len(set(pg.rewards)) > 1
```

## Why

Zero-variance groups contribute no gradient signal under GRPO advantages (all advantages are zero), so the recipe drops them before the optimizer step. If a user notices "some samples are filtered" — this is why.

## Customizing

Replace `should_accept` in a fork when you want different filtering:

- Minimum reward threshold: `return max(pg.rewards) >= 0.1`
- Response-length bound: `return all(l >= MIN_LEN for l in pg.completion_lens)`
- Tool-call count: `return all(has_expected_tool(c) for c in pg.completions)`
- Combination: `AND` them together

The filter runs **after rollouts complete and before the train step**, so filtered groups still cost rollout time — use it to protect gradient quality, not to save rollout cost. For the latter, cull prompts upstream.
