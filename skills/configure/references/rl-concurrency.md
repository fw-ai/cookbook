# RL: rollout concurrency

The synchronous and asynchronous recipes expose different concurrency
semantics because they own different schedulers.

## Synchronous `rl_loop`

`prompt_groups_per_step` is the batch-native concurrency boundary. The recipe
samples up to that many prompt groups together, refills rows that return
`None` or fail the dynamic filter, trains the completed batch, and hotloads
before starting another batch.

There is no separate adaptive-concurrency config on the synchronous recipe.
Each built-in prompt-group request asks the deployment for
`completions_per_prompt` completions in one call.

## `async_rl_loop`

The async recipe has independent rollout and training workers. Its scheduling
knobs are documented in [`rl-async.md`](rl-async.md):

- `max_concurrency_rollout_sample` caps in-flight rollout calls;
- `prompt_groups_per_step` sets the optimizer-batch size; and
- `max_head_offpolicy_versions` bounds behavior-policy staleness.

Use the async recipe when rollouts must refill while training is active.

## Deployment sizing

Recipe concurrency is only admission control. Deployment replicas and batch
capacity still determine actual serving throughput. If the trainer repeatedly
waits for rollout batches, increase rollout capacity or reduce the optimizer
batch size after checking the async performance metrics.
