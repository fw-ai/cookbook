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

## Large batches and metadata-only future retrieval

Separate the request-upload path from the future-result download path:

- A large request uploads the input datums. The SDK may split one logical batch
  into transport chunks, submit chunks `2..N` in parallel, and submit chunk
  `1` last so the trainer runs the chunks together. Metadata-only retrieval
  does not reduce these uploads and cannot prevent a sequence-arrival gap while
  the request is still in flight.
- A large completed future response downloads outputs such as per-token
  logprobs. On a compatible trainer, metadata-only retrieval lets the trainer
  report that the future is complete and state its response size before the
  client fetches the full body. The client can then reserve download capacity
  and avoid concurrent multi-megabyte response fan-out.

For a large batch that produces large future results, prefer metadata-only
retrieval after confirming that the selected trainer and client both support
the protocol. Do not describe it as a generic large-batch or upload-timeout
mode.

The current staged cookbook intentionally disables metadata-only retrieval
while the trainer fleet is being upgraded. It is not a cookbook default, and
the cookbook does not currently expose a supported public toggle. If Fireworks
has approved the staged rollout for a compatible trainer/client pair, enable it
manually using the rollout-specific instructions. Do not monkeypatch private
`tinker` internals. Otherwise, keep the compatibility behavior until the
backend rollout is complete and the cookbook publishes a supported opt-in or
changes the default.
