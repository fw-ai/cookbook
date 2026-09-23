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

The recipe creates one sampler and injects it into `RolloutSetup`; training and
evaluation rollouts share its HTTP pool and adaptive-concurrency controller.
Rollout factories must reuse `setup.sampler` when present rather than creating a
second sampler or controller. This sampler-level controller is distinct from
the recipe's rollout-call admission cap above.

On a dedicated deployment, the controller uses prefill queue duration as its
congestion signal. Serverless inference does not expose that metric, so the
controller logs a one-time fallback warning and uses response status instead:
HTTP 429, HTTP 503, and transport failures multiplicatively decrease the
window, while an interval of successful responses additively increases it.
Other HTTP failures do not cause the window to grow. Congestion reduces the
window immediately; a short cooldown avoids repeated reductions from one burst.

## Deployment sizing

Harbor Mini-SWE and Pi materialize TITO artifacts in one owned, spawned CPU
process and return the normal `RolloutRun`. This is independent of
the negotiated communication version. Keep script entrypoints under `if __name__ == "__main__":`;
close runners with `aclose()` to release the worker. Cancellation drains a
running conversion before its temporary files are removed.

Recipe concurrency is only admission control. Deployment replicas and batch
capacity still determine actual serving throughput. If the trainer repeatedly
waits for rollout batches, increase rollout capacity or reduce the optimizer
batch size after checking the async performance metrics.

## Large batches and metadata-only future retrieval

Use `build_training_datum_from_token_mask` when only the datum is needed;
`build_datum_from_token_mask` also returns rendered token metadata. Both preserve
input ownership and next-token alignment. For server-side GRPO,
`perf/fwd_bwd_time` measures SDK submission through decoded results, excluding
datum preparation, Torch conversion and PPO/KL diagnostics.

The SDK automatically enables optimized request encoding, bounded numeric
validation and concurrent chunk submission when the selected trainer's completed
model-creation response advertises `comms: "v2"`. This works for both dedicated
and serverless trainers; an unbound serverless session cannot advertise a
trainer capability. Missing or unknown capabilities retain legacy behavior.
Policy and reference models negotiate independently, including after resume.
Inspect `training_client.comms` to see the resolved `"v1"` or `"v2"` mode. No
manual service or cookbook flag is needed. Only v2 requests carry `comms: "v2"`;
v1 requests retain the original wire format. No control-plane change or extra
request is needed. Comms v2 F/B polling waits up to 20 seconds; v1
polling retains the five-second cadence. Older SDKs ignore the added capability.

Compatible optimizations apply to every client: request-size estimation,
trainer numeric preparation, result gathering, asynchronous completion and
forward/F/B response caching. Result bytes are reused for metadata peeks and
retries with the existing sanitization, telemetry and retention semantics.

Request uploads and future-result downloads have separate contracts:

- The SDK splits large requests into transport chunks. Comms v1 clients submit
  chunks `2..N` concurrently, then chunk `1`; comms v2 clients start chunk
  `1` first and submit the rest concurrently. Trainer sequence ordering stays
  intact, but a logical call can span multiple physical batches. Metadata-only
  retrieval does not change uploads or enable streamed responses.
- A large completed future response downloads outputs such as per-token
  logprobs. On a compatible trainer, metadata-only retrieval lets the trainer
  report that the future is complete and state its response size before the
  client fetches the full body. The client can then reserve download capacity
  and avoid concurrent multi-megabyte response fan-out.

Metadata-only retrieval is the default for supported Tinker clients and
trainers. The client reserves the advertised response size before fetching the
full body, bounding concurrent result downloads without changing forward
execution or replaying the request. The cookbook does not expose a separate
toggle and must not monkeypatch private `tinker` internals.

Do not describe metadata-only retrieval as a generic large-batch or
upload-timeout mode. It controls completed future-result downloads, not request
uploads or trainer execution.
