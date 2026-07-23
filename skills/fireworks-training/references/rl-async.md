# RL: async recipe (`async_rl_loop.py`)

> ⚠️ **EXPERIMENTAL.** API surface and config fields may change without
> backward-compatibility shims. Pin the cookbook version for production runs.

Use this reference for implementation details, tuning, and debugging. The
customer-facing overview stays in the Fireworks docs.

## Responsibility boundary

Users provide:

- dataset rows through `Config.dataset` or `rows=`;
- `rollout_fn_factory(setup) -> rollout_fn`;
- environment interaction, scoring, and trainer-ready rollout segments;
- algorithm and scheduling config;
- an optional `dynamic_filter_fn`.

The recipe owns trainer/deployment lifecycle, rollout fan-out and admission,
group assembly, advantages, reference and old-policy forwards, GRPO/TIS/KL,
training chunks, the optimizer, sampler hotload, version publication, metrics,
checkpointing, and cleanup.

Keep custom environment logic in the rollout function. Do not put scheduler or
trainer lifecycle state in the rollout.

## Rollout API

The factory runs once. Each rollout call produces one trajectory:

```python
def make_rollout_fn(setup: RolloutSetup) -> RolloutFn:
    async def rollout_fn(sample_prompt: dict) -> RolloutRun | None:
        ...

    return rollout_fn
```

The recipe invokes `rollout_fn` `completions_per_prompt` times for each dataset
row. Return `None` to drop one trajectory draw. Return a `RolloutRun` with one
or more `RolloutSample` segments on success:

```python
@dataclass
class RolloutRun:
    segments: list[RolloutSample]

@dataclass
class RolloutSample:
    tokens: list[int]
    logprobs: list[float]
    loss_mask: list[int]
    reward: float
```

For every segment:

- `tokens`, `logprobs`, and `loss_mask` must have equal lengths;
- `loss_mask=1` marks tokens trained on; prompt, user, tool, and environment
  tokens stay `0`;
- non-generated positions should use `0.0` logprobs;
- all segments in one run must share the same scalar reward.

`RolloutSetup` contains the tokenizer, tokenizer ID, sampling kwargs, inference
base URL, API key, deployment model, group size, and caller-provided `extras`.
The rollout may optionally declare any of these keyword parameters when it
needs dataset position context: `cursor_index`, `row_index`, `epoch`,
`rollout_idx`, `sample_index`, and `end_of_epoch`.

The scheduler's `RolloutRow` is not part of this user contract. It is the
recipe-owned envelope that binds a dataset row to `completions_per_prompt`
rollout calls and an ordered durability callback.

## Architecture and ownership

The scheduler is split by one owner per concern:

| Component | Owns |
|---|---|
| `RolloutProducer` | In-flight rollout tasks, completion-triggered refill, row-atomic admission, group assembly, failure policy, and the durable row cursor |
| `OptimizerBatch` | One optimizer batch, fixed balanced chunk targets, accepted rows, and the async queue of ready `TrainingChunk`s |
| `AsyncRLCoordinator` | The producer/batch handoff, one serialized trainer worker, failure boundaries, and publication |
| `AsyncRLTelemetry` | Rate-limited producer snapshots, completed-batch metric reduction, and reporting |
| `async_rl_loop.main` | Visible algorithm phases: reference/old-policy forwards, forward/backward, optimizer, hotload, telemetry handoff, and checkpoints |

Blocking trainer calls run on the coordinator's single worker thread. The event
loop remains free to retire rollout tasks and refill the producer while trainer
work is active. Gradient mutation is serialized; rollout production is not.

## Five scheduling knobs

Use these symbols when reasoning about capacity:

| Config field | Symbol | Unit | Meaning |
|---|---:|---|---|
| `completions_per_prompt` | `G` | samples per row | GRPO trajectories for one dataset row; must be at least 2 |
| `prompt_groups_per_step` | `P` | rows | Complete prompt groups in one optimizer batch |
| `pipeline_chunks_per_step` | `K` | chunks | Requested forward/backward chunks per optimizer batch |
| `max_head_offpolicy_versions` | `O` | published versions | Admission headroom beyond the current policy version; `0` is fully on-policy |
| `max_concurrency_rollout_sample` | `C` | samples | Optional hard cap on in-flight rollout calls |

One optimizer batch contains `B = P × G` samples. The scheduler creates
`min(P, K)` non-empty, balanced chunk targets before production begins. For
example, `P=10, K=4` creates targets `(3, 3, 2, 2)`.

Chunking changes when forward/backward work can start; it does not change the
optimizer batch. Exactly one optimizer mutation, one sampler hotload, and one
version publication follow all chunks in a batch.

## Row-atomic admission gate

Admission bookkeeping uses samples, the same unit as rollout concurrency. A
row is submitted only when both budgets can fit all `G` rollout calls:

```text
B = prompt_groups_per_step * completions_per_prompt

staleness_capacity =
    (published_version + max_head_offpolicy_versions + 1) * B
    - (accepted_samples_offset + accepted_samples + reserved_samples)

concurrency_capacity =
    max_concurrency_rollout_sample - in_flight_samples  # infinite when unset

admit one row iff min(staleness_capacity, concurrency_capacity) >= G
```

All calls from one row receive the same submit version. Accepted samples remain
accounted for, and each publication adds one batch of staleness capacity.

`O=0` is the fully on-policy setting: every optimizer batch trains groups from
its current published policy version. It does not disable overlap inside that
batch: with `K>1`, the trainer can process an early chunk while remaining
rollouts for the same batch finish.

When `C` is set, it must be at least `G`. To keep a concurrency window full,
the admission budget must also cover it. A useful sizing check is:

```text
O >= ceil(C / B) - 1
```

This is a capacity heuristic, not a claim about the exact model version that
generated every token. `async/version_offset_*` measures submit-version lag.

## Completion-driven refill

The producer attempts refill at startup, after publication, and after every
retired rollout task. For a completion-triggered refill:

1. retire the completed task;
2. resolve its row when all `G` draws have settled;
3. emit a training chunk if its predetermined target is full;
4. immediately retry row admission;
5. submit as many complete rows as both budgets allow.

This path runs while the trainer worker is active. If the staleness gate is
full, the refill attempt submits nothing and waits for publication. If only the
concurrency gate is full, the next rollout completion reopens capacity.

## Batch lifecycle

For each optimizer batch:

1. The first ready chunk exposes the `OptimizerBatch` to the recipe.
2. `async for chunk in batch.chunks()` waits only for the next predetermined
   chunk. Later chunks can queue while the current chunk trains.
3. Each chunk runs reference/old-policy work and forward/backward.
4. After the final chunk, the recipe performs one optimizer step.
5. The recipe saves and hotloads sampler weights.
6. `coordinator.publish(batch)` advances the policy version, commits accepted
   rows to the durable cursor, records metrics, and wakes the producer.

The recipe performs an unconditional initial sampler sync and one sync per
optimizer batch. It has no `weight_sync_interval` or conditional initial-sync
knob.

When the source ends, the producer seals and trains a final partial optimizer
batch rather than silently dropping accepted prompt groups.

## Failure policy

One flaky rollout does not immediately abort a long run:

- `None` is an explicit dropped draw;
- timeouts, connection failures, HTTP 408/429/5xx, explicitly marked
  `RecoverableRolloutError`s, and known client transport errors are recoverable;
- recoverable errors drop that draw and continue if the row still satisfies
  `min_group_size`;
- the default circuit breaker aborts after 5 consecutive recoverable failures,
  or a failure rate of at least 50% after 10 observations in a 20-observation
  window;
- invalid return values, data-contract errors, non-retryable HTTP errors,
  unexpected cancellation, and unknown exceptions are fatal.

The classifier is intentionally narrow. Do not catch arbitrary exceptions in a
rollout and return `None`; mark known infrastructure failures explicitly or let
programming errors remain fatal.

If a producer failure affects the active optimizer batch, training stops before
its optimizer mutation. If it affects only a future batch, the recipe finishes
the current optimizer, hotload, and publication so trainer and sampler versions
do not diverge.

## Durability and resume

Rejected rows become durable when their draws settle. Accepted rows become
durable only after the optimizer step, hotload, and publication succeed. A
crash during an unfinished batch therefore does not silently advance the
dataset cursor past uncommitted training data.

`dcp_save_interval=0` disables resumable checkpoints. Set a positive interval
when resume is required. A bare checkpoint name resumes trainer state and the
dataset cursor for the same trainer. Checkpoint paths and URIs remain
weights-only inputs and reset the recipe cursor.

## Metrics and tuning

Static settings are stored once in the run configuration. Optimizer-batch
metrics use `rollout/step`; rate-limited producer gauges and cumulative refill
counters use the independent `producer/event` axis. Producer records are
emitted while training is in progress, so refill and admission behavior is not
reconstructed from one snapshot at publication.

Read [`async-rl-metrics.md`](async-rl-metrics.md) for the metric glossary,
invariants, W&B/JSONL reading procedure, refill proof, and tuning decision
table.

## Loss path

The async recipe has one client-side GRPO path; it does not expose a
`policy_loss` selector. `anchor_logp="old_policy"` snapshots trainer logprobs and
applies TIS against rollout behavior logprobs. `anchor_logp="rollout"` skips the
old-policy forward and makes the TIS ratio identity.

`TISConfig` controls correction and clipping. Reference KL is enabled when
`kl_beta > 0`. Raw inference-logprob drift metrics are observational and never
replace behavior logprobs in PPO or TIS.

## Examples and related references

- `training/examples/rl/single_turn_token_in/` — minimal token-in/token-out
  rollout
- `training/examples/rl/multi_turn_message_in/` — multi-turn message rollout
- `training/examples/rl/coding_agent/` — black-box coding-agent trajectory
- [`rl-hotload.md`](rl-hotload.md) — sampler weight transfer
- [`rl-sampling-timeouts.md`](rl-sampling-timeouts.md) — sampler timeout diagnosis
- [`rl-gradient-accumulation.md`](rl-gradient-accumulation.md) — optimizer gradient
  normalization
- [`rl-dynamic-filter.md`](rl-dynamic-filter.md) — prompt-group filtering
