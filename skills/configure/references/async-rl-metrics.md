# Reading async RL metrics

Use this procedure to verify producer liveness, rollout/train overlap, admission
gates, and pipeline sizing for `training/recipes/async_rl_loop.py`.

## Keep the two metric streams separate

The recipe deliberately has two observation clocks:

| Stream | Axis | Contents |
|---|---|---|
| Optimizer batch | `rollout/step` | Reward, loss, version lag, trainer waits, realized chunks, and sampled producer means |
| Rollout producer | `producer/event` | Live producer gauges and cumulative refill/admission counters |

Producer records are emitted at startup, shutdown, and at most once every 10
seconds while state changes. They are not synchronized to optimizer steps. Use
`producer/elapsed_time_s` for rates and `producer/published_version` to align a
producer record with training. Never join or deduplicate the two streams on the
generic JSONL `step` field.

Scheduling knobs and other static settings live in the run configuration, not
in every metric record. Read these first:

- `completions_per_prompt` (`G`)
- `prompt_groups_per_step` (`P`)
- `pipeline_chunks_per_step` (`K`)
- `max_head_offpolicy_versions` (`O`)
- `max_concurrency_rollout_sample` (`C`, optional)

One optimizer batch contains `B = P × G` samples. When `C` is unset there is no
concurrency gate, and `producer/concurrency_capacity_samples` is intentionally
absent. Do not interpret a missing concurrency-capacity series as zero.

## Read complete histories

For W&B, use `scan_history`; `history` may sample long runs. Scan once without
a `keys` filter because W&B treats a multi-key filter as requiring every key on
the same record, while optional metrics are intentionally absent:

```python
import wandb

run = wandb.Api().run("ENTITY/PROJECT/RUN_ID")
config = run.config
rows = list(run.scan_history())

producer_rows = [row for row in rows if "producer/event" in row]
train_rows = [
    row
    for row in rows
    if "rollout/step" in row and "rollout/raw_samples" in row
]
```

The payload-key check excludes the empty `rollout/step=0` row that W&B may
materialize when the custom step axis is initialized. It is not an optimizer
batch. Do not filter only on `rollout/step`.

For `metrics.jsonl`, parse every line and separate records by the presence of
`producer/event` or `rollout/step`. Preserve file order.

## Verify invariants before tuning

1. Confirm `producer/event` increases and
   `producer/published_version` eventually advances. A changing producer stream
   with a fixed published version means rollout work is happening but no
   optimizer batch is publishing.
2. Read `async/version_offset_*` as realized submit-version age, not as a
   second admission gate. `O` bounds how much accepted plus in-flight work the
   producer may own when it submits a row. A long-tail rollout can finish after
   newer batches publish, so its realized offset may exceed `O`. Investigate a
   rising tail, but do not classify `offset_max > O` alone as a violation.
3. Confirm `producer/in_flight_samples <= C` whenever `C` is configured.
4. Confirm producer totals are monotonic. Counter decreases usually mean two
   runs were concatenated or histories were sorted on the wrong axis.
5. Confirm `async/realized_training_chunks <= min(P, K)`. A final partial
   optimizer batch may realize fewer chunks.

## Read rollout filtering

The optimizer record exposes exactly five rollout-quality metrics:

| Metric | Meaning |
|---|---|
| `rollout/raw_reward` | Mean reward before `dynamic_filter_fn` |
| `rollout/filtered_reward` | Mean reward on samples sent to forward/backward |
| `rollout/raw_samples` | Successfully assembled samples before filtering |
| `rollout/filtered_samples` | Samples sent to forward/backward |
| `rollout/filter_ratio` | `1 - filtered_samples / raw_samples` |

Raw inference-logprob drift metrics are optional: when the sampler does not
return the distinct raw model-logprob field, the coverage and drift series are
absent rather than reported as a misleading constant zero. Other optional
metrics follow the same rule; an unsupported capability does not produce a
sentinel or constant-zero series.

Optimizer telemetry keeps `train/grad_norm` and `train/grad_norm_rms`.
`train/grad_norm_post_clip` appears only when clipping changes the norm. API
aggregation aliases and LoRA/pre-clip duplicates are intentionally omitted.

## Compare producer gauges with optimizer steps

`producer/in_flight_samples` and the capacity gauges use the independent
producer clock. The optimizer record contains time-weighted means from
rate-limited samples over the preceding publication window:

- `async/in_flight_samples_mean`
- `async/admission_capacity_samples_mean`
- `async/staleness_capacity_samples_mean`
- `async/concurrency_capacity_samples_mean` when `C` is configured

These are low-overhead diagnostics, not exact event-driven extrema. Do not copy
a point from the producer stream onto a train step; the clocks are not
synchronized.

## Prove completion-driven refill

Use deltas between consecutive producer records, not the absolute totals:

```text
attempts = Δ producer/completion_refill_attempts_total
rows     = Δ producer/completion_refill_rows_submitted_total
rate     = attempts / Δ producer/elapsed_time_s
```

- `attempts > 0` proves completed rollouts triggered admission retries.
- `rows > 0` proves those retries admitted new complete prompt rows.
- `attempts > 0` and `rows = 0` is valid when the staleness gate is full. A
  completion converts reserved samples to accepted samples but does not create
  version headroom. Publication creates that headroom.
- With a configured `C`, remaining source rows, and staleness headroom of at
  least `G`, a completion should free concurrency capacity and a subsequent
  retry should submit a row.

The scheduler intentionally does not instrument every producer transition with
trainer state. Therefore these records prove standalone refill but do not, by
themselves, identify the exact physical trainer call during which it occurred.
For a strict overlap acceptance test, block a train-chunk call, snapshot the
producer totals, release one rollout, and assert that the refill totals advance
before unblocking the trainer. The cookbook unit test uses this procedure.

## Diagnose the bottleneck

| Evidence | Interpretation | First action |
|---|---|---|
| High `perf/trainer_wait_for_sampler_time`; in-flight near `C` | Rollout-bound and using its configured window | Add rollout capacity or raise `C` if deployment capacity and the off-policy budget allow it |
| High trainer wait; in-flight below `C`; admission capacity below `G` | Admission gate, not deployment capacity, is limiting rollout | Inspect staleness capacity and `O`; do not add replicas first |
| High `perf/sampler_wait_for_trainer_time` | Producer exhausted version headroom while waiting for publication | Speed up trainer/hotload, or raise `O` only if more off-policy data is acceptable |
| High `perf/trainer_wait_for_chunk_time` | Later chunks arrive too slowly | Increase rollout capacity before increasing `K` |
| Near-zero chunk wait with a large `async/realized_training_chunks` | Trainer RPCs may be finer-grained than needed | Benchmark a smaller `K` to trade pipeline latency for fewer RPCs |
| Positive `async/in_flight_samples_mean` with nonzero train time | Rollouts and training occupied the same publication window | Use the strict blocked-trainer test when exact physical overlap must be proven |

`perf/trainer_idle_ratio` includes both the initial wait for a batch and waits
for later chunks, divided by scheduler-step wall time. It is therefore the
step-level idle fraction, not just the initial rollout wait.

## Admission-capacity check

The row-atomic gate admits one prompt row only when capacity is at least `G`:

```text
admission_capacity = min(staleness_capacity, concurrency_capacity)
```

Concurrency capacity is infinite when `C` is unset. A useful configuration
check is:

```text
O >= ceil(C / B) - 1
```

This ensures the version window is large enough to hold the requested
concurrency window. It does not guarantee high utilization: deployment
throughput, sequence lengths, source exhaustion, filtering, and failures still
matter.
