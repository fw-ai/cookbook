# Recipes — fork, don't reinvent

Each recipe is a single Python file in `training/recipes/` that wires the Training SDK. Copy, edit the `Config` at the top, run.

| Task | File |
|------|------|
| SFT | `training/recipes/sft_loop.py` |
| DPO | `training/recipes/dpo_loop.py` |
| ORPO | `training/recipes/orpo_loop.py` |
| **RL (primary)** — write a rollout function; recipe owns the loop. Async by default; strict on-policy scheduling via `max_head_offpolicy_versions=0` | `training/recipes/async_rl_loop.py` — see [`rl/async-rl.md`](rl/async-rl.md) |
| RL (simpler, synchronous GRPO scaffold) | `training/recipes/rl_loop.py` |
| Information Gain-based Policy Optimization (IGPO) | `training/recipes/igpo_loop.py` |
| Distillation / OPD / SDFT | `training/recipes/distillation_loop.py` — see [`distillation.md`](distillation.md) |

## "Reference loop" means these files

They are the canonical wiring of `FiretitanServiceClient` + `FiretitanTrainingClient` + `TrainingCheckpoints` + deployment sampler hotload. Do not rewrite — fork.

## What to fill in on `Config`

Always required on `Config` (with `trainer=TrainerConfig(...)`):

- `base_model` — `accounts/fireworks/models/<name>`
- `dataset` — path to JSONL
- `tokenizer_model` — HF model name
- `log_path` — directory for `dataloader.json` and logs
- `trainer.training_shape_id` — optional override; leave unset for auto-selection. Do not set manual `accelerator_type` / `node_count` (see [`shapes.md`](shapes.md))

RL-specific: for the primary `async_rl_loop.py`, you write a `rollout_fn` (typically a `rollout.py`) and a `train.py` that sets the `Config` (policy loss, reward wiring, deployment) and calls `main(cfg, rollout_fn_factory=..., rows=...)`; the recipe owns the loop. The simpler synchronous `rl_loop.py` takes a reward function, rollout batch sizes, and a deployment config directly. See [`rl/async-rl.md`](rl/async-rl.md).

Forward/backward metrics retain Tinker's reducer suffixes after SDK request
chunking. Runtime input-sharding telemetry therefore appears as
`train/dp_sharded_counts:min`, `train/dp_sharded_counts:max`, and
`train/local_input_sequences:sum`. Consumers that need proof of the executed
path must require min and max to agree, then use the summed rank-local input
count. Do not change these to `:last`: Tinker does not support that reducer for
chunked forward/backward results and silently drops those metrics.

Distillation-specific: use `distillation_loop.py` for OPD/SDFT. Open [`distillation.md`](distillation.md) before changing its config, dataset format, teacher routing, or top-K objective plumbing.

## Resume

Auto-resume is scoped to one trainer. Pin both runs to the same trainer via `cfg.trainer.job_id`, keep the same `log_path`, and rerun. `TrainingCheckpoints.resume()` lists the trainer's authoritative RPC rows, picks the newest resumable row, and restores its exact row cursor from the local `trainer job -> step -> cursor` KV mapping. Set `cfg.dataloader_cursor` to override that lookup explicitly. See [`checkpoints.md`](checkpoints.md) for the full contract.

## Init from another job

```python
config = Config(
    log_path="./new_run",
    init_from_checkpoint=checkpoint_row,  # row returned by list_checkpoints
    dataloader_cursor=128,  # optional explicit override; bypasses local lookup
    ...
)
```

You may also pass the row's full resource name or `"i44pvd4syzg8hjfk:step-4"`. Cross-job resume preserves the checkpoint step.

## RL specifics

RL has its own skill folder. Open [`rl/`](rl/) when working with `rl_loop.py`:

- [`rl/loss-paths.md`](rl/loss-paths.md) — server-side built-in vs client-side custom (and why one costs an extra forward)
- [`rl/gradient-accumulation.md`](rl/gradient-accumulation.md) — `optim_step` normalization; the trap custom losses fall into
- [`rl/dynamic-filter.md`](rl/dynamic-filter.md) — `should_accept`, why zero-variance groups get dropped
- [`rl/custom-loss.md`](rl/custom-loss.md) — interface + reference implementation + RL `Config` fields
- [`rl/hotload.md`](rl/hotload.md) — weight-sync cadence, `weight_sync_timeout`, on-policy vs off-policy, base/delta chain
- [`rl/concurrency.md`](rl/concurrency.md) — rollout concurrency control for the **sync** `rl_loop.py` (adaptive is the default)
- [`rl/sampling-timeouts.md`](rl/sampling-timeouts.md) — diagnose `DeploymentSamplerTimeoutError` from request shape and serving metrics
- [`rl/async-rl.md`](rl/async-rl.md) — `async_rl_loop.py` overlap recipe: sample-level cap, off-policy budget, pipeline chunks

SFT / DPO / ORPO users do not need these.
