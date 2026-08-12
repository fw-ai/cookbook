# Recipes — fork, don't reinvent

Each recipe is a single Python file in `training/recipes/` that wires the Training API SDK. Copy, edit the `Config` at the top, run.

| Task | File |
|------|------|
| SFT | `training/recipes/sft_loop.py` |
| DPO | `training/recipes/dpo_loop.py` |
| ORPO | `training/recipes/orpo_loop.py` |
| **RL (primary)** — write a rollout function; recipe owns the loop. Async by default; strict on-policy scheduling via `max_head_offpolicy_versions=0` | `training/recipes/async_rl_loop.py` — see [`rl-async.md`](rl-async.md) |
| Async RL on the shared serverless pool (experimental) | `training/recipes/experiment/async_rl_loop_serverless.py` — same rollout contract; serverless snapshot publication |
| RL (simpler, synchronous GRPO scaffold) | `training/recipes/rl_loop.py` |
| Information Gain-based Policy Optimization (IGPO) | `training/recipes/igpo_loop.py` |
| Distillation / OPD / SDFT | `training/recipes/distillation_loop.py` — see [`sdk-distillation.md`](sdk-distillation.md) |

## "Reference loop" means these files

They are the canonical wiring of `FiretitanServiceClient` + `FiretitanTrainingClient` + `TrainingCheckpoints` + deployment sampler hotload. Do not rewrite — fork.

## What to fill in on `Config`

Always required on `Config` (with `trainer=TrainerConfig(...)`):

- `base_model` — `accounts/fireworks/models/<name>`
- `dataset` — path to JSONL
- `tokenizer_model` — HF model name
- `log_path` — directory for `dataloader.json` and logs
- `trainer.training_shape_id` — optional override; leave unset for auto-selection. `accelerator_type` and `accelerator_count` are unsupported; do not set manual `node_count` (see [`sdk-shapes.md`](sdk-shapes.md))
- `trainer.use_reservation` — optional, default `True`. Tries reservation
  capacity first; set `False` to use shared trainer capacity. A
  full-parameter DPO reference trainer inherits the option and tries
  independently; an existing `trainer.job_id` is reused as-is.

RL-specific: for the primary `async_rl_loop.py`, you write a `rollout_fn` (typically a `rollout.py`) and a `train.py` that sets the `Config` (policy loss, reward wiring, deployment) and calls `main(cfg, rollout_fn_factory=..., rows=...)`; the recipe owns the loop. The simpler synchronous `rl_loop.py` takes a reward function, rollout batch sizes, and a deployment config directly. See [`rl-async.md`](rl-async.md).

Forward/backward metrics retain Tinker's reducer suffixes after SDK request
chunking. Runtime input-sharding telemetry therefore appears as
`train/dp_sharded_counts:min`, `train/dp_sharded_counts:max`, and
`train/local_input_sequences:sum`. Consumers that need proof of the executed
path must require min and max to agree, then use the summed rank-local input
count. Do not change these to `:last`: Tinker does not support that reducer for
chunked forward/backward results and silently drops those metrics.

Distillation-specific: use `distillation_loop.py` for OPD/SDFT. Open [`sdk-distillation.md`](sdk-distillation.md) before changing its config, dataset format, teacher routing, or top-K objective plumbing.

## Resume

Auto-resume is scoped to one trainer. Pin both runs to the same trainer via `cfg.trainer.job_id` (all recipes; the reference trainer is SDK-managed, so there is no separate reference job id to pin), keep the same `log_path`, and rerun. `TrainingCheckpoints.resume()` lists the trainer's checkpoints on the control plane, picks the newest resumable row, and restores the rollout cursor from `dataloader.json`. See [`sdk-checkpoints.md`](sdk-checkpoints.md) for the full priority order and constraints.

## Initialize from another job

```python
config = Config(
    log_path="./new_run",
    init_from_checkpoint="i44pvd4syzg8hjfk:step-4",  # job_id:checkpoint_name
    ...
)
```

Loads DCP weights and optimizer state from the other job, then starts the new
recipe at step/cursor 0. For a full continuation, reattach to the same trainer,
keep the same `log_path`, and use auto-resume or
`"<current_job_id>:<checkpoint>"`. See
[`sdk-checkpoints.md`](sdk-checkpoints.md#cross-run-resume) for the serverless
cross-run reference format.

## RL specifics

RL details stay in this skill. Open only the relevant reference when working with `rl_loop.py`:

- [`rl-loss-paths.md`](rl-loss-paths.md) — server-side built-in vs client-side custom (and why one costs an extra forward)
- [`rl-gradient-accumulation.md`](rl-gradient-accumulation.md) — `optim_step` normalization; the trap custom losses fall into
- [`rl-dynamic-filter.md`](rl-dynamic-filter.md) — `should_accept`, why zero-variance groups get dropped
- [`rl-custom-loss.md`](rl-custom-loss.md) — interface + reference implementation + RL `Config` fields
- [`rl-hotload.md`](rl-hotload.md) — strict sync hotload, `weight_sync_timeout`, on-policy vs off-policy, base/delta chain
- [`rl-concurrency.md`](rl-concurrency.md) — sync batch concurrency vs async sample-level admission
- [`rl-sampling-timeouts.md`](rl-sampling-timeouts.md) — diagnose `DeploymentSamplerTimeoutError` from request shape and serving metrics
- [`rl-async.md`](rl-async.md) — `async_rl_loop.py` overlap recipe: sample-level cap, off-policy budget, pipeline chunks

SFT / DPO / ORPO users do not need these.
