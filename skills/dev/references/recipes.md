# Recipes — fork, don't reinvent

Each recipe is a single Python file in `training/recipes/` that wires the Training SDK. Copy, edit the `Config` at the top, run.

| Task | File |
|------|------|
| SFT | `training/recipes/sft_loop.py` |
| DPO | `training/recipes/dpo_loop.py` |
| ORPO | `training/recipes/orpo_loop.py` |
| **RL (primary)** — write a rollout function; recipe owns the loop. Async by default, sync via `synchronous_training=True` | `training/recipes/async_rl_loop.py` — see [`rl/async-rl.md`](rl/async-rl.md) |
| RL (simpler, synchronous GRPO scaffold) | `training/recipes/rl_loop.py` |
| Information Gain-based Policy Optimization (IGPO) | `training/recipes/igpo_loop.py` |

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

## Resume

Auto-resume is scoped to one trainer. Pin both runs to the same trainer via `cfg.trainer.job_id` (all recipes; the reference trainer is SDK-managed, so there is no separate reference job id to pin), keep the same `log_path`, and rerun. `TrainingCheckpoints.resume()` lists the trainer's checkpoints on the control plane, picks the newest resumable row, and restores the rollout cursor from `dataloader.json`. See [`checkpoints.md`](checkpoints.md) for the full priority order and constraints.

## Init from another job

```python
config = Config(
    log_path="./new_run",
    init_from_checkpoint="i44pvd4syzg8hjfk:step-4",  # job_id:checkpoint_name
    ...
)
```

Loads weights from the other job; resets step to 0.

## RL specifics

RL has its own skill folder. Open [`rl/`](rl/) when working with `rl_loop.py`:

- [`rl/loss-paths.md`](rl/loss-paths.md) — server-side built-in vs client-side custom (and why one costs an extra forward)
- [`rl/gradient-accumulation.md`](rl/gradient-accumulation.md) — `optim_step` normalization; the trap custom losses fall into
- [`rl/dynamic-filter.md`](rl/dynamic-filter.md) — `should_accept`, why zero-variance groups get dropped
- [`rl/custom-loss.md`](rl/custom-loss.md) — interface + reference implementation + RL `Config` fields
- [`rl/hotload.md`](rl/hotload.md) — weight-sync cadence, `weight_sync_timeout`, on-policy vs off-policy, base/delta chain
- [`rl/concurrency.md`](rl/concurrency.md) — rollout concurrency control for the **sync** `rl_loop.py` (adaptive is the default)
- [`rl/async-rl.md`](rl/async-rl.md) — `async_rl_loop.py` overlap recipe: sample-level cap, off-policy budget, PPO inner minibatches

SFT / DPO / ORPO users do not need these.
