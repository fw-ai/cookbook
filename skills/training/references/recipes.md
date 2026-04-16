# Recipes — fork, don't reinvent

Each recipe is a single Python file in `training/recipes/` that wires the Training SDK. Copy, edit the `Config` at the top, run.

| Task | File |
|------|------|
| SFT | `training/recipes/sft_loop.py` |
| DPO | `training/recipes/dpo_loop.py` |
| ORPO | `training/recipes/orpo_loop.py` |
| Importance-weighted GRPO | `training/recipes/igpo_loop.py` |
| Generic RL loop (GRPO scaffold) | `training/recipes/rl_loop.py` |

## "Reference loop" means these files

They are the canonical wiring of `FiretitanTrainingClient` + `DeploymentManager` + `WeightSyncer`. Do not rewrite — fork.

## What to fill in on `Config`

Always required on `Config` + `InfraConfig`:

- `base_model` — `accounts/fireworks/models/<name>`
- `dataset` — path to JSONL
- `tokenizer_model` — HF model name
- `log_path` — directory for `checkpoints.jsonl` and logs
- `infra.training_shape_id` — **required**; do not set manual `accelerator_type` / `node_count` (see [`shapes.md`](shapes.md))

RL-specific (in `rl_loop.py`'s `Config`): reward function, rollout batch sizes, deployment config (shape is auto-filled from the profile).

## Resume

Rerun the same script with the same `log_path`. The recipe reads `checkpoints.jsonl`, picks up the last row with a `state_path`, restores DCP state, and continues. See [`checkpoints.md`](checkpoints.md) for the layout.

## Init from another job

```python
config = Config(
    log_path="./new_run",
    init_from_checkpoint="i44pvd4syzg8hjfk:step-4",  # job_id:checkpoint_name
    ...
)
```

Loads weights from the other job; resets step to 0.
