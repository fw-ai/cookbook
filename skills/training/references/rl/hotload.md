# RL: weight sync (hotload) during training

RL is the main consumer of hotload: the recipe saves sampler checkpoints mid-training and pushes them to the serving deployment so new rollouts come from the updated policy. SFT / DPO / ORPO don't typically hotload — they save once at the end and call it a day.

All hotload behaviour in `rl_loop.py` is controlled by `cfg.weight_sync: WeightSyncConfig`.

## Trainer-first vs deployment-first (two creation orders)

Mixing these is the most common cause of `Hotload failed` and `checkpoint not found in GCS`.

### Trainer-first (modern, required for new work)

1. Create the trainer via `TrainerJobManager.create_and_wait(...)`.
2. Create the deployment via `DeploymentManager.create_or_get(DeploymentConfig(hot_load_trainer_job=<trainer_job>, ...))`. The deployment copies the trainer's bucket URL at creation.
3. The **trainer owns** the checkpoints. Promote reads them from the trainer's bucket without any deployment ID.

Contract test: `training/tests/smoke_test/test_grpo_deepmath_trainer_first_smoke.py`.

### Deployment-first (legacy, deprecated — recognise only)

1. The deployment was created with its own `hotLoadBucketUrl`.
2. The trainer was launched with `hot_load_deployment_id=<deployment-id>` pointing at that deployment's bucket.
3. The **deployment owns** the bucket. Every `promote_checkpoint` call must pass the deployment ID as well.

New runs should not use this flow. It exists only so that existing deployment-first runs can still be promoted.

### How to tell which flow a run used

- Recipe `cfg.deployment.hot_load_trainer_job` set ⇒ trainer-first.
- Trainer launch args include `hot_load_deployment_id=<...>` ⇒ deployment-first.
- `checkpoints.jsonl` has a `source_job_id` matching the trainer and no deployment-owned bucket in the trainer's launch args ⇒ trainer-first.

## The knobs

| Field | Default | Meaning |
|---|---|---|
| `weight_sync_interval` | `1` | Sync every N optimizer steps. `1` = after every step (on-policy). `0` = no weight sync at all (rollouts always come from the initial policy — you almost never want this for RL). |
| `first_checkpoint_type` | `"base"` | First sampler save is a full snapshot; subsequent saves can be deltas. Do not change. |
| `weight_sync_timeout` | `600` | Per-hotload timeout in seconds. Bump if you see `Hotload did not complete within 600s` on large models. |
| `weight_sync_before_training` | `False` | Push an initial base snapshot before step 0. Useful when the deployment starts from a different snapshot than the trainer's base. |
| `dcp_save_interval` | `0` | DCP (optimizer + weights) save cadence for **resume**. Orthogonal to sampler hotload. `0` = off; no intermediate resume points. |
| `dcp_timeout` | `2700` | 45 min default for `save_state` / `load_state_with_optimizer`. |

## On-policy vs off-policy (weight-sync timing)

- `weight_sync_interval = 1` + strict 1:1 per step (the recipe default) → **on-policy**. Rollouts for step K+1 are sampled from the policy that step K produced.
- `weight_sync_interval > 1` → **off-policy** between syncs. Rollouts continue to come from an older snapshot until the next sync. CISPO / DRO / IS tolerate this better than vanilla GRPO.

## Base vs delta chain

For full-parameter training, the first sampler save is `base` (full weights, ~16 GB for 8B). Subsequent saves are `delta` (XOR diff, ~10× smaller). `WeightSyncer` manages this automatically — users don't pick per-step.

- LoRA always saves the full adapter regardless of `checkpoint_type` — every LoRA sampler checkpoint is promotable.
- Full-param `delta` saves are **not** promotable. Only `base` saves are. `CheckpointKind` enum + save helpers: `training/utils/checkpoint_utils.py`.

## `dcp_save_interval` for resume

Separate from hotload: DCP saves persist the full train state (weights + optimizer) so you can resume training if the job dies. `0` (default) = off — if your run crashes mid-training, there is no intermediate resume point. Set this if your run is long enough that a crash is painful.

## Two deployments, one trainer

On-policy sampler + held-out eval deployment is a common pattern. Both copy the trainer's `hotLoadBucketUrl` at creation, and both can be hotloaded from the same `WeightSyncer`.

```python
sampler = deploy_mgr.create_or_get(DeploymentConfig(
    deployment_id="my-sampler",
    base_model=...,
    hot_load_trainer_job=trainer_endpoint.job_name,
))
eval_dep = deploy_mgr.create_or_get(DeploymentConfig(
    deployment_id="my-eval",
    base_model=...,
    hot_load_trainer_job=trainer_endpoint.job_name,   # same trainer
    min_replica_count=0,                              # scale down when idle
))
```

This pattern is only possible on the trainer-first flow — deployment-first couples each run to one deployment.

## Self-check when hotload fails

Symptom: `Hotload did not complete within <N>s` or `Hotload failed for snapshot <id>` or `checkpoint "<name>" not found in GCS`.

1. **First, check the SDK version matches the cookbook's pin** (see [`../../SKILL.md#first-debug-step--always`](../../SKILL.md#first-debug-step--always)).
2. The most common cause is a trainer-first vs deployment-first mix-up. Ask:
   - Did you create this deployment before the trainer, and then later point a new trainer-first trainer at it?
   - Was the trainer originally launched with `hot_load_deployment_id`, and is something now also setting `hot_load_trainer_job` on the deployment?

   Either answer means the trainer's bucket and the deployment's bucket disagree, and the hotload cannot find the snapshot on the side that is asked to load it.
3. If neither applies, or you're unsure how to untangle it, reach out to Fireworks support. Server-side recovery (re-pointing a deployment's bucket, recovering an orphaned sampler blob, looking up a legacy deployment ID) is handled by the Fireworks team.

## Promoting a legacy deployment-first run

Symptom: `checkpoint "<name>" not found in GCS`. Cause: the promote API was looking at the trainer's own bucket; on deployment-first the blobs live in the deployment's bucket.

Users on a legacy run can try:

```bash
python training/examples/snippets/promote_checkpoint.py \
    --checkpoints-jsonl <path> \
    --hot-load-deployment-id <deployment-id>
```

If the deployment ID is unknown, or the promote still fails, reach out to Fireworks support.

## New runs don't use the legacy assumption

Agents sometimes copy old cookbook snippets that reference `hot_load_deployment_id` into a new run. Do not. New runs are trainer-first; `hot_load_deployment_id` only exists for recovering existing legacy checkpoints.

## See also

- `WeightSyncer` lifecycle: `fireworks.training.sdk.weight_syncer.WeightSyncer` (installed under `src/fireworks/training/sdk/weight_syncer.py`).
- `save_weights_for_sampler_ext`, `save_state`, `list_checkpoints`: `fireworks.training.sdk.client.FiretitanTrainingClient`.
- Trainer + deployment managers this flow depends on: `fireworks.training.sdk.trainer.TrainerJobManager` and `fireworks.training.sdk.deployment.DeploymentManager`.
