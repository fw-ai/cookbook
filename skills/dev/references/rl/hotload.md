# RL: weight sync (hotload) during training

RL is the main consumer of hotload: the recipe saves sampler checkpoints mid-training and pushes them to the serving deployment so new rollouts come from the updated policy. SFT / DPO / ORPO don't typically hotload — they save once at the end and call it a day.

All hotload behaviour in `rl_loop.py` is controlled by `cfg.weight_sync: WeightSyncConfig`.

## Weight sync scope: PER_TRAINER vs PER_DEPLOYMENT

`DeployConfig.weight_sync_scope` (a `WeightSyncScope` enum) controls who owns the GCS bucket and what happens on resume.

Mixing these is the most common cause of `Hotload failed` and `checkpoint not found in GCS`.

### PER_TRAINER (default — required for new work)

```python
DeployConfig(weight_sync_scope=WeightSyncScope.PER_TRAINER, ...)
```

1. Create the trainer via `TrainerJobManager.create_and_wait(...)`.
2. Create the deployment via `DeploymentManager.create_or_get(DeploymentConfig(hot_load_trainer_job=<trainer_job>, ...))`. The deployment copies the trainer's bucket URL at creation.
3. The **trainer owns** the checkpoints. Promote reads them from the trainer's bucket without any deployment ID.
4. On resume, the deployment is re-attached to the new trainer (PATCH `hotLoadTrainerJob`), which briefly restarts the serving pod.

Contract test: `training/tests/smoke_test/test_grpo_deepmath_trainer_first_smoke.py`.

### PER_DEPLOYMENT (legacy, deprecated — recognise only)

```python
DeployConfig(weight_sync_scope=WeightSyncScope.PER_DEPLOYMENT, ...)
```

1. The deployment was created with its own `hotLoadBucketUrl`.
2. The trainer was launched with `hot_load_deployment_id=<deployment-id>` pointing at that deployment's bucket.
3. The **deployment owns** the bucket. Every `promote_checkpoint` call must pass the deployment ID as well.
4. On resume, a new trainer is created with the same `hot_load_deployment_id` — no pod restart required.

New runs should not use this scope. It exists only so that existing `PER_DEPLOYMENT` runs can still be promoted.

### How to tell which scope a run used

- `cfg.deployment.hot_load_trainer_job` set ⇒ `PER_TRAINER`.
- Trainer launch args include `hot_load_deployment_id=<...>` ⇒ `PER_DEPLOYMENT`.
- `checkpoints.jsonl` has a `source_job_id` matching the trainer and no deployment-owned bucket in the trainer's launch args ⇒ `PER_TRAINER`.

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

This pattern is only possible with `WeightSyncScope.PER_TRAINER` — `PER_DEPLOYMENT` couples each run to one deployment.

## Bucket mismatch: trainer wrote to a different bucket than the deployment watches

The trainer proactively logs this before hotload is even attempted:

```
[save_weights_for_sampler] Bucket mismatch — the deployment will not find this
snapshot and hotload will fail. Trainer wrote to gs://<trainer-bucket>/..., but
the deployment's hot_load_bucket_url is gs://<deployment-bucket>/...
```

Root cause: a `PER_TRAINER` and `PER_DEPLOYMENT` run got crossed — the deployment was created pointing at one bucket, and this trainer is writing to another.

Two recovery options. Pick whichever fits the situation:

1. **Re-attach the existing deployment to this trainer.** Useful when the deployment is already warmed up / serving traffic and you don't want to re-provision. The helper is `setup_or_reattach_deployment` in `training/utils/infra.py` — it PATCHes `hot_load_trainer_job` on the deployment, waits for the serving pod's rolling restart, and calls `WeightSyncer.reset_delta_chain()` for you. `training/recipes/rl_loop.py` uses it inline; look there for the call shape.

2. **Create a fresh deployment with `hot_load_trainer_job=<this trainer>`.** Simpler and safer when you don't need to preserve the existing deployment. The new deployment inherits the trainer's bucket at creation, so no mismatch.

For legacy `PER_DEPLOYMENT` runs whose promote then fails with `checkpoint "<name>" not found in GCS`, see [Promoting a legacy PER_DEPLOYMENT run](#promoting-a-legacy-per_deployment-run) below.

If neither option applies or you can't determine which scope the run used, reach out to Fireworks support — server-side state is sometimes needed to untangle.

## Self-check when hotload fails

Symptom: `Hotload did not complete within <N>s` or `Hotload failed for snapshot <id>` or `checkpoint "<name>" not found in GCS`.

1. **First, check the SDK version matches the cookbook's pin** (see [`../../SKILL.md#first-debug-step--always`](../../SKILL.md#first-debug-step--always)).
2. The most common cause is a `PER_TRAINER` vs `PER_DEPLOYMENT` scope mix-up. Ask:
   - Did you create this deployment before the trainer, and then later point a new `PER_TRAINER` trainer at it?
   - Was the trainer originally launched with `hot_load_deployment_id`, and is something now also setting `hot_load_trainer_job` on the deployment?

   Either answer means the trainer's bucket and the deployment's bucket disagree, and the hotload cannot find the snapshot on the side that is asked to load it.
3. If neither applies, or you're unsure how to untangle it, reach out to Fireworks support. Server-side recovery (re-pointing a deployment's bucket, recovering an orphaned sampler blob, looking up a legacy deployment ID) is handled by the Fireworks team.

## Promoting a legacy PER_DEPLOYMENT run

Symptom: `checkpoint "<name>" not found in GCS`. Cause: the promote API was looking at the trainer's own bucket; on `PER_DEPLOYMENT` runs the blobs live in the deployment's bucket.

Users on a legacy run can try:

```bash
python training/examples/tools/promote_checkpoint.py \
    --checkpoints-jsonl <path> \
    --hot-load-deployment-id <deployment-id>
```

If the deployment ID is unknown, or the promote still fails, reach out to Fireworks support.

## New runs use PER_TRAINER

Agents sometimes copy old cookbook snippets that reference `hot_load_deployment_id` into a new run. Do not. New runs use `WeightSyncScope.PER_TRAINER`; `hot_load_deployment_id` only exists for recovering existing legacy checkpoints.

## See also

- `WeightSyncer` lifecycle: `fireworks.training.sdk.weight_syncer.WeightSyncer` (installed under `src/fireworks/training/sdk/weight_syncer.py`).
- `save_weights_for_sampler_ext`, `save_state`, `list_checkpoints`: `fireworks.training.sdk.client.FiretitanTrainingClient`.
- Trainer + deployment managers this flow depends on: `fireworks.training.sdk.trainer.TrainerJobManager` and `fireworks.training.sdk.deployment.DeploymentManager`.
