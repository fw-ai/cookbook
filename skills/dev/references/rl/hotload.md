# RL: weight sync (hotload) during training

RL is the main consumer of hotload: the recipe saves sampler checkpoints mid-training and pushes them to the serving deployment so new rollouts come from the updated policy. SFT / DPO / ORPO don't typically hotload — they save once at the end and call it a day.

All hotload behaviour in `rl_loop.py` is controlled by `cfg.weight_sync: WeightSyncConfig`; the *scope* (who owns the GCS bucket) is set on `DeployConfig.weight_sync_scope`.

For the user-facing overview of the two scopes, see the docs page: [Hotload flows: trainer-first vs deployment-first](https://docs.fireworks.ai/fine-tuning/training-api/cookbook/hotload-flows). This skill is the deep reference — server-side validation details, knob tuning, recovery playbooks.

## Weight sync scope: PER_TRAINER vs PER_DEPLOYMENT

`DeployConfig.weight_sync_scope` (a `WeightSyncScope` enum) controls who owns the GCS bucket and what happens on resume. Mixing scopes on the same trainer ↔ deployment pair is the historical cause of `Hotload failed` and `checkpoint not found in GCS`. The **server now catches most mix-ups at `CreateDeployment` / `CreateRlorTrainerJob` time** (see [Server-side validation](#server-side-validation)), so runtime-level failures are rare in modern runs. They can still surface on trainers created with `--skip-validations` or from pre-validation runs.

### PER_TRAINER (default — required for new work)

```python
DeployConfig(weight_sync_scope=WeightSyncScope.PER_TRAINER, ...)
```

1. Create the trainer via `TrainerJobManager.create_and_wait(...)`.
2. Create the deployment via `DeploymentManager.create_or_get(DeploymentConfig(hot_load_trainer_job=<trainer_job>, ...))`. The deployment copies the trainer's bucket URL at creation.
3. The **trainer owns** the checkpoints. Promote reads them from the trainer's bucket without any deployment ID.
4. On resume, the deployment is re-attached to the new trainer (PATCH `hotLoadTrainerJob`), which briefly restarts the serving pod.

Bucket path: `gs://.../rl-checkpoints/{account}/trainer-{trainer_id}/`. Good when one trainer feeds multiple deployments, or for clean per-run isolation. Contract test: `training/tests/smoke_test/test_grpo_deepmath_trainer_first_smoke.py`.

### PER_DEPLOYMENT

```python
DeployConfig(weight_sync_scope=WeightSyncScope.PER_DEPLOYMENT, ...)
```

1. The deployment is created with its own `hot_load_bucket_url` (or auto-filled by the server from `deployment_id`).
2. Each trainer is launched with `hot_load_deployment_id=<deployment-id>` pointing at that deployment's bucket.
3. The **deployment owns** the bucket.

Bucket path: `gs://.../rl-checkpoints/{account}/{deployment_id}/`. Good when you want to re-use a warmed-up deployment across multiple sequential trainer runs without a deployment restart — the bucket URL is stable, each new trainer just writes to it. Trade-off: the deployment is keyed to one bucket, so you can't fan a single trainer out to multiple deployments.

### How to tell which scope a run used

- `cfg.deployment.weight_sync_scope` / `cfg.deployment.hot_load_trainer_job` set ⇒ `PER_TRAINER`.
- Trainer launch args include `hot_load_deployment_id=<...>` ⇒ `PER_DEPLOYMENT`.
- Neither ⇒ trainer and deployment are unrelated; no hotload.

The two scopes are mutually exclusive for the same trainer ↔ deployment pair.

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

## Two deployments, one trainer (PER_TRAINER only)

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

`PER_DEPLOYMENT` couples each run to one deployment, so this fan-out is a `PER_TRAINER`-only capability.

## Server-side validation

The control plane catches scope mix-ups at `CreateDeployment` / `CreateRlorTrainerJob` time and rejects the call with a message that names both resources and suggests the fix. The error links back to the user-facing docs page.

| Error | Root cause | Recovery |
|---|---|---|
| `hotload flow mismatch: trainer wants deployment-first (hot_load_deployment_id=X) but deployment Y is trainer-first (hot_load_trainer_job=Z)` | Trying to launch a `PER_DEPLOYMENT` trainer against a `PER_TRAINER` deployment. The two scopes are mutually exclusive. | Drop `hot_load_deployment_id` from the trainer, or drop `hot_load_trainer_job` from the deployment — pick one scope. |
| `hotload flow mismatch: trainer T is deployment-first-keyed for deployment D, but you're trying to use it as the source for deployment Y` | Pointing a new deployment's `hot_load_trainer_job` at a `PER_DEPLOYMENT` trainer whose bucket URL is keyed by a different deployment. | Use a `PER_TRAINER` trainer (no `hot_load_deployment_id`) as the source. |
| `hot_load_bucket_url %q conflicts with hot_load_trainer_job %s; set exactly one` | CreateDeployment got both fields. | Drop whichever field is wrong. `PER_TRAINER` deployments set `hot_load_trainer_job` only. |
| `hot_load_bucket_url %q conflicts with hot_load_trainer_job %s; update hot_load_trainer_job instead, or clear it first` | UpdateDeployment got both fields. | Clear `hot_load_bucket_url` first, then PATCH `hot_load_trainer_job`. |

These fire at the gateway, before any DB row is written. You won't see pods silently hotloading nothing for these classes.

The structural / reachability errors (`invalid FW_HOSTED hot_load_bucket_url`, `FW_HOSTED hot_load_bucket_url must use gs:// scheme`, `path must start with rl-checkpoints/{account}/`, `configured FW_HOSTED hot_load bucket is not reachable`, `control plane lacks permission to read the configured FW_HOSTED hot_load bucket`) are also rejected at the gateway. The last two don't echo the bucket name (server-managed; not caller-provided) — for those, reach out to Fireworks support; the other three are user URL issues and the error text tells you what to fix.

## Trainer deletion and retention

`DeleteRlorTrainerJob` no longer hard-deletes the trainer row immediately — it transitions the row to `JOB_STATE_DELETED` and marks `DeletionTime`. A background GC hard-deletes rows after **30 days**, aligned with the RL-checkpoints GCS bucket lifecycle.

During the 30-day retention window:
- `list_checkpoints` on the deleted trainer keeps working.
- `promote_checkpoint` keeps working — just pass `name` as usual.
- `ListRlorTrainerJobs` / `ListDpoJobs` filter out `JOB_STATE_DELETED` so customer-facing listings don't surface tombstones.
- A second `DeleteRlorTrainerJob` / `DeleteDpoJob` on a tombstoned row is a no-op; it does **not** reset the retention clock.
- `CancelRlorTrainerJob` / `CancelDpoJob` on a tombstoned row returns `cannot cancel job in state: JOB_STATE_DELETED` — expected, no action needed.

After retention, both `list_checkpoints` and `promote_checkpoint` return `NOT_FOUND`. By that point checkpoints in GCS are also gone, so the behaviour is consistent.

**Practical implication:** you can safely delete a trainer as soon as a run finishes. You have a month to promote checkpoints via the API.

## Promoting a checkpoint

One field — the `name` from the list response. Works identically for `PER_TRAINER` and `PER_DEPLOYMENT` runs, alive or within retention.

```python
entries = client.list_checkpoints(job_id=trainer_endpoint.job_name)
entry = max((e for e in entries if e["promotable"]), key=lambda e: e["createTime"])
client.promote_checkpoint(
    name=entry["name"],   # 4-segment resource name, pass verbatim
    output_model_id="my-promoted-model",
    base_model=base_model,
)
```

The server parses the trainer id out of `name` and reads `dbJob.HotLoadBucketUrl` as the single source of truth — which was populated at trainer-create time regardless of which scope was used. No `trainer_job_id`, no `checkpoint_id` split, no `hot_load_deployment_id` guess.

The old local-`checkpoints.jsonl` + `promote_checkpoint.py --hot-load-deployment-id` path is no longer needed: the API covers both scopes. It remains in the repo only for deployments that predate the stored-bucket-URL migration; new runs should not use it.

## Bucket mismatch: trainer wrote to a different bucket than the deployment watches

Caught proactively before hotload is even attempted:

```
[save_weights_for_sampler] Bucket mismatch — the deployment will not find this
snapshot and hotload will fail. Trainer wrote to gs://<trainer-bucket>/..., but
the deployment's hot_load_bucket_url is gs://<deployment-bucket>/...
```

Root cause: `PER_TRAINER` and `PER_DEPLOYMENT` got crossed — the deployment was created pointing at one bucket, and this trainer is writing to another. With server-side validation, this almost always means the trainer was created with `--skip-validations` or is from a run predating validation; the normal path would have been rejected at `CreateRlorTrainerJob` time.

Two recovery options:

1. **Re-attach the existing deployment to this trainer.** When the deployment is warmed up / serving traffic and you don't want to re-provision. `setup_infra` in `training/utils/infra.py` handles re-attach inline when an existing `deployment_id` is passed alongside a fresh `PER_TRAINER` trainer: it PATCHes `hot_load_trainer_job`, waits for the serving pod's rolling restart, and re-runs `WeightSyncer`'s one-time deployment-state check. `training/recipes/rl_loop.py` is the reference call shape.

2. **Create a fresh deployment with `hot_load_trainer_job=<this trainer>`.** Simpler and safer when you don't need to preserve the existing deployment. The new deployment inherits the trainer's bucket at creation so no mismatch.

If neither applies or you can't determine which scope the run used, reach out to Fireworks support — server-side state is sometimes needed to untangle.

## Self-check when hotload fails

Symptom: `Hotload did not complete within <N>s` or `Hotload failed for snapshot <id>` or `checkpoint "<name>" not found in GCS`.

1. **First, check the SDK version matches the cookbook's pin** (see [`../../SKILL.md#first-debug-step--always`](../../SKILL.md#first-debug-step--always)).
2. **Check if it's a retention-expired trainer.** `list_checkpoints` / `promote_checkpoint` returning `NOT_FOUND` > 30 days after delete is expected — the row is gone and the checkpoints in GCS have been GC'd too.
3. **If the trainer is alive or within retention:** most causes are a `PER_TRAINER` vs `PER_DEPLOYMENT` scope mix-up on a `--skip-validations` trainer or a pre-validation run. Ask:
   - Did you create this deployment before the trainer, and then later point a new `PER_TRAINER` trainer at it?
   - Was the trainer originally launched with `hot_load_deployment_id`, and is something now also setting `hot_load_trainer_job` on the deployment?

   Either answer means the trainer's bucket and the deployment's bucket disagree. Use one of the [bucket mismatch recovery](#bucket-mismatch-trainer-wrote-to-a-different-bucket-than-the-deployment-watches) options.
4. If neither applies, or you're unsure how to untangle it, reach out to Fireworks support. Server-side recovery (re-pointing a deployment's bucket, recovering an orphaned sampler blob, looking up a legacy deployment ID) is handled by the Fireworks team.

## New runs: don't blind-copy `hot_load_deployment_id`

Agents sometimes copy old cookbook snippets that reference `hot_load_deployment_id` into a new run. Only do so if you deliberately want `PER_DEPLOYMENT` (re-using a warmed deployment across sequential trainers). For a single run or a fan-out pattern, `PER_TRAINER` is the right default — set `hot_load_trainer_job` on the deployment and leave `hot_load_deployment_id` unset.

## See also

- [Hotload flows docs page](https://docs.fireworks.ai/fine-tuning/training-api/cookbook/hotload-flows) — user-facing overview (this skill is the deep reference)
- `WeightSyncer` lifecycle: `fireworks.training.sdk.weight_syncer.WeightSyncer` (installed under `src/fireworks/training/sdk/weight_syncer.py`).
- `save_weights_for_sampler_ext`, `save_state`, `list_checkpoints`: `fireworks.training.sdk.client.FiretitanTrainingClient`.
- Trainer + deployment managers this flow depends on: `fireworks.training.sdk.trainer.TrainerJobManager` and `fireworks.training.sdk.deployment.DeploymentManager`.
