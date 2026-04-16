# Trainer-first vs. deployment-first

Two creation orders exist. Mixing them is the most common cause of `Hotload failed` and `checkpoint not found in GCS`.

## Trainer-first (modern, required for new work)

1. Create the trainer via `TrainerJobManager.create_and_wait(...)`.
2. Create the deployment via `DeploymentManager.create_or_get(DeploymentConfig(hot_load_trainer_job=<trainer_job>, ...))`. The deployment copies the trainer's bucket URL at creation.
3. The **trainer owns** the checkpoints. Promote reads them from the trainer's bucket without any deployment ID.

Contract test: `training/tests/smoke_test/test_grpo_deepmath_trainer_first_smoke.py`.

## Deployment-first (legacy, deprecated)

1. Create the deployment first with its own `hotLoadBucketUrl`.
2. Create the trainer with `hot_load_deployment_id=<deployment-id>`, pointing at the deployment's bucket.
3. The **deployment owns** the bucket. Every `promote_checkpoint` call must pass the deployment ID as well:

   ```bash
   python training/examples/snippets/promote_checkpoint.py \
       --checkpoints-jsonl <path> \
       --hot-load-deployment-id <deployment-id>
   ```

Contract test: `training/tests/smoke_test/test_grpo_deepmath_deployment_first_smoke.py`.

## How to tell which flow a run used

- Look at the recipe's `cfg.deployment.hot_load_trainer_job` — set ⇒ trainer-first.
- Look at how the trainer was launched — if the startup args include `hot_load_deployment_id=<...>`, it's deployment-first.
- If `checkpoints.jsonl` lists a `source_job_id` that matches the trainer and there is no deployment-owned bucket in the trainer's launch args, it's trainer-first.

## Common mistakes

### Mixing in one run

Picking one flow per run is non-negotiable. Do not launch a trainer with `hot_load_deployment_id` and then try to create a deployment with `hot_load_trainer_job` pointing at that same trainer — the buckets disagree and hotload will fail.

### Promoting a legacy run without `--hot-load-deployment-id`

Symptom: `checkpoint "<name>" not found in GCS`. Cause: the promote API was looking at the trainer's own bucket; on deployment-first the blobs live in the deployment's bucket.

Fix: pass `--hot-load-deployment-id <deployment-id>`. If the ID is unknown, contact support — the server can look it up from the bucket.

### New run inherits the legacy assumption

Agents sometimes see old cookbook snippets that reference `hot_load_deployment_id` and carry that into a new run. Do not. New runs are trainer-first; `hot_load_deployment_id` is only for recovering existing legacy checkpoints.

## Diagnosing a `Hotload failed for snapshot <id>` error

The SDK's `wait_for_hotload` path raises this when the deployment replica reports a loading error. Check, in order:

1. `hotLoadBucketUrl` on the deployment matches the bucket where the trainer actually wrote the snapshot.
   - Grep the trainer log for `[save_weights_for_sampler] promote_ready snapshot_name=<id> base_model=<m> result_path=gs://...`.
   - If `result_path` is not under the deployment's `hotLoadBucketUrl`, this is a flow-mix — go back up this file.
2. `base_model` on the deployment matches the trainer's base model.
3. The snapshot actually exists (the `promote_ready` line above). If it is missing, the save failed upstream.

Recovery for the flow-mix case: re-attach the deployment to the trainer's bucket with `training/examples/snippets/reconnect_and_adjust_lr.py` and call `syncer.reset_delta_chain()` before the next `save_and_hotload`.
