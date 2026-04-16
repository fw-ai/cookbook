# Trainer-first vs. deployment-first

Two creation orders exist. Mixing them is the most common cause of `Hotload failed` and `checkpoint not found in GCS`.

## Trainer-first (modern, required for new work)

1. Create the trainer via `TrainerJobManager.create_and_wait(...)`.
2. Create the deployment via `DeploymentManager.create_or_get(DeploymentConfig(hot_load_trainer_job=<trainer_job>, ...))`. The deployment copies the trainer's bucket URL at creation.
3. The **trainer owns** the checkpoints. Promote reads them from the trainer's bucket without any deployment ID.

Contract test: `training/tests/smoke_test/test_grpo_deepmath_trainer_first_smoke.py`.

## Deployment-first (legacy, deprecated — recognise only)

1. The deployment was created with its own `hotLoadBucketUrl`.
2. The trainer was launched with `hot_load_deployment_id=<deployment-id>` pointing at that deployment's bucket.
3. The **deployment owns** the bucket. Every `promote_checkpoint` call must pass the deployment ID as well.

New runs should not use this flow. It exists only so that existing deployment-first runs can still be promoted.

## Two deployments per trainer (sampler + eval)

A common pattern once a run is trainer-first: one trainer writes checkpoints, and **multiple** deployments share that trainer's bucket. Typical setup is one deployment for on-policy sampling during training and a second for held-out evaluation.

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

Both deployments copy the trainer's `hotLoadBucketUrl` at creation, so every promote / hotload just works. This is only possible on the trainer-first flow — deployment-first couples each run to one deployment.

## How to tell which flow a run used

- Recipe `cfg.deployment.hot_load_trainer_job` set ⇒ trainer-first.
- Trainer launch args include `hot_load_deployment_id=<...>` ⇒ deployment-first.
- `checkpoints.jsonl` has a `source_job_id` matching the trainer and no deployment-owned bucket in the trainer's launch args ⇒ trainer-first.

## Self-check when something looks wrong

When a user hits `Hotload failed for snapshot ...` or `checkpoint "<name>" not found in GCS`, the first thing to rule out is a flow-mix. Ask them:

- Did you create this deployment before the trainer, and then later point a new trainer-first trainer at it?
- Was the trainer originally launched with `hot_load_deployment_id`, and is something now also setting `hot_load_trainer_job` on the deployment?

Either answer means the trainer's bucket and the deployment's bucket disagree, and the hotload cannot find the snapshot on the side that is asked to load it.

**If the user is confident they did not mix flows, or they are unsure how to untangle it, ask them to reach out to Fireworks support.** Recovery on the server side (re-pointing an existing deployment's bucket, recovering an orphaned sampler blob, or looking up the deployment ID for a legacy promote) is handled by the Fireworks team. This skill intentionally does not walk users through it — the steps require server-side state the user does not have.

## Promoting a legacy deployment-first run

Symptom: `checkpoint "<name>" not found in GCS`. Cause: the promote API was looking at the trainer's own bucket; on deployment-first the blobs live in the deployment's bucket.

Users on a legacy run can try:

```bash
python training/examples/snippets/promote_checkpoint.py \
    --checkpoints-jsonl <path> \
    --hot-load-deployment-id <deployment-id>
```

If the deployment ID is unknown, or the promote still fails, reach out to Fireworks support.

## New run inherits the legacy assumption

Agents sometimes copy old cookbook snippets that reference `hot_load_deployment_id` into a new run. Do not. New runs are trainer-first; `hot_load_deployment_id` only exists for recovering existing legacy checkpoints.

## See also

- Concept / API reference for the two managers: [`TrainerJobManager`](https://docs.fireworks.ai/fine-tuning/training-api/reference/trainer-job-manager) and [`DeploymentManager`](https://docs.fireworks.ai/fine-tuning/training-api/reference/deployment-manager).
- End-to-end walkthrough (how the trainer-first flow looks in real code): [Training and Sampling](https://docs.fireworks.ai/fine-tuning/training-api/training-and-sampling).
