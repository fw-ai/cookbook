# GPU quota accounting: why `used` disagrees with the dashboard

*Source of truth: [Quotas](https://docs.fireworks.ai/getting-started/quotas.md) · [On-demand deployments](https://docs.fireworks.ai/guides/ondemand-deployments.md) — defer to the live docs for current quota names and limits.*

`firectl quota list` reports more GPUs `used` than you can find in the UI. This
is the reference for reconciling the two, attributing GPUs to a person, and
finding capacity that is being held by nothing.

Three rules explain nearly every mismatch. None of them are visible from the
fine-tuning page alone.

## Rule 1 — an RL job is a trainer *plus* a rollout deployment

A Training API RL job provisions two billable resources, and **both charge the
same `training-<gpu>-count` quota**:

| Half | Where it appears | GPUs |
|---|---|---|
| Trainer | Fine-tuning page (`?type=rlor-trainer`), `firectl rlor-trainer-job list` | `trainerReplicaCount x trainingShape.acceleratorCount` |
| Rollout (sampler) deployment | **Deployments page**, `firectl deployment list` | `acceleratorCount x maxReplicaCount` |

The fine-tuning page lists the trainer side only. Counting jobs there and
multiplying by the trainer width **always undercounts** a running RL fleet — by
roughly 2x for a typical symmetric configuration.

The per-replica trainer width lives on the *training shape*, not on the job, so
a job whose dashboard entry reads "8 GPUs" can be far wider. Resolve it
explicitly:

```bash
firectl rlor-trainer-job get <JOB_ID> -o json -a <ACCOUNT_ID>   # trainerReplicaCount, trainingShapeVersion
firectl training-shape list -o json -a fireworks                # acceleratorType, acceleratorCount
```

Worked example — one job on a 4-GPU-per-replica B200 training shape with
`trainerReplicaCount=2` and a 4-replica rollout fleet of 2-GPU replicas:

```
trainer  2 replicas x 4 GPU =  8 B200
rollout  4 replicas x 2 GPU =  8 B200
                       job  = 16 B200   <- dashboard shows the 8 on the left
```

Four such jobs charge 64 B200 while the fine-tuning page adds up to 32.

## Rule 2 — quota is charged on the ceiling, not on running replicas

A deployment charges `acceleratorCount x maxReplicaCount` **even when zero
replicas are running** — cold, unschedulable (`RESOURCE_EXHAUSTED: no available
capacity`), or scaled down. Quota reserves the ceiling so an autoscale-up can
never fail on quota.

Consequences:

- `minReplicaCount: 0` alone does **not** release quota. Set **both** min and
  max to 0, or delete. The SDK's `DeploymentManager.scale_to_zero()` PATCHes
  both, so it does release quota; a hand-rolled `--min-replica-count 0` does not.
- A deployment stuck at 0 ready replicas because of capacity still consumes its
  full quota ceiling while serving nothing.

## Rule 3 — attachment decides which quota bucket a rollout lands in

| Deployment | Quota charged |
|---|---|
| Rollout attached to a live trainer (`hotLoadTrainerJob` set) | `training-<gpu>-count` |
| Everything else, including a **detached** rollout | `global--<gpu>-count` |

So "just ignore `global--<gpu>-count`, that's production inference" is wrong
advice during a leak hunt: a rollout fleet that lost its trainer moves *out* of
the training bucket and into the global one. Reconcile both.

## The leak: a rollout deployment that outlived its trainer

This is the usual source of GPUs that are impossible to find in the UI.

1. Deleting a trainer job **tombstones** it — the row moves to
   `JOB_STATE_DELETED` and `ListRlorTrainerJobs` filters it out, so it vanishes
   from the fine-tuning page immediately (see `rl-hotload.md`, "Trainer deletion
   and retention").
2. Deleting the trainer does **not** tear down the paired rollout deployment.
   It keeps running and keeps charging quota.
3. Cookbook rollout deployments are pinned to
   `min_replica_count == max_replica_count` (`training/utils/config.py`), so the
   orphan can **never** scale itself to zero. Idle auto-deletion applies to
   `min-0` deployments, which this is not.
4. The result is invisible on the fine-tuning page, immune to autoscaling, and
   billed by the GPU-second until someone deletes it by hand.

The recipes clean up on a *clean* exit only: `rl_loop` passes
`cleanup_deployment_on_close="scale_to_zero"` when `cleanup_on_exit` is set, and
`training/provision/provision.py` uses `"delete"`. Neither runs when the driver
is `SIGKILL`ed, the notebook is restarted, CI times out, or the trainer is
deleted from the dashboard or CLI instead of through the client. Both cleanup
paths also only *warn* on failure. Assume nothing was released and verify.

An orphan created by an older SDK carries neither the
`fireworks-training-sdk/managed-rollout` annotation nor a `<trainer-id>-rollout`
name, so it is easy to mistake for a production inference deployment. Its
**deployment shape** is the tell: rollout fleets run on the deployment shape
named by a training shape's `deploymentShapeVersion`.

## Audit procedure

`training/examples/tools/audit_gpu_usage.py` does all of this in one read-only
pass — it enumerates trainers, deployments, and quotas, charges each resource by
the rules above, attributes it to an owner, and prints what fails to reconcile:

```bash
export FIREWORKS_API_KEY=...
python training/examples/tools/audit_gpu_usage.py --account-id <ACCOUNT_ID>
python training/examples/tools/audit_gpu_usage.py --account-id <ACCOUNT_ID> --orphans-only
```

By hand:

```bash
firectl quota list -a <ACCOUNT_ID>                       # reported usage per bucket
firectl rlor-trainer-job list -a <ACCOUNT_ID>            # trainer half (tombstones hidden)
firectl deployment list -a <ACCOUNT_ID>                  # rollout + inference halves
firectl deployment get <DEPLOYMENT_ID> -o json -a <ACCOUNT_ID>
#   -> acceleratorCount x maxReplicaCount is the quota charge
#   -> hotLoadTrainerJob empty on a rollout shape = orphan
```

Release an orphan's GPUs (confirm ownership first — this is someone's run):

```bash
firectl deployment update <DEPLOYMENT_ID> -a <ACCOUNT_ID> --min-replica-count 0 --max-replica-count 0
firectl deployment delete <DEPLOYMENT_ID> -a <ACCOUNT_ID>
```

## Attributing GPUs to a person

Deployments carry **no `createdBy` field**; trainer jobs do. Per-user
attribution therefore always routes through the trainer:

- Attached rollout → `hotLoadTrainerJob` → that job's `createdBy`.
- Detached rollout → the `<trainer-id>-rollout` naming convention → **direct GET
  on the trainer job**, which still resolves a tombstoned row for 30 days and
  returns the `createdBy` of whoever deleted it.
- Neither available → genuinely unattributable from the API. Report it as such
  rather than assuming it is production inference.

## When quota reports usage nothing explains

If reported usage exceeds the sum of every visible trainer and deployment, the
remainder is a **stale reservation** held control-plane side. It is not
releasable by the account owner — no resource exists to delete. Collect the
account, quota name, reported usage, the computed total, and the audit output,
then escalate through <https://support.fireworks.ai/>. Do not tell a customer to
hunt for a resource that is not there.

Reported usage *below* the computed total is normal for a few seconds after a
create; the quota row lags. Re-read before acting.

## Critical rules

- **Count both halves of an RL job.** Trainer GPUs alone are never the quota
  charge.
- **Quota follows `maxReplicaCount`, not running replicas.** Only min *and* max
  at 0, or deletion, releases it.
- **Deleting a trainer does not delete its rollout deployment** — and the
  tombstoned trainer stops being listed, so the orphan loses its only signpost.
- **Never dismiss `global--<gpu>-count`** when reconciling: detached rollouts
  land there.
- **Check the Deployments page, not just the fine-tuning page,** before
  concluding GPUs are missing or that a quota reading is a platform bug.
