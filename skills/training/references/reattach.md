# Re-attach a deployment to a new trainer

Use this page when the user wants to point an existing deployment at a new trainer's checkpoint bucket ("switch deployment to new trainer", "re-point hot-load bucket", "reattach deployment", "keep deployment warm across training runs").

When you delete a trainer job and start a fresh one, you can keep an existing inference deployment warm by re-pointing its hot-load bucket at the new trainer. This avoids 5-20 minutes of deployment cold start each time you iterate.

The cookbook exposes this via `training.utils.infra.setup_or_reattach_deployment()`. Under the hood it:

1. Captures the current pod identity
2. PATCHes `hot_load_trainer_job` on the deployment -- this triggers a rolling restart of the serving container
3. Waits for the new pod to come up with a different identity
4. Resets the `WeightSyncer` delta chain (new bucket has no prior snapshots)

**Don't reinvent this logic.** Use `setup_or_reattach_deployment()` from `training/utils/infra.py`.

---

## Step 1: Inspect the user's code and state

Before running anything, determine:

1. **Does a deployment already exist?**
   ```bash
   # Check the user's recipe Config for deployment_id
   grep -n "deployment_id" <user_recipe_or_config>
   ```

   Then query the deployment:
   ```python
   from fireworks.training.sdk import DeploymentManager
   dm = DeploymentManager(api_key=..., base_url=...)
   info = dm.get("<deployment_id>")
   # info.state: "READY", "CREATING", "FAILED", "DELETED", "DELETING", etc.
   ```

2. **Is there a new trainer job to re-attach to?**

   | User's state | Action |
   |--------------|--------|
   | Has new trainer's `job_name` already | Pass to `setup_or_reattach_deployment(trainer_job_name=...)` |
   | Trainer just finished creating | Use `endpoint.job_name` from `TrainerJobManager.create_and_wait()` |
   | Hasn't created new trainer yet | Create it first (`create_trainer_job()` from `utils/infra.py`) |

3. **Is the deployment currently hotloading?**

   **Do not re-attach mid-hotload.** Switching the bucket URL while weights are loading leaves the serving container in an undefined state. If a hotload is in progress, wait for it to finish:

   ```python
   status = dm.hotload_check_status(deployment_id, base_model)
   stage = status.get("replicas", [{}])[0].get("loading_state", {}).get("stage")
   # Wait until stage is "done" or "error" before re-attaching
   ```

4. **Is there a `WeightSyncer` instance that will hotload after the re-attach?**
   - Yes → pass it via `weight_syncer=` so its delta chain gets reset
   - No → omit the parameter; the re-attach still works

---

## Step 2: Run the re-attach

Use the cookbook helper:

```python
from training.utils.infra import setup_or_reattach_deployment
from training.utils.config import DeployConfig, InfraConfig

deploy_cfg = DeployConfig(
    deployment_id="my-deploy",
    # ... other fields from your original deployment creation
)

dep_info = setup_or_reattach_deployment(
    deploy_mgr=dm,
    deploy_cfg=deploy_cfg,
    base_model="accounts/fireworks/models/qwen3-8b",
    infra=InfraConfig(),
    trainer_job_name="accounts/<acct>/rlorTrainerJobs/new-trainer-v2",
    weight_syncer=syncer,       # optional but strongly recommended for RL
    reattach_settle_timeout_s=600,
)
```

Behavior:

- **If `deployment_id` is set and the deployment is live** (state not `FAILED`/`DELETED`/`DELETING`): PATCHes `hotLoadTrainerJob`, waits for the rolling restart, resets the syncer. Returns the existing `DeploymentInfo`.
- **Otherwise**: creates a fresh deployment pointing at the new trainer.

Both paths return `DeploymentInfo`.

---

## Step 3: Verify the new pod is serving

After `setup_or_reattach_deployment` returns, the new pod's hotload manager is up but the deployment may not have weights yet. Immediately follow with a base hotload from the new trainer:

```python
syncer.save_and_hotload("reattach-base", checkpoint_type="base")
```

Then proceed with training. Subsequent checkpoints are `delta` (fast).

---

## When NOT to re-attach

| Situation | What to do instead |
|-----------|-------------------|
| Deployment is `FAILED` or `DELETED` | Create a new deployment (`setup_or_reattach_deployment` auto-handles this) |
| You want different infra (different shape, accelerator, region) | Delete the old deployment, create a new one -- re-attach can't change shape |
| You want the deployment to serve the *same* trainer's weights | Just hotload; no re-attach needed |
| A hotload is currently in progress on the deployment | Wait for it to finish first |
| You're creating both trainer and deployment for the first time | Skip re-attach; use `setup_deployment()` directly |

---

## Raw SDK pattern (if not using the cookbook helper)

If the user has their own code that doesn't use `setup_or_reattach_deployment`, they still need to follow the same pattern. The raw SDK primitives:

```python
import time

# 1. Capture pod identity
status = dm.hotload_check_status(deployment_id, base_model)
replicas = status.get("replicas") or []
prev_identity = replicas[0].get("identity") if replicas else None

# 2. PATCH
dm.update(
    deployment_id,
    body={"hotLoadTrainerJob": new_trainer_job_name},
    update_mask="hot_load_trainer_job",
)

# 3. Two-phase wait (old pod gone -> new pod up with different identity)
deadline = time.time() + 600
while time.time() < deadline:
    status = dm.hotload_check_status(deployment_id, base_model)
    replicas = status.get("replicas") or []
    current = replicas[0].get("identity") if replicas else None
    if prev_identity is None:
        if current is not None:
            break
    elif current and current != prev_identity:
        break
    time.sleep(5)
else:
    raise TimeoutError("Re-attach did not produce a fresh pod")

# 4. Reset syncer delta chain
syncer.reset_delta_chain()
```

If you find yourself pasting this in user code, prefer pointing them at the cookbook helper.

---

## Common failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `TimeoutError: Re-attach did not produce a fresh pod` | New pod stuck in `Pending` (no GPUs, bad image) | Check `kubectl get pods -l fireworks.ai/deploymentId=<id>`; verify region capacity |
| Next hotload fails with `stage=error` | Skipped the rolling-restart wait; hit the old pod or the gap | Use `setup_or_reattach_deployment()` or include the two-phase wait |
| Serving container in bad state | Invoked re-attach mid-hotload | Wait for in-flight hotload to finish before PATCHing |
| Hotload after re-attach sends wrong delta | Forgot `weight_syncer.reset_delta_chain()` | Pass `weight_syncer=` to `setup_or_reattach_deployment` so it handles the reset |
| Deployment stays `CREATING` after PATCH | Misconfigured shape or region | `setup_or_reattach_deployment` only re-attaches live deployments; check state first |
