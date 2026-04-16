---
name: fireworks-training-debug
description: Debug common Fireworks training mistakes. Covers mixing trainer-first with deployment-first, forgetting --hot-load-deployment-id on legacy promote, skipping training/deployment shapes, and the usual `checkpoint "<name>" not found in GCS` orphan. Points the agent at the exact recipe / tool / util file to inspect. For building a new run, use the sibling `dev` skill.
---

# Fireworks training — debug

Action-oriented. Each entry tells the agent what to check and which file to read. No tutorial.

---

## Mistake 1 — mixing trainer-first and deployment-first

**Symptom:** promote or hotload fails with errors like `checkpoint not found in GCS`, replica hotload error, or silently-stale checkpoints.

**Rule (modern / trainer-first):**
- Create the **trainer first**.
- Create the deployment **after**, passing `hot_load_trainer_job=<trainer>` so it inherits the trainer's bucket.
- The trainer owns the checkpoints.

**Legacy / deployment-first (deprecated):**
- Deployment was created first with its own bucket, then a trainer was attached via `hot_load_deployment_id=<deployment>`.
- The **deployment** owns the bucket. Every subsequent `promote_checkpoint` call must pass `--hot-load-deployment-id=<deployment-id>`.

**Diagnose:** look at the recipe's `cfg.deployment.hot_load_trainer_job` vs how the trainer job was created. If the trainer was started with `hot_load_deployment_id` pointing at a deployment, it is legacy. See `training/tests/smoke_test/test_grpo_deepmath_trainer_first_smoke.py` for the modern contract and `training/tests/smoke_test/test_grpo_deepmath_deployment_first_smoke.py` for the legacy one.

**Fix:** do not mix. Pick one flow per run. New code should be trainer-first; legacy runs must pass `--hot-load-deployment-id` on every promote.

---

## Mistake 2 — promoting a legacy run without `--hot-load-deployment-id`

**Symptom:** `promote_checkpoint` returns `checkpoint "<name>" not found in GCS`.

**Check:** is this a legacy deployment-first run (see Mistake 1)?
- Yes → re-run with `--hot-load-deployment-id=<deployment-id>`:
  ```bash
  python training/examples/snippets/promote_checkpoint.py \
      --checkpoints-jsonl <path> \
      --hot-load-deployment-id=<deployment-id>
  ```
  If the deployment ID is unknown, contact support — the server can look it up from the bucket.
- No → it is a trainer-first run; the checkpoint was orphaned. See Mistake 4.

Source: `training/examples/snippets/promote_checkpoint.py`.

---

## Mistake 3 — skipping training / deployment shapes

**Symptom:** manual `accelerator_type` / `node_count` / image tag mismatch with what the backend actually supports; deployment creation rejected; hotload incompatibility.

**Rule:** always set `cfg.infra.training_shape_id` and let the profile populate every other infra field.

```python
profile = rlor_mgr.resolve_training_profile(cfg.infra.training_shape_id)
# profile.deployment_shape_version is what cfg.deployment.deployment_shape needs
```

The recipes do this already — see `training/recipes/rl_loop.py` (search `resolve_training_profile`) and `training/recipes/sft_loop.py`. The `DeployConfig.to_deployment_config` in `training/utils/config.py` auto-clears manual accelerator fields when `deployment_shape` is set.

**Do not** set `deployment_accelerator_type`, `accelerator_count`, or `custom_image_tag` when a shape is in use — they will either be ignored or rejected.

---

## Mistake 4 — `checkpoint "<name>" not found in GCS` on a trainer-first run

**Symptom:** promote returns the error even though the list endpoint still shows the checkpoint.

**Root cause (typical):** an earlier `promote_checkpoint` for the same `checkpoint_id` was rejected by the server (usually `output_model_id` > 63 chars or illegal chars). The staged sampler blob was never anchored to a Model and GC reclaimed the bytes. The checkpoint record remains.

**Check:** list the trainer's checkpoints and pick a different promotable one:

```bash
curl "https://api.fireworks.ai/v1/accounts/<account>/rlorTrainerJobs/<job>/checkpoints?pageSize=200" \
  -H "Authorization: Bearer $FIREWORKS_API_KEY"
```

Pick the latest `createTime` with `promotable: true`. Step numbers alone mislead when a trainer inherits checkpoints from a predecessor via hotload.

**Fix:** retry `promote_checkpoint.py` against a different checkpoint, with a validated `output_model_id` (≤63 chars, `[a-z0-9-]`). Validate with:

```python
from fireworks.training.sdk import validate_output_model_id
errors = validate_output_model_id(output_model_id)
```

If no blob is promotable but a DCP row survives in `checkpoints.jsonl`, spin up a fresh trainer, `load_state_with_optimizer(path)`, `save_weights_for_sampler_ext(checkpoint_type="base")`, then promote the fresh blob.

---

## Mistake 5 — skipping `reset_delta_chain` after re-attach

**Symptom:** hotload fails right after a `setup_or_reattach_deployment` call because the next save goes out as a delta against a base that is not in the new bucket.

**Fix:** call `syncer.reset_delta_chain()` after every re-attach, before the next `save_and_hotload`. Source: `training/utils/infra.py` (see the docstring on `setup_or_reattach_deployment`) and `WeightSyncer` in the SDK.

---

## Triage cheat sheet

| Signal | Read |
|--------|------|
| `Hotload failed for snapshot ...` | SDK `deployment.py::wait_for_hotload`; grep trainer log for `promote_ready snapshot_name=…` — if the `result_path` is not under the deployment's `hotLoadBucketUrl`, this is Mistake 1 or Mistake 3 |
| `checkpoint "<name>" not found in GCS` | Mistake 2 if legacy; Mistake 4 if trainer-first |
| `HTTP 400` on promote with a model-id complaint | Mistake 4 root cause — trim `output_model_id` to ≤63 chars, retry |
| Deployment stuck in `CREATING`/`UPDATING` after re-attach | Wait for `READY`, then `reset_delta_chain()` (Mistake 5) |

---

For building a new run, see [`../dev/SKILL.md`](../dev/SKILL.md).
