---
name: training-cookbook-debug
description: Debug Fireworks Training Cookbook runs. Covers promoting a checkpoint to a deployable model (trainer-first vs legacy deployment-first, `checkpoint "<name>" not found in GCS`, `output_model_id` 63-char cap), re-attaching a deployment to a new trainer (`setup_or_reattach_deployment`), and the standalone tools under `training/tools/` (`promote_checkpoint.py`, `reconnect_and_adjust_lr.py`, `verify_logprobs.py`). Use this skill when a training run has already failed or when the user needs to promote, re-attach, or verify a checkpoint. For configuring and running recipes, use the sibling `dev` skill instead.
---

# Fireworks Training Cookbook — Debug

Recovery and operational flows for an existing training run. Use this skill when a run has failed mid-flight, a promote was rejected, a deployment drifted out of sync with its trainer, or the user needs to verify checkpoint numerics. For **building** a new run, switch to [`../dev/SKILL.md`](../dev/SKILL.md).

**Docs:** <https://docs.fireworks.ai/fine-tuning/training-api/saving-and-loading> — the troubleshooting and recovery sections mirror this skill. Prefer this skill for current-branch details.

---

## When to use this skill

- `promote_checkpoint` returned `checkpoint "<name>" not found in GCS`, a 400 on `output_model_id`, or a shape / charset complaint.
- Deployment is out of sync with the trainer after a restart, pod swap, or explicit re-attach.
- Need to verify training-time vs inference-time logprobs for a given checkpoint.
- Need to re-promote a legacy deployment-first checkpoint (passing `--hot-load-deployment-id`).
- Triaging a broken `checkpoints.jsonl` row.

**Not covered here:** writing a new recipe, picking a loss function, customizing rewards → see [`../dev/SKILL.md`](../dev/SKILL.md).

---

## Triage order

1. **Check `checkpoints.jsonl`.** Find the row for the step you want. Presence of `sampler_path` ⇒ promotable row; presence of `state_path` only ⇒ DCP state only (not promotable, but usable to re-export).
2. **Check trainer logs for `promote_ready`.** The trainer (firetitan) emits `[save_weights_for_sampler] promote_ready snapshot_name=<id> base_model=<m> result_path=gs://...` on every successful sampler save. This is the authoritative source for the promote coordinates.
3. **List checkpoints on the trainer** to disambiguate inherited vs current steps:
   ```bash
   curl "https://api.fireworks.ai/v1/accounts/<account>/rlorTrainerJobs/<job>/checkpoints?pageSize=200" \
     -H "Authorization: Bearer $FIREWORKS_API_KEY"
   ```
   Pick by latest `createTime` with `promotable: true`, not by step number (trainers created via hotload inherit their predecessor's checkpoints).
4. **Pick the right tool** — see the table below.

---

## Tool table

| Task | Tool | Reference |
|------|------|-----------|
| Promote a checkpoint to a deployable model | `training/tools/promote_checkpoint.py` | [`references/promote.md`](references/promote.md) |
| Re-attach a deployment to a new trainer | `training/tools/reconnect_and_adjust_lr.py` (wraps `setup_or_reattach_deployment`) | [`references/reattach.md`](references/reattach.md) |
| Verify logprobs across train/inference | `training/tools/verify_logprobs.py` | [`references/tools.md`](references/tools.md) |

All three are standalone scripts under `training/tools/`. They do not require an active trainer.

---

## Common failure modes

### `checkpoint "<name>" not found in GCS`

Server's checkpoint record exists but the underlying GCS blob was garbage-collected. Almost always caused by a prior `promote_checkpoint` call that the control plane rejected (usually for `output_model_id` length or charset). The staged sampler blob was never anchored to a Model, and GC reclaimed it.

**Fix:** pick a different checkpoint with `promotable: true` from the list endpoint, validate the new `output_model_id`, and retry. Details in [`references/promote.md`](references/promote.md#recovering-from-not-found-in-gcs).

### `output_model_id` rejected

Cap is **63 chars**, charset `[a-z0-9-]`. When deriving from a long policy / run name, trim the prefix before concatenating the step suffix. Example:

```python
# Bad: "<long-workflow>-<long-policy>-<long-base-model>-step-14"  # 70+ chars
# Good: "my-policy-step-14"
output_model_id = f"{short_policy}-step-{step}"[:63].rstrip("-")
```

### Legacy deployment-first promote

If the trainer was created with `hot_load_deployment_id` pointing at a deployment's own bucket (deprecated path), promote needs `--hot-load-deployment-id` too:

```bash
python training/tools/promote_checkpoint.py \
    --checkpoints-jsonl ./my_training/checkpoints.jsonl \
    --hot-load-deployment-id=<deployment-id>
```

New trainer-first runs do not need this flag. See [`references/promote.md`](references/promote.md) for the full decision tree.

### Deployment drifted from trainer

Use `setup_or_reattach_deployment(...)` from `training/utils/infra.py`, or the wrapper script `training/tools/reconnect_and_adjust_lr.py`. After a re-attach, call `syncer.reset_delta_chain()` before the next `save_and_hotload` — otherwise the delta references a base that isn't in the new bucket.

Details in [`references/reattach.md`](references/reattach.md).

---

## References

- [`references/promote.md`](references/promote.md) — promote decision tree, not-found recovery, legacy deployment-first flow
- [`references/reattach.md`](references/reattach.md) — when to re-attach, when not to, the rolling-restart gotcha
- [`references/tools.md`](references/tools.md) — `training/tools/*` scripts and when to reach for each

For recipe config, loss / reward customization, and run scaffolding, see [`../dev/SKILL.md`](../dev/SKILL.md).
