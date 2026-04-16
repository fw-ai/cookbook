# Promote a training checkpoint

Use this page when the user wants to promote a training checkpoint to a deployable Fireworks model ("promote my checkpoint", "deploy these weights", "turn my trained weights into a model I can call").

The cookbook produces `checkpoints.jsonl` during training; `training/tools/promote_checkpoint.py` reads it and calls the Fireworks promotion API to create a permanent, deployable model. No temporary trainer is needed -- promotion is a metadata + file-copy operation that works even after the trainer job has been deleted.

**Before you run the tool, work out the user's flow.** There are two flows and the wrong invocation fails loudly.

---

## Step 1: Inspect the user's code and config

**Do this first, before asking the user anything.** Look for:

1. **Is there a `checkpoints.jsonl`?**
   ```bash
   # Check the user's log_path. Common locations:
   ls <log_path>/checkpoints.jsonl
   ls sft_logs/checkpoints.jsonl
   ls rl_logs/checkpoints.jsonl
   ```
   If missing: the recipe didn't save sampler checkpoints. Ask if training completed or if they used `CheckpointKind.BOTH` (default).

2. **Read the checkpoints.jsonl to identify the flow:**
   ```python
   import json
   with open("<log_path>/checkpoints.jsonl") as f:
       entries = [json.loads(line) for line in f if line.strip()]
   ```
   Check each entry for these fields:

   | Field | Means |
   |-------|-------|
   | `sampler_path` + `source_job_id` | **Trainer-first** (modern, cookbook ≥ 0.3.0). Checkpoint is owned by the trainer. |
   | `sampler_path` + `hot_load_deployment_id` present | **Deployment-first** (legacy). Checkpoint is associated with a deployment. |
   | `base_model` field present | Good -- `--model` flag not needed. |
   | `base_model` missing | User must pass `--model`. |

3. **Check the recipe the user ran** (`recipes/sft_loop.py`, `recipes/rl_loop.py`, etc.) and their `Config`:

   - If the recipe uses `save_checkpoint(... kind=CheckpointKind.BOTH)` → trainer-first.
   - If you see references to `hot_load_deployment_id` in their config or old `checkpoints.jsonl` → legacy deployment-first.

4. **Check recipe version / SDK version:**
   ```bash
   grep "fireworks-ai" pyproject.toml  # or requirements
   # >= 1.0.0a36 + cookbook >= 0.3.0 → trainer-first is the only flow
   # Anything older may have deployment-first checkpoints
   ```

---

## Step 2: If the flow is unclear, ask the user explicitly

Only ask if Step 1 was genuinely inconclusive. Ask the specific questions:

> I need to figure out the right promotion command. Two questions:
>
> 1. **Which flow?** Is your checkpoint from a **trainer-first** run (modern, checkpoint owned by the trainer job) or a **deployment-first** run (legacy, checkpoint associated with a deployment ID)? If you ran a cookbook recipe on SDK 1.0.0a36 or later, it's trainer-first. If you have a legacy run with a hot-load deployment, it's deployment-first.
>
> 2. **Deployment ID** (only needed for deployment-first): what's the `hot_load_deployment_id` the checkpoint was saved against?

Wait for the answer before running anything.

---

## Step 3: Run the promotion

### Trainer-first (modern -- the common case)

```bash
python -m training.tools.promote_checkpoint \
    --checkpoints-jsonl <log_path>/checkpoints.jsonl \
    --step <step_or_omit_for_latest> \
    --output-model-id <desired_model_id>
```

- `--step` omitted → promotes the latest checkpoint
- `--output-model-id` omitted → auto-generated (`<base>-promote-<ckpt>-<ts>`)
- `--model` usually not needed -- base model is auto-detected from checkpoint metadata

### Deployment-first (legacy)

```bash
python -m training.tools.promote_checkpoint \
    --checkpoints-jsonl <log_path>/checkpoints.jsonl \
    --step <step_or_omit> \
    --output-model-id <desired_model_id> \
    --hot-load-deployment-id <deployment_id>
```

The tool will log a deprecation warning. Consider migrating the user to trainer-first if they plan to do more runs.

---

## Step 4: Verify the result

The tool logs the promoted model's name, state, and kind. Follow-up:

```bash
# Check the model exists and is in a usable state
firectl get model accounts/<account>/models/<output_model_id>
```

Expected states: `STATE_READY` (ready to deploy), `STATE_PENDING` (uploading). If `STATE_FAILED`, read the tool's output for the error.

---

## Common failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No checkpoint entries with sampler_path and source_job_id` | Recipe saved state-only checkpoints | Re-run with `CheckpointKind.BOTH` (the default) |
| `Step N not found` | Wrong step number | List available steps: `jq .step <log_path>/checkpoints.jsonl` |
| `--model is required` | `base_model` missing from checkpoint metadata | Pass `--model accounts/fireworks/models/<name>` explicitly |
| `403 forbidden` on `promote_checkpoint` call | API key lacks permissions or wrong account | Check `FIREWORKS_API_KEY` and account ownership of the trainer job |
| Deprecation warning about `hot_load_deployment_id` | Running legacy flow | Expected; tool still works. Migrate to trainer-first for future runs. |
| `source_job_id` refers to deleted trainer | Trainer job was cleaned up before promotion | **Still works** -- promotion is a metadata + file-copy op; the trainer does not need to be running |

---

## What NOT to do

- **Don't** spin up a new trainer job just to promote. The tool works without one.
- **Don't** hotload the checkpoint first. Promotion reads directly from the checkpoint bucket.
- **Don't** pass `--hot-load-deployment-id` on trainer-first runs. It will be ignored but adds noise.
- **Don't** guess the output model ID format -- use `--output-model-id` or let the tool auto-generate.
