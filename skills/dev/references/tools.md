# Tools

Four standalone scripts live in `training/examples/tools/`. Each does one operation; none needs an active trainer.

## `promote_checkpoint.py`

Promote a sampler checkpoint to a deployable Fireworks model.

```bash
# Promote the latest promotable row
python training/examples/tools/promote_checkpoint.py \
    --checkpoints-jsonl ./my_training/checkpoints.jsonl

# Promote a specific step
python training/examples/tools/promote_checkpoint.py \
    --checkpoints-jsonl ./my_training/checkpoints.jsonl \
    --step 50

# Override the generated model id (always ≤63 chars, [a-z0-9-])
python training/examples/tools/promote_checkpoint.py \
    --checkpoints-jsonl ./my_training/checkpoints.jsonl \
    --output-model-id my-policy-step-14

# Legacy deployment-first only — pass the deployment that owns the bucket
python training/examples/tools/promote_checkpoint.py \
    --checkpoints-jsonl ./my_training/checkpoints.jsonl \
    --hot-load-deployment-id <deployment-id>
```

Source: `training/examples/tools/promote_checkpoint.py`. Reads `sampler_path` / `source_job_id` / `base_model` from the jsonl row. For the legacy deployment-first case, see [`rl/hotload.md`](rl/hotload.md#promoting-a-legacy-deployment-first-run).

`output_model_id` is validated server-side at 63 chars — validate client-side too:

```python
from fireworks.training.sdk import validate_output_model_id
errors = validate_output_model_id(output_model_id)
```

## `reconnect_and_adjust_lr.py`

Reconnect a **training client** to an already-running RLOR trainer job and resume training with a new learning rate. Client-side only — the script calls `TrainerJobManager.reconnect_and_wait` and builds a fresh `ReconnectableClient`. It does **not** touch the deployment.

```bash
python training/examples/tools/reconnect_and_adjust_lr.py \
    --job-id <policy-job-id> \
    --base-model accounts/fireworks/models/qwen3-8b \
    --new-lr 5e-6
```

Re-attaching a deployment to a new trainer (to fix a flow-mix bucket-URL drift) is a different operation and is not exposed as a user script. Users who suspect a flow-mix should self-check per [`rl/hotload.md`](rl/hotload.md#self-check-when-hotload-fails) and reach out to Fireworks support.

## Listing checkpoints (`FireworksClient.list_checkpoints`)

Use this when `checkpoints.jsonl` is missing / stale, or you want to confirm which rows the server will actually accept for `promote_checkpoint`. Works for dead trainers (completed / failed / cancelled / deleted) — only the DB record and GCS blobs need to exist.

From Python:

```python
from fireworks.training.sdk import FireworksClient

client = FireworksClient(api_key=api_key)          # account auto-resolved
rows = client.list_checkpoints(job_id)              # auto-paginates
promotable = [r for r in rows if r.get("promotable")]
```

Each row is the raw JSON from the server: `name`, `createTime`, `updateTime`, `checkpointType` (an opaque server enum — there are several variants including `CHECKPOINT_TYPE_INFERENCE_BASE`, `CHECKPOINT_TYPE_INFERENCE_LORA`, `CHECKPOINT_TYPE_TRAINING_LORA`, `CHECKPOINT_TYPE_INFERENCE_ARC_V2`), and `promotable` (bool — authoritative; filter on this). Pick the **latest `createTime` with `promotable: true`** — step numbers mislead when a trainer inherits from a predecessor.

A thin CLI wrapper is available at `training/examples/tools/list_checkpoints.py` for quick terminal use:

```bash
python training/examples/tools/list_checkpoints.py --job-id <job-id>                     # table
python training/examples/tools/list_checkpoints.py --job-id <job-id> --promotable-only   # filter
python training/examples/tools/list_checkpoints.py --job-id <job-id> --json              # machine-readable
```

Requires `fireworks-ai[training] >= 1.0.0a62`. On older SDKs the method doesn't exist and the script will fail on import.

## `verify_logprobs.py`

Sanity-check numerical alignment between training-time logprobs and inference-time logprobs for a given checkpoint. Use when reward models or evaluation results look suspicious.

```bash
python training/examples/tools/verify_logprobs.py \
    --deployment <deployment-id> \
    --checkpoint <checkpoint> \
    ...
```

Source: `training/examples/tools/verify_logprobs.py`.
