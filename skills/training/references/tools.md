# Tools

Four standalone scripts live in `training/examples/snippets/`. Each does one operation; none needs an active trainer.

## `promote_checkpoint.py`

Promote a sampler checkpoint to a deployable Fireworks model.

```bash
# Promote the latest promotable row
python training/examples/snippets/promote_checkpoint.py \
    --checkpoints-jsonl ./my_training/checkpoints.jsonl

# Promote a specific step
python training/examples/snippets/promote_checkpoint.py \
    --checkpoints-jsonl ./my_training/checkpoints.jsonl \
    --step 50

# Override the generated model id (always ≤63 chars, [a-z0-9-])
python training/examples/snippets/promote_checkpoint.py \
    --checkpoints-jsonl ./my_training/checkpoints.jsonl \
    --output-model-id my-policy-step-14

# Legacy deployment-first only — pass the deployment that owns the bucket
python training/examples/snippets/promote_checkpoint.py \
    --checkpoints-jsonl ./my_training/checkpoints.jsonl \
    --hot-load-deployment-id <deployment-id>
```

Source: `training/examples/snippets/promote_checkpoint.py`. Reads `sampler_path` / `source_job_id` / `base_model` from the jsonl row. For the legacy deployment-first case, see [`rl/hotload.md`](rl/hotload.md#promoting-a-legacy-deployment-first-run).

`output_model_id` is validated server-side at 63 chars — validate client-side too:

```python
from fireworks.training.sdk import validate_output_model_id
errors = validate_output_model_id(output_model_id)
```

## `reconnect_and_adjust_lr.py`

Reconnect a **training client** to an already-running RLOR trainer job and resume training with a new learning rate. Client-side only — the script calls `TrainerJobManager.reconnect_and_wait` and builds a fresh `ReconnectableClient`. It does **not** touch the deployment.

```bash
python training/examples/snippets/reconnect_and_adjust_lr.py \
    --job-id <policy-job-id> \
    --base-model accounts/fireworks/models/qwen3-8b \
    --new-lr 5e-6
```

Re-attaching a deployment to a new trainer (to fix a flow-mix bucket-URL drift) is a different operation and is not exposed as a user script. Users who suspect a flow-mix should self-check per [`rl/hotload.md`](rl/hotload.md#self-check-when-hotload-fails) and reach out to Fireworks support.

## `list_checkpoints.py`

List all checkpoints on a trainer job (authoritative server view, including any inherited from a predecessor via hotload). Use when `checkpoints.jsonl` is missing, stale, or you want to confirm which rows are actually promotable.

```bash
# All checkpoints
python training/examples/snippets/list_checkpoints.py --job-id <job-id>

# Only rows the server will accept for promote, newest first
python training/examples/snippets/list_checkpoints.py --job-id <job-id> --promotable-only

# Machine-readable
python training/examples/snippets/list_checkpoints.py --job-id <job-id> --json
```

Pick the **latest `createTime` with `promotable: true`** — step numbers mislead when a trainer inherits from a predecessor.

## `verify_logprobs.py`

Sanity-check numerical alignment between training-time logprobs and inference-time logprobs for a given checkpoint. Use when reward models or evaluation results look suspicious.

```bash
python training/examples/snippets/verify_logprobs.py \
    --deployment <deployment-id> \
    --checkpoint <checkpoint> \
    ...
```

Source: `training/examples/snippets/verify_logprobs.py`.
