# Tools

Three standalone scripts live in `training/examples/snippets/`. Each does one operation; none needs an active trainer.

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

Source: `training/examples/snippets/promote_checkpoint.py`. Reads `sampler_path` / `source_job_id` / `base_model` from the jsonl row. For the legacy deployment-first case, see [`../../debug/references/trainer-first-vs-deployment-first.md`](../../debug/references/trainer-first-vs-deployment-first.md).

`output_model_id` is validated server-side at 63 chars — validate client-side too:

```python
from fireworks.training.sdk import validate_output_model_id
errors = validate_output_model_id(output_model_id)
```

## `reconnect_and_adjust_lr.py`

Re-attach a deployment to a new trainer job and optionally adjust the learning rate for the next run. Wraps `setup_or_reattach_deployment` from `training/utils/infra.py`.

```bash
python training/examples/snippets/reconnect_and_adjust_lr.py \
    --trainer-job <new-job> \
    --deployment-id <existing-deployment> \
    ...
```

After a re-attach, always call `syncer.reset_delta_chain()` before the next `save_and_hotload` — otherwise the next delta references a base that is not in the new bucket. See [`../../debug/references/reattach.md`](../../debug/references/reattach.md).

## `verify_logprobs.py`

Sanity-check numerical alignment between training-time logprobs and inference-time logprobs for a given checkpoint. Use when reward models or evaluation results look suspicious.

```bash
python training/examples/snippets/verify_logprobs.py \
    --deployment <deployment-id> \
    --checkpoint <checkpoint> \
    ...
```

Source: `training/examples/snippets/verify_logprobs.py`.
