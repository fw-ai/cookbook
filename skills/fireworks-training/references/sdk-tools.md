# Tools

Five standalone scripts live in `training/examples/tools/`. Each does one operation. Most need no active trainer; `merge_lora_and_promote.py` provisions its own short-lived trainer.

Renderer work is part of the canonical training skill:

- [`renderer.md`](renderer.md) — implementing a new renderer.
- [`renderer-verification.md`](renderer-verification.md) — validating a renderer against the live gateway and the upstream HF chat template.

## `promote_checkpoint.py`

Promote a sampler checkpoint to a deployable Fireworks model. Queries the control plane directly via `list_checkpoints(job_id)` — no `checkpoints.jsonl` needed (and none is written; that legacy registry was removed when the cookbook moved to the CP-as-source-of-truth model).

```bash
# Promote the newest promotable checkpoint on a trainer job
python training/examples/tools/promote_checkpoint.py \
    --job-id <trainer-job-id> \
    --base-model accounts/fireworks/models/qwen3-8b

# Promote a specific checkpoint. `step-50` matches both an exact row
# and one stored as `step-50-<8-hex-session-suffix>`.
python training/examples/tools/promote_checkpoint.py \
    --job-id <trainer-job-id> \
    --checkpoint-name step-50 \
    --base-model accounts/fireworks/models/qwen3-8b

# Override the generated model id (always ≤63 chars, [a-z0-9-])
python training/examples/tools/promote_checkpoint.py \
    --job-id <trainer-job-id> \
    --base-model accounts/fireworks/models/qwen3-8b \
    --output-model-id my-policy-step-14

# Pre-migration PER_DEPLOYMENT only — pass the deployment that owns the bucket
python training/examples/tools/promote_checkpoint.py \
    --job-id <trainer-job-id> \
    --base-model accounts/fireworks/models/qwen3-8b \
    --hot-load-deployment-id <deployment-id>
```

Source: `training/examples/tools/promote_checkpoint.py`. Hands the row's 4-segment `name` (`accounts/<a>/rlorTrainerJobs/<j>/checkpoints/<c>`) verbatim to `TrainerJobManager.promote_checkpoint(name=...)` — the modern promote API. See the public docs at [`/fine-tuning/training-api/saving-and-loading`](https://docs.fireworks.ai/fine-tuning/training-api/saving-and-loading) for the full contract. The `--hot-load-deployment-id` flag is only needed for deployments that predate the stored-bucket-URL migration; passing it emits a `DeprecationWarning`.

`output_model_id` is validated server-side at 63 chars — validate client-side too:

```python
from fireworks.training.sdk import validate_output_model_id
errors = validate_output_model_id(output_model_id)
```

## `merge_lora_and_promote.py`

Turn an existing HF PEFT/LoRA adapter into a deployable full `HF_BASE_MODEL`. Provisions a short-lived service-mode LoRA trainer from the adapter's **base** model, explicitly loads the adapter with `load_adapter(<adapter gcs uri>)`, saves a merged-base sampler checkpoint with `save_weights_for_sampler(checkpoint_type="merged_base")`, then promotes and waits for `READY`.

For a promoted Fireworks LoRA model, pass the model resource and let the script resolve the rest:

```bash
python training/examples/tools/merge_lora_and_promote.py \
    --adapter-model accounts/<acct>/models/<lora-id> \
    --output-model-id my-merged-qwen3-8b
```

For a raw adapter directory, pass the adapter's own base model, directory, and rank:

```bash
python training/examples/tools/merge_lora_and_promote.py \
    --base-model accounts/fireworks/models/qwen3-8b \
    --adapter-gcs gs://my-bucket/adapters/my-lora \
    --lora-rank 8 \
    --output-model-id my-merged-qwen3-8b
```

Key points:
- **Do not** use `warmStartFrom` to merge a LoRA. RLOR `warmStartFrom` of a PEFT addon is not effective — the control plane downloads the adapter but the trainer session never loads it, so the save folds a zero-delta adapter and yields a base-identical checkpoint. The gateway rejects service-mode `warmStartFrom` of a LoRA addon. There is no shared base LoRA; every adapter is loaded explicitly.
- `merged_base` is LoRA-only: it folds `W <- W + scaling*(B@A)` into the base, strips adapter metadata, and the result promotes as `INFERENCE_BASE` / `HF_BASE_MODEL`. Saving from a fresh LoRA session (without `load_adapter`) exports base-identical weights.
- `--adapter-model` reads the base model and rank from `peftDetails` and recovers the adapter's `gs://` directory from the `getDownloadEndpoint` signed URL for `adapter_config.json`, which is otherwise not exposed by the API or `firectl`. Explicit `--base-model` / `--lora-rank` / `--adapter-gcs` override the resolved values.
- Leave `--region` and `--training-shape` unset. The trainer is provisioned from a validated shape the backend picks for the base model, in a region that carries that shape's accelerator. Pinning a region that does not provision the accelerator (e.g. a B300-only model family pinned to a region with no B300) fails during provisioning, and placement rejection can surface as a bare "Internal error".
- Trainer provisioning is lazy: the service client creates the trainer on the first `create_*_client` call, so `service.trainer_job_id` only resolves after that call and raises before it.

Source: `training/examples/tools/merge_lora_and_promote.py`.

## `reconnect_and_adjust_lr.py`

Reconnect a **training client** to an already-running RLOR trainer job and resume training with a new learning rate. Client-side only — the script calls `TrainerJobManager.reconnect_and_wait` and builds a fresh `ReconnectableClient`. It does **not** touch the deployment.

```bash
python training/examples/tools/reconnect_and_adjust_lr.py \
    --job-id <policy-job-id> \
    --base-model accounts/fireworks/models/qwen3-8b \
    --new-lr 5e-6
```

Re-attaching a deployment to a new trainer (to fix a flow-mix bucket-URL drift) is a different operation and is not exposed as a user script. Users who suspect a flow-mix should self-check per [`rl-hotload.md`](rl-hotload.md#self-check-when-hotload-fails) and reach out to Fireworks support.

## Listing checkpoints (`FireworksClient.list_checkpoints`)

The authoritative list of what the server has for a trainer. Works for dead trainers (completed / failed / cancelled / deleted) — only the DB record and GCS blobs need to exist. This is what `promote_checkpoint.py` calls under the hood; use it directly when you want to inspect or pick a row by hand.

From Python:

```python
from fireworks.training.sdk import FireworksClient

client = FireworksClient(api_key=api_key)          # account auto-resolved
rows = client.list_checkpoints(job_id)              # auto-paginates
promotable = [r for r in rows if r.get("promotable")]
```

Each row is the raw JSON from the server: `name`, `createTime`, `updateTime`, `checkpointType` (opaque server enum — treat as a string; several variants exist including `CHECKPOINT_TYPE_INFERENCE_BASE`, `CHECKPOINT_TYPE_INFERENCE_LORA`, `CHECKPOINT_TYPE_INFERENCE_ARC_V2`, `CHECKPOINT_TYPE_TRAINING`, `CHECKPOINT_TYPE_TRAINING_LORA`), and `promotable` (bool — authoritative; filter on this). The server returns rows **oldest-first**; the CLI wrapper re-sorts newest-first before printing. Pick the **latest `createTime` with `promotable: true`** — step numbers mislead when a trainer inherits from a predecessor.

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
