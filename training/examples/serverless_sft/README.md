# Serverless SFT — support-ticket triage

Supervised fine-tuning on Fireworks **serverless training**, with the three
pieces a real SFT job needs beyond the training loop itself: **DCP checkpoints**,
**resume**, and **promotion** to a servable model.

If you have used [Tinker](https://tinker-docs.thinkingmachines.ai/), this will
feel familiar: one service object hands you a training client, and you write the
loop yourself.

## What "serverless" means here

On the dedicated path (`recipes/sft_loop.py`) the SDK provisions a **trainer
job** for your run and you manage its lifecycle.

On the **serverless** path there is **nothing to provision**. You connect to a
shared, already-running pooled trainer through the gateway and get back a
Tinker-compatible `FiretitanServiceClient`:

```python
service = FiretitanServiceClient(base_url=".../training/v1/serverless")
training_client = service.create_lora_training_client(base_model, rank)
for step in range(steps):
    # prompt tokens weighted 0, response tokens weighted 1
    training_client.forward_backward(datums, "cross_entropy").result()
    training_client.optim_step(adam).result()
    if step % dcp_save_interval == 0:
        training_client.save_state(f"triage-{step:04d}").result()
```

## The task

Given a customer support message, emit a strict JSON object:

```json
{"category": "billing", "severity": "high", "needs_human": true}
```

`category` ∈ `billing | bug | feature_request | account`, `severity` ∈
`low | medium | high`. Loss should fall steadily as the adapter learns both the
output format and the label mapping.

## The two kinds of checkpoint

This is the part worth internalizing — they are **not interchangeable**.

| | `save_state` (DCP) | `save_weights_for_sampler` |
| --- | --- | --- |
| Contains | weights **+ optimizer state** | weights only, in servable form |
| Use it to | **resume training** | **sample from / promote** |
| Pass it to | `create_training_client_from_state_with_optimizer`, `load_state` | `create_sampling_client(model_path=...)`, promote |

Passing a sampler path to `load_state` does not work, and vice versa. This
example saves both: DCP on an interval (and at the end) so training can resume,
and one sampler checkpoint at the end for promotion.

### Resuming

Serverless checkpoints are namespaced under the run that wrote them, so a bare
name only resolves inside the same session. The portable, cross-process form is
`<account>/<run-id>/<checkpoint>`, which the script prints when it finishes:

```
resume this run with:
  --resume-from fireworks/run-d4a0c70aac564b6e9ae70cfa795652bc/triage-0002
```

A resumed run derives `base_model` / `lora_rank` / `train_*` **from the
checkpoint itself**, so it cannot silently disagree with the run that wrote it.
The step counter and dataset cursor are restored from the checkpoint's step
suffix, so the resumed process does not re-train on rows it already saw.

### Promoting

The final sampler checkpoint is listed on the owning `TrainingSession` and
promoted into a first-class Fireworks model:

```
promoted model: accounts/<account>/models/serverless-sft-support-triage-c4e55d2e
```

The run id is appended so repeated runs do not collide (a 409). Promotion uses
the session-scoped control-plane endpoints, not the serverless training surface,
so it goes through a separate REST client — that split is handled for you.

## Files

| File | What it is |
| --- | --- |
| `support_triage_sft.py` | The whole example. A `Config` dataclass at the top; every field is also a CLI flag. |
| `data/support_triage_train.jsonl` | 72-row sample so the example runs out of the box. Point `--dataset` at your own JSONL for real training. |

Dataset rows are plain chat records — `{"messages": [{"role": "system"|"user"|"assistant", "content": ...}]}`
ending in the assistant turn that gets trained on.

## Run it

From `training/` (see the [top-level README](../../README.md) for install):

```bash
export FIREWORKS_API_KEY=fw_...          # or put it in training/.env
python -m examples.serverless_sft.support_triage_sft
```

Resume a previous run in a fresh process:

```bash
python -m examples.serverless_sft.support_triage_sft \
    --resume-from <account>/<run-id>/<checkpoint>
```

Useful flags: `--steps`, `--batch-size`, `--learning-rate`,
`--dcp-save-interval` (0 disables periodic saves), `--no-promote`, `--no-plot`.

### Kimi K3

[`run_kimi_k3.sh`](run_kimi_k3.sh) wraps the flags for a K3 run — LoRA rank 8,
6 steps of batch 4, a DCP checkpoint every 3 steps, and a timestamped output
model id so repeated runs don't collide:

```bash
./run_kimi_k3.sh                                   # train + checkpoint + promote
./run_kimi_k3.sh --steps 20 --batch-size 8         # extra flags are passed through
./run_kimi_k3.sh --resume-from fireworks/run-<32 hex>/triage-0006
```

Two K3-specific details the script handles for you:

- It exports `HF_TRUST_REMOTE_CODE=1`, since K3 ships a custom image processor
  that `transformers` will otherwise refuse to load non-interactively.
- It defaults `--tokenizer-model` to `moonshotai/Kimi-K3`, which resolves to the
  `kimi_k3` renderer. Overridable via the `BASE_MODEL` / `TOKENIZER_MODEL` env
  vars.

K3 is a **gated model**: your API key must belong to an account that can read
`accounts/fireworks/models/kimi-k3`, otherwise `create_model` fails with
`resource not found`. See the access note under *Notes / requirements*.

Metrics are written to `metrics.jsonl` and a `loss_curve.png` under a fresh
`/tmp/serverless_triage_sft_*` run directory (set `--run-dir` to pin it).

## Notes / requirements

- **Serverless pool capacity.** `create_lora_training_client` attaches to a
  pooled LoRA trainer for `base_model`. If no pooled trainer serves that model
  you get `no eligible shared trainer found for base model ...` — use a model
  that has a pool, or the dedicated recipe.
- **Model access.** The base model must be readable by your API key's account.
  A private model in another account fails with `create_model: resource not
  found`, and a key with access to *multiple* accounts is rejected outright
  (`create_session: account not found`) — use an account-scoped key.
- **LoRA only.** The serverless pool is LoRA-only (`lora_rank > 0`).
- **Set `max_seq_len` explicitly.** There is no training shape to infer it from.
  The example defaults to 32768 and rejects datums that would exceed it.
- **`base_model` / `tokenizer_model` must match.** The tokenizer and renderer
  format prompts client-side; a mismatch corrupts the training signal. Defaults
  target `kimi-k3` / `moonshotai/Kimi-K3`.
- **Checkpoint names stay short.** The promotable id is
  `{run_id}-{name}-{suffix}` against a 63-character limit, so names are capped
  at 17 characters (validated up front).
- **Cost.** Defaults are a real training run against a real pooled trainer. Drop
  `--steps` / `--batch-size`, or switch to a smaller `base_model`, for a cheap
  smoke run.

For the dedicated (provisioned trainer) SFT path, multi-epoch dataloading,
evaluation, and W&B integration, see [`recipes/sft_loop.py`](../../recipes/sft_loop.py).
For the serverless RL counterpart, see
[`examples/serverless_rl/`](../serverless_rl/README.md).
