# Serverless SFT — support-ticket triage

Supervised fine-tuning on Fireworks **serverless training**, including the
parts a real SFT job needs beyond the loop itself: **DCP checkpoints**,
**resume**, and **promotion** to a servable model.

The example is a dataset plus one call: it builds a
[`recipes/sft_loop.py`](../../recipes/sft_loop.py) `Config` with
`serverless=True` and runs `sft_loop.main(cfg)`. The recipe owns batching,
checkpointing, resume, and promotion; the same loop runs dedicated trainers
when `serverless=False`.

```python
cfg = sft_loop.Config(
    serverless=True,
    log_path="./serverless_sft_logs/<run>",
    base_model="accounts/fireworks/models/kimi-k3",
    tokenizer_model="moonshotai/Kimi-K3",
    dataset="data/support_triage_train.jsonl",
    lora_rank=8,
    max_seq_len=32768,
    epochs=1,
    batch_size=8,
    dcp_save_interval=3,
    output_model_id="serverless-sft-triage-demo",
)
sft_loop.main(cfg)
```

## What "serverless" means here

On the dedicated path the SDK provisions a **trainer job** for your run. On the
**serverless** path there is **nothing to provision**: the recipe attaches to a
shared, already-running pooled LoRA trainer through
`{FIREWORKS_BASE_URL}/training/v1/serverless`
(`training.utils.serverless.setup_serverless_training`) and lists/promotes
checkpoints through the owning training session.

## The task

Given a customer support message, emit a strict JSON object:

```json
{"category": "billing", "severity": "high", "needs_human": true}
```

`category` ∈ `billing | bug | feature_request | account`, `severity` ∈
`low | medium | high`. Loss should fall as the adapter learns both the output
format and the label mapping.

## Run it

From `training/` (see the [top-level README](../../README.md) for install):

```bash
export FIREWORKS_API_KEY=fw_...          # or put it in training/.env
python -m examples.sft.train_sft_serverless \
    --output-model-id serverless-sft-triage-demo
```

Flags: `--base-model`, `--tokenizer-model`, `--renderer-name`, `--dataset`,
`--lora-rank`, `--max-seq-len`, `--epochs`, `--batch-size`,
`--learning-rate`, `--dcp-save-interval` (0 disables periodic saves),
`--log-path`, `--init-from-checkpoint`, `--output-model-id` (omit to skip
promotion), `--wandb-entity` (enables W&B).

### Kimi K3

[`run_serverless.sh`](run_serverless.sh) wraps the flags for a K3 run — LoRA rank 8,
one epoch of batch 8 (9 steps), a DCP checkpoint every 3 steps, and a
timestamped output model id so repeated runs don't collide:

```bash
./run_serverless.sh                                   # train + checkpoint + promote
./run_serverless.sh --epochs 2 --batch-size 4         # extra flags are passed through
```

It exports `HF_TRUST_REMOTE_CODE=1` (K3 ships a custom image processor) and
defaults `--tokenizer-model` to `moonshotai/Kimi-K3`, which resolves to the
`kimi_k3` renderer. Override with the `BASE_MODEL` / `TOKENIZER_MODEL` env vars.

## Checkpoints and resume

The recipe saves a DCP checkpoint (weights **and** optimizer state) named
`step-<N>` every `--dcp-save-interval` steps, and a final `step-<N>` that is
both resumable and promotable. Each DCP save also records the step and dataset
cursor in `<log-path>/dataloader.json`.

Serverless checkpoints are namespaced under the run that wrote them. At start
the run logs its reference prefix:

```
Serverless training session ts-..., run run-<32 hex>. To continue from one of
this run's checkpoints in a new process, pass
init_from_checkpoint='<account>/run-<32 hex>/<checkpoint>' with the same log_path.
```

Continue in a fresh process with the same `--log-path`, so the step and dataset
cursor are restored along with the trainer state:

```bash
python -m examples.sft.train_sft_serverless \
    --log-path ./serverless_sft_logs/<first run> \
    --init-from-checkpoint <account>/run-<32 hex>/step-6
```

With a different `--log-path` the weights and optimizer still load, but the
step counter and dataset cursor restart at 0.

## Promotion

With `--output-model-id`, the final checkpoint is promoted to
`accounts/<account>/models/<output-model-id>` through the training session's
checkpoint endpoints. Model ids must be unique, so pick a new id per run (the
K3 script timestamps it).

## Files

| File | What it is |
| --- | --- |
| `train_sft_serverless.py` | CLI → `sft_loop.Config(serverless=True)` → `sft_loop.main`. |
| `data/support_triage_train.jsonl` | 72-row sample so the example runs out of the box. Point `--dataset` at your own JSONL. |
| `run_serverless.sh` | Kimi K3 wrapper. |
| `train_sft.py`, `run.sh` | The same recipe on a dedicated trainer (`serverless=False`). Pass `--training-shape` to pick the trainer; extra `run.sh` flags override its defaults. |

Dataset rows are plain chat records —
`{"messages": [{"role": "system"|"user"|"assistant", "content": ...}]}` — and
every assistant turn is trained on.

## Notes / requirements

- **Serverless pool capacity.** The run attaches to a pooled LoRA trainer for
  `base_model`. If no pooled trainer serves that model you get
  `no eligible shared trainer found for base model ...` — use a model that has
  a pool, or the dedicated recipe.
- **Model access.** The base model must be readable by your API key's account,
  and serverless training rejects keys with access to *multiple* accounts
  (`create_session: account not found`) — use an account-scoped key.
- **LoRA only** (`--lora-rank > 0`), and **set `--max-seq-len`** — there is no
  training shape to infer it from. Longer rows are dropped by the recipe.
- **`base_model` / `tokenizer_model` must match.** Prompts are rendered
  client-side; a mismatch corrupts the training signal.
- **Cost.** Defaults are a real training run against a real pooled trainer.
  Lower `--epochs` / use `--dataset` with fewer rows for a cheap smoke run.

For the dedicated SFT path see [`train_sft.py`](train_sft.py) and [`run.sh`](run.sh); for the serverless
RL counterpart see [`examples/rl/deepmath/`](../rl/deepmath/train_deepmath_serverless.py).
