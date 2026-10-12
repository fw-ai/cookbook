# Serverless DPO — preference training on UltraFeedback

Direct Preference Optimization on Fireworks **serverless training**: teach a
model to prefer chosen responses over rejected ones, with **no reward model
and nothing to provision**.

The example is data prep plus one call: `prepare_data.py` writes the preference
pairs, and `train_dpo_serverless.py` builds a
[`recipes/dpo_loop.py`](../../recipes/dpo_loop.py) `Config` with
`serverless=True` and runs `dpo_loop.main(cfg)`.

```python
cfg = dpo_loop.Config(
    serverless=True,
    log_path="./serverless_dpo_logs/<run>",
    base_model="accounts/fireworks/models/kimi-k3",
    tokenizer_model="moonshotai/Kimi-K3",
    dataset="preference_train.jsonl",
    max_pairs=32,
    lora_rank=8,
    max_seq_len=32768,
    batch_size=4,          # preference pairs per optimizer step
    beta=0.1,
    dcp_save_interval=4,
    runner=RunnerConfig(reference_checkpoint_file="<log_path>/reference_checkpoint.json"),
)
dpo_loop.main(cfg)
```

On the serverless path the recipe attaches to a shared, already-running pooled
LoRA trainer and snapshots the policy before its first update. LoRA starts at
zero, so that snapshot **is** the base model; the recipe serves it as the frozen
DPO reference, scores every pair against it once, then trains with
`L = -log sigmoid(beta * [(pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)])`.

For a hand-written version of the same loop, step by step, see the
[`dpo_serverless` case study](../../case-studies/dpo_serverless/README.md).

## The data

`prepare_data.py` downloads
[`argilla/ultrafeedback-binarized-preferences`](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences)
and writes `preference_train.jsonl` in the chosen/rejected messages schema:

```bash
python -m examples.dpo.prepare_data                  # 512 pairs
python -m examples.dpo.prepare_data --max-rows 5000  # bigger run
```

Each row is `{"chosen": {"messages": [...]}, "rejected": {"messages": [...]}}`
sharing one prompt; point `--dataset` at any JSONL in that schema (or
`samples`-style / OpenAI-style preference rows) for your own data.

## Run it

```bash
export FIREWORKS_API_KEY=fw_...          # or put it in training/.env
./run_serverless.sh                         # from anywhere: data prep + train + promote
# or by hand, from training/:
python -m examples.dpo.prepare_data
python -m examples.dpo.train_dpo_serverless --output-model-id serverless-dpo-uf-demo
```

Flags: `--base-model`, `--tokenizer-model`, `--renderer-name`, `--dataset`,
`--max-pairs` (default 32 → 8 steps of 4 pairs), `--lora-rank`,
`--max-seq-len`, `--epochs`, `--batch-size`, `--learning-rate`, `--beta`,
`--dcp-save-interval`, `--log-path`, `--init-from-checkpoint`,
`--output-model-id` (omit to skip promotion), `--wandb-entity` (enables W&B).

[`run_serverless.sh`](run_serverless.sh) prepares the data on first run, exports
`HF_TRUST_REMOTE_CODE=1` for the K3 tokenizer, and promotes to a timestamped
model id. Extra flags are passed through.

## Did it learn?

Each step logs `train/dpo_loss` (starts near ln 2 ≈ 0.693 and should fall),
`train/margin` and `train/accuracy` (should climb). With `--wandb-entity` they
also go to W&B.

## Checkpoints and resume

DCP checkpoints (`step-<N>`, weights + optimizer) are saved every
`--dcp-save-interval` steps and at the end. The step and dataset cursor go to
`<log-path>/dataloader.json`, and the frozen reference snapshot id to
`<log-path>/reference_checkpoint.json`. At start the run logs its checkpoint
reference prefix, `<account>/run-<32 hex>/<checkpoint>`.

Continue in a fresh process with the **same `--log-path`**:

```bash
python -m examples.dpo.train_dpo_serverless \
    --log-path ./serverless_dpo_logs/<first run> \
    --init-from-checkpoint <account>/run-<32 hex>/step-4
```

The resumed run keeps scoring against the original frozen reference; the
recipe refuses to resume without it. Reference scores are recomputed from
that snapshot. Floating-point inference can vary across requests, so a
resumed loss need not be bit-identical even when the policy and cursor restore
correctly.

## Notes

- **Pool capacity.** If no pooled trainer is admitting you, creation fails
  with `no eligible shared trainer found for base model ...` (a transient
  capacity gap — retry). `base model ... is not available for serverless
  training` means the model is not on the serverless menu; use dedicated
  training (`examples/dpo/`).
- **Account-scoped keys.** Serverless training rejects keys with access to
  multiple accounts (`create_session: account not found`).
- **LoRA only**, and set `--max-seq-len` — there is no training shape to infer
  it from. Pairs longer than it are skipped.
- **`base_model` / `tokenizer_model` must match** — prompts are rendered
  client-side; a mismatch corrupts the training signal.
- **β in (0, 0.5).** Start at 0.1; higher keeps the policy closer to the
  reference.

For the dedicated (provisioned trainer) DPO example see
[`examples/dpo/`](../dpo/); for the serverless RL counterpart see
[`examples/rl/deepmath/`](../rl/deepmath/train_deepmath_serverless.py).
