# Serverless DPO — support-tone preference training

Direct Preference Optimization on Fireworks **serverless training**: teach a
model to prefer chosen responses over rejected ones, with **no reward model
and nothing to provision**. A port of Tinker's DPO recipe
([`tinker_cookbook/preference/train_dpo.py`](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/preference/dpo)) —
same loop, same loss, same metrics.

You connect to a shared, already-running pooled trainer and get back a
Tinker-compatible client. The frozen DPO reference model is just a sampler
bound to a snapshot of your step-0 weights:

```python
service = FiretitanServiceClient(base_url=".../training/v1/serverless")
training_client = service.create_lora_training_client(base_model, rank)

# Reference = frozen step-0 snapshot (zero-init LoRA == base model).
ref_path = training_client.save_weights_for_sampler("dpo-ref").result().path
reference_client = service.create_sampling_client(model_path=ref_path)

for step in range(steps):
    # datums interleaved [chosen_0, rejected_0, chosen_1, rejected_1, ...]
    ref_logprobs = [reference_client.compute_logprobs(seq) for seq in batch]
    training_client.forward_backward_custom(datums, dpo_loss_fn).result()
    training_client.optim_step(adam).result()
```

Tinker's `save_weights_and_get_sampling_client()` becomes
`save_weights_for_sampler` + `create_sampling_client` here; everything else
maps 1:1, and the loss is shared with the managed `recipes/dpo_loop.py`.

## The task

Given a customer support message, prefer the empathetic, actionable reply over
the dismissive or robotic one. Watch `dpo_loss` fall from ~0.693 while
`margin` and `accuracy` climb.

Two built-in checks answer "did it learn?":

- **Held-out margin** (`--eval-pairs`, default 4): preference margin on
  reserved rows, reported before vs after training.
- **Generation comparison** (`--no-compare` to skip): reference vs final
  checkpoint sampled side by side on two fresh prompts at the end of the run.
  The reference is the base model on a fresh run, the resumed weights on a
  resumed run. (Checkpoint sampling is session-scoped; the durable handoff is
  promotion.)

## Files

| File | What it is |
| --- | --- |
| `support_tone_dpo.py` | The whole example. A `Config` dataclass at the top; every field is also a CLI flag. |
| `data/support_tone_preference.jsonl` | 32-pair sample; point `--dataset` at your own JSONL for real training. |
| `run_kimi_k3.sh` | One-command wrapper (K3 tokenizer + `HF_TRUST_REMOTE_CODE` handled). |

Dataset rows are `{"chosen": {"messages": [...]}, "rejected": {"messages": [...]}}`
sharing one prompt; `samples`-style and OpenAI-style rows work too.

## Run it

```bash
export FIREWORKS_API_KEY=fw_...          # or put it in training/.env
python -m examples.serverless_dpo.support_tone_dpo      # from training/
./run_kimi_k3.sh                                       # or via the wrapper

# resume in a fresh process (the run prints the reference when it finishes):
python -m examples.serverless_dpo.support_tone_dpo \
    --resume-from <account>/<run-id>/<checkpoint>
```

On a *resumed* run the reference snapshot is taken from the resumed weights,
re-anchoring the reference — same as Tinker's resume path.

Useful flags: `--steps`, `--batch-size`, `--learning-rate`, `--dpo-beta`,
`--eval-pairs`, `--dcp-save-interval`, `--wandb-entity` (enables W&B),
`--no-promote`, `--no-compare`, `--no-plot`.

## Notes

- **Pool capacity.** `create_lora_training_client` attaches to a pooled LoRA
  trainer for `base_model`; if none serves it you get
  `no eligible shared trainer found for base model ...`.
- **Account-scoped keys.** Serverless training rejects keys with access to
  multiple accounts (`create_session: account not found`).
- **LoRA only**, and set `max_seq_len` explicitly — there is no training
  shape to infer it from.
- **`base_model` / `tokenizer_model` must match** — prompts are rendered
  client-side; a mismatch corrupts the training signal.
- **β in (0, 0.5).** Start at 0.1; higher keeps the policy closer to the
  reference.

For the dedicated (provisioned trainer) DPO path see
[`recipes/dpo_loop.py`](../../recipes/dpo_loop.py); for the serverless SFT/RL
counterparts see [`examples/serverless_sft/`](../serverless_sft/README.md) and
[`examples/serverless_rl/`](../serverless_rl/README.md).
