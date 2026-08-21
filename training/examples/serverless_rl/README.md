# Serverless RL — Countdown

A self-contained reinforcement-learning loop on Fireworks **serverless
training**, on the [Countdown](https://en.wikipedia.org/wiki/Countdown_(game_show)#Numbers_round)
numbers task. If you have used [Tinker](https://tinker-docs.thinkingmachines.ai/),
this will feel familiar: you get a training client and a sampling client from a
single service object and write the RL loop yourself.

## What "serverless" means here

On the dedicated path (`recipes/rl_loop.py`, `recipes/async_rl_loop.py`) the SDK
provisions a **trainer job** and an **inference deployment** for your run, and
you manage their lifecycle.

On the **serverless** path there is **nothing to provision**. You connect to a
shared, already-running pooled trainer through the gateway and get back a
Tinker-compatible `FiretitanServiceClient`. That one service gives you:

- `create_lora_training_client(base_model, rank)` → a training client, and
- `create_sampling_client(model_path=...)` → a sampler bound to a snapshot you
  just saved,

with no deployment to stand up or tear down. This is the fastest way to try
Fireworks training and the closest analogue to the Tinker workflow.

```
service = FiretitanServiceClient(base_url=".../training/v1/serverless")
training_client = service.create_lora_training_client(base_model, rank)
for step in range(steps):
    snapshot = training_client.save_weights_for_sampler(name).result().path
    sampler  = service.create_sampling_client(model_path=snapshot, tokenizer=...)
    #   sample a group of completions per prompt → score → group-relative
    #   advantages → importance-sampling training datums
    training_client.forward_backward(datums, "importance_sampling").result()
    training_client.optim_step(adam).result()
```

## The loop

Each optimizer step:

1. **Save** the current LoRA weights for the sampler (`save_weights_for_sampler`).
2. **Sample** `group_size` completions for each of `prompt_groups_per_step`
   Countdown prompts through a sampling client bound to that snapshot.
3. **Score** every completion with `composite_reward` (partial credit for a
   well-formed `<answer>`, using the right numbers, and hitting the target).
4. **Advantages**: standardize rewards within each prompt group (GRPO). Groups
   with no reward spread are dropped (zero signal).
5. **Train**: one `forward_backward(..., "importance_sampling")` + `optim_step`.

Reward should climb as the policy learns to produce valid Countdown equations.

## Files

| File | What it is |
| --- | --- |
| `countdown_rl.py` | The whole demo: dataset prep, the RL loop, metrics, W&B, reward plot. Model-agnostic CLI; defaults target Kimi K3. |
| `countdown_rewards.py` | Vendored Countdown reward (`composite_reward`) — no external imports. |
| `runs/run_countdown_k3_serverless.sh` | Ready-made launcher for Kimi K3 on Fireworks serverless training. |
| `data/countdown_train.jsonl` | 32-row sample for eyeballing the schema. Real runs use the prepared dataset below. |

## Dataset

The default dataset is the
[TinyZero Countdown tasks](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
(~490k rows of 3–4 number puzzles). Materialize it once (writes
`data/countdown_3to4_train.jsonl`):

```bash
python -m training.examples.serverless_rl.countdown_rl --prepare-dataset
```

Rows are `{"messages": [...], "ground_truth": {"numbers": [...], "target": N}}`.
The run script does this automatically on first run.

## Run it

The quickest path is the Kimi K3 launcher, which runs against
`api.fireworks.ai` with W&B tracking when configured:

```bash
export FIREWORKS_API_KEY=fw_...
export WANDB_ENTITY=<your-wandb-entity>   # optional
export WANDB_API_KEY=...                  # optional
bash training/examples/serverless_rl/runs/run_countdown_k3_serverless.sh
```

Or invoke the loop directly with any model (see the
[top-level README](../../README.md) for install first). The tokenizer and chat
renderer are loaded with the cookbook's model-aware helpers
(`training.utils.tokenizers.load_tokenizer` /
`training.utils.supervised.resolve_renderer_name`), so models the default Tinker
tokenizer/renderer tables don't cover (DeepSeek V4, Kimi K3) work out of the
box. The renderer is auto-resolved from the tokenizer; pass `--renderer-name`
only to override it.

```bash
export FIREWORKS_API_KEY=fw_...           # or put it in training/.env
export HF_TRUST_REMOTE_CODE=1             # Kimi K3 ships a custom tokenizer
python -m training.examples.serverless_rl.countdown_rl \
  --base-model accounts/fireworks/models/kimi-k3 \
  --tokenizer-model moonshotai/Kimi-K3 \
  --wandb-entity <your-wandb-entity>
```

Optionally override the API endpoint with `FIREWORKS_BASE_URL` (the
`/training/v1/serverless` suffix is added for you).

Per-step metrics (`rollout/raw_reward`, `rollout/filtered_reward`,
`rollout/filter_ratio`, `train/loss`, `perf/step_wall_time`) stream to
`metrics.jsonl` and to W&B; the closing `reward_curve.png` is also logged as a
W&B image. All artifacts live under the run directory (`--run-dir`, default
`/tmp/countdown-k3-*` via the launcher).

## Notes / requirements

- **Serverless pool capacity.** `create_lora_training_client` attaches to a
  pooled LoRA trainer for `base_model`. If the pool is full you'll get an
  out-of-capacity error — retry, or use the dedicated recipes.
- **LoRA only.** The serverless pool is LoRA-only (`lora_rank > 0`).
- **Set `max_seq_len` explicitly.** The example rejects prompts or training
  datums that would exceed it. Lower it for a smaller context budget.
- **`base_model` / `tokenizer_model` must match.** The tokenizer renders prompts
  and decodes sampled tokens client-side; a mismatch corrupts rewards. Defaults
  target `kimi-k3` / `moonshotai/Kimi-K3`.
- **Cost.** Defaults (Kimi K3, 20 steps × 16 prompts × 8 samples) are a real
  training run. Drop `--steps` / `--group-size` / `--max-sample-tokens` for a
  cheaper smoke run.

For the dedicated (provisioned trainer + deployment) RL path and the full menu
of losses (GRPO, DAPO, GSPO, CISPO, …), see [`recipes/rl_loop.py`](../../recipes/rl_loop.py)
and [`recipes/async_rl_loop.py`](../../recipes/async_rl_loop.py).

For the same rollout-function and async scheduling contract on the shared
serverless pool, see the experimental
[`recipes/experiment/async_rl_loop_serverless.py`](../../recipes/experiment/async_rl_loop_serverless.py).
