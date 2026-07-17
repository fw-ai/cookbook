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
| `countdown_rl.py` | The whole loop. A `Config` dataclass at the top; edit the `__main__` block and fork. |
| `countdown_rewards.py` | Vendored Countdown reward (`composite_reward`) — no external imports. |
| `data/countdown_train.jsonl` | 32-row sample so the example runs out of the box. Point `Config.dataset` at your own JSONL for real training. |

Dataset rows are `{"messages": [...], "ground_truth": {"numbers": [...], "target": N}}`.

## Run it

From `training/` (see the [top-level README](../../README.md) for install):

```bash
export FIREWORKS_API_KEY=fw_...          # or put it in training/.env
python -m examples.serverless_rl.countdown_rl
```

Point at a non-prod gateway with `FIREWORKS_BASE_URL` (the
`/training/v1/serverless` suffix is added for you):

```bash
export FIREWORKS_BASE_URL=https://dev.api.fireworks.ai
```

Metrics are written to `metrics.jsonl` and a `reward_curve.png` under a fresh
`/tmp/serverless_countdown_*` run directory (set `Config.run_dir` to pin it).

## Notes / requirements

- **Serverless pool capacity.** `create_lora_training_client` attaches to a
  pooled LoRA trainer for `base_model`. If the pool is full you'll get an
  out-of-capacity error — retry, or use the dedicated recipes.
- **LoRA only.** The serverless pool is LoRA-only (`lora_rank > 0`).
- **Set `max_seq_len` explicitly.** The example defaults to `65536` and rejects
  prompts or training datums that would exceed it. Lower it for a smaller
  context budget.
- **`base_model` / `tokenizer_model` must match.** The tokenizer renders prompts
  and decodes sampled tokens client-side; a mismatch corrupts rewards. Defaults
  target `qwen3p6-27b` / `Qwen/Qwen3.6-27B`.
- **Cost.** Defaults (`qwen3p6-27b`, 10 steps × 8 prompts × 8 samples) are a real
  training run. Drop `steps` / `group_size` / `max_sample_tokens`, or switch to a
  smaller `base_model`, for a cheaper smoke run.

For the dedicated (provisioned trainer + deployment) RL path and the full menu
of losses (GRPO, DAPO, GSPO, CISPO, …), see [`recipes/rl_loop.py`](../../recipes/rl_loop.py)
and [`recipes/async_rl_loop.py`](../../recipes/async_rl_loop.py).
