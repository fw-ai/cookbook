# RL: gradient accumulation at `optim_step`

Every `forward_backward(...)` / `forward_backward_custom(...)` **accumulates gradients on the server**. Gradients only get applied when you call `optim_step(...)`. Multiple forward-backwards between optim_steps = gradient accumulation.

`rl_loop.py` collects and prepares a complete batch, then sends one logical
forward-backward. IGPO uses `prepare_batch_fn=igpo.prepare_batch(...)`: answer
scoring is batched while advantages remain normalized within each group.
Dedicated and serverless async RL submit currently ready complete groups
together, without a collection delay. All requests in `prompt_groups_per_step`
accumulate into exactly one `optim_step`, after every backward succeeds.

Do **not** use trainer-job `grad_accum`, `TrainerJobConfig.gradient_accumulation_steps`, or the old REST `gradientAccumulationSteps` field. Cookbook recipes and examples do not expose the server-side knob; SDK compatibility handling rejects values above `1`. Real gradient accumulation is only client-side control flow: call `forward_backward` N times, then call one `optim_step`.

## The normalization mode is the easily missed part

`optim_step` takes a `grad_accumulation_normalization` argument that tells the server how to normalize accumulated gradients before clipping and stepping:

| Value | Meaning | Use when |
|---|---|---|
| `NUM_LOSS_TOKENS` | Divide by total loss tokens (per-token mean) | Your loss returns a **raw sum**, including built-in cross-entropy and common RL/GRPO-style losses |
| `NUM_SEQUENCES` | Divide by total sequences (per-sequence mean) | You want per-trajectory weighting |
| `None` / `NONE` | No normalization | Your loss **already** returns a per-token or per-sequence mean |

## Defaults that matter

- The RL loops pass `objective.NORMALIZATION`; for GRPO that is `GradAccNormalization.NUM_LOSS_TOKENS`. The built-in GRPO loss is a raw sum, so token-mean normalization keeps the gradient, `train/grad_norm` and `grad_clip_norm` independent of how many tokens a step trains. There is no `Config` field.
- Every objective module in `training.utils.rl.algorithm` exports its mode as `NORMALIZATION` (GSPO: `NUM_SEQUENCES`; the others `NUM_LOSS_TOKENS`), so swapping the objective module also swaps the normalization.
- The SFT recipe passes `reduction="mean"` while constructing each datum, so
  every example's loss weights sum to 1 and `optim_step` needs no additional
  normalization.
- Raw-sum losses should pass `GradAccNormalization.NUM_LOSS_TOKENS`; pre-normalized losses should leave this as `None`. If the mode does not match what your loss returns, you either skip normalization entirely or double-normalize, both of which can break training in a way that's hard to spot from loss metrics alone.

## Rule of thumb for custom losses

- Loss returns `(...).sum()` → raw sum → use `NUM_LOSS_TOKENS`.
- Loss returns `(...).sum() / num_tokens` → already per-token mean → use `None`.

Match the mode to what your loss returns. Double-normalization is the most common custom-loss bug.
