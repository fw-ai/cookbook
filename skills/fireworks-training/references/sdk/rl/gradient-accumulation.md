# RL: gradient accumulation at `optim_step`

Every `forward_backward(...)` / `forward_backward_custom(...)` **accumulates gradients on the server**. Gradients only get applied when you call `optim_step(...)`. Multiple forward-backwards between optim_steps = gradient accumulation.

`rl_loop.py` runs a **strict 1:1 ratio** per step (one forward-backward → one optim_step, see `recipes/rl_loop.py:797` `ref_forward + fwd_bwd + optim_step + metrics (1:1)`). All `prompt_groups_per_step * completions_per_prompt` datums flow through a single call.

Do **not** use trainer-job `grad_accum`, `TrainerJobConfig.gradient_accumulation_steps`, or the old REST `gradientAccumulationSteps` field. Cookbook recipes and examples do not expose the server-side knob; SDK compatibility handling rejects values above `1`. Real gradient accumulation is only client-side control flow: call `forward_backward` N times, then call one `optim_step`.

## The normalization mode is the easily missed part

`optim_step` takes a `grad_accumulation_normalization` argument that tells the server how to normalize accumulated gradients before clipping and stepping:

| Value | Meaning | Use when |
|---|---|---|
| `NUM_LOSS_TOKENS` | Divide by total loss tokens (per-token mean) | Your loss returns a **raw sum**, common for RL/GRPO-style losses |
| `NUM_SEQUENCES` | Divide by total sequences (per-sequence mean) | You want per-trajectory weighting |
| `None` / `NONE` | No normalization | Your loss **already** returns a per-token or per-sequence mean — SFT / DPO / ORPO shapes |

## Defaults that matter

- Cookbook RL recipe defaults leave `grad_accumulation_normalization=None`, so `optim_step` performs **no server-side normalization** unless the recipe/user opts in.
- Raw-sum losses should pass `GradAccNormalization.NUM_LOSS_TOKENS`; pre-normalized losses should leave this as `None`. If the mode does not match what your loss returns, you either skip normalization entirely or double-normalize, both of which can break training in a way that's hard to spot from loss metrics alone.

## Rule of thumb for custom losses

- Loss returns `(...).sum()` → raw sum → use `NUM_LOSS_TOKENS`.
- Loss returns `(...).sum() / num_tokens` → already per-token mean → use `None`.

Match the mode to what your loss returns. Double-normalization is the most common custom-loss bug.
