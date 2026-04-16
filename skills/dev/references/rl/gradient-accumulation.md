# RL: gradient accumulation at `optim_step`

Every `forward_backward(...)` / `forward_backward_custom(...)` **accumulates gradients on the server**. Gradients only get applied when you call `optim_step(...)`. Multiple forward-backwards between optim_steps = gradient accumulation.

`rl_loop.py` runs a **strict 1:1 ratio** per step (one forward-backward → one optim_step, see `recipes/rl_loop.py:797` `ref_forward + fwd_bwd + optim_step + metrics (1:1)`). All `prompt_groups_per_step * completions_per_prompt` datums flow through a single call.

## The normalization mode is the easily missed part

`optim_step` takes a `grad_accumulation_normalization` argument that tells the server how to normalize accumulated gradients before clipping and stepping:

| Value | Meaning | Use when |
|---|---|---|
| `NUM_LOSS_TOKENS` | Divide by total loss tokens (per-token mean) | Your loss returns a **raw sum** — the default for RL/GRPO |
| `NUM_SEQUENCES` | Divide by total sequences (per-sequence mean) | You want per-trajectory weighting |
| `None` / `NONE` | No normalization | Your loss **already** returns a per-token or per-sequence mean — SFT / DPO / ORPO shapes |

## Defaults that matter

- `rl_loop.Config.grad_accumulation_normalization = GradAccNormalization.NUM_LOSS_TOKENS` (the recipe passes it on every `optim_step` call — `recipes/rl_loop.py:807-810`).
- The raw SDK default is `optim_step(..., grad_accumulation_normalization=None)` — **no normalization**. If you fork the recipe and forget to pass this argument, you silently double-normalize (if your loss returned a mean) or skip normalization entirely (if your loss returned a raw sum). Either breaks training in a way that's hard to spot from loss metrics alone.

## Rule of thumb for custom losses

- Loss returns `(...).sum()` → raw sum → use `NUM_LOSS_TOKENS`.
- Loss returns `(...).sum() / num_tokens` → already per-token mean → use `None`.

Match the mode to what your loss returns. Double-normalization is the most common custom-loss bug.
