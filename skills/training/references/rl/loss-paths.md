# RL: server-side vs client-side loss

`training/recipes/rl_loop.py` dispatches to one of two loss paths per run, decided at step 0 from `cfg.policy_loss`.

| Path | SDK call | Cost | When |
|---|---|---|---|
| **Server-side built-in** (default) | `training_client.forward_backward(...)` | Single fused forward+backward on the trainer | `cfg.policy_loss` resolves to a built-in kernel via `training/utils/rl/losses.py:resolve_builtin_loss` — GRPO / DAPO / DRO / GSPO / CISPO — **and** the shape's `pipeline_parallelism == 1` |
| **Client-side custom** | `training_client.forward_backward_custom(...)` | **Extra forward** — client computes logprobs (forward #1), runs the Python loss, server runs forward+backward (#2) with gradients flowing from the Python loss | `cfg.policy_loss` has no built-in kernel (registered client-only), PP > 1, or a custom Python loss |

Default in stock `rl_loop.py` is **server-side built-in** — `cfg.policy_loss = "grpo"` resolves to the GRPO kernel. The recipe logs which path it picked at step 0; confirm from the boot logs.

## Cost

Switching to a custom loss roughly doubles the forward cost per step. If you can express the loss as a built-in variant (DAPO / CISPO / GSPO have tuning flags), stay server-side.

## Why you might fall back unintentionally

- `pipeline_parallelism > 1` on the chosen training shape → `resolve_builtin_loss` raises and the recipe falls back to client-side. If you want server-side performance, pick a shape with PP=1.
- GSPO with TP/CP is caught server-side only (the profile doesn't expose TP/CP). You can see the rejection in the trainer's startup logs if this happens.

## Related

- How to write a custom loss → [`custom-loss.md`](custom-loss.md)
- Why `optim_step` normalization matters → [`gradient-accumulation.md`](gradient-accumulation.md)
