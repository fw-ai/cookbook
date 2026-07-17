# RL: server-side vs client-side loss

`training/recipes/rl_loop.py` and `training/recipes/async_rl_loop.py` dispatch
to one of two loss paths per run, decided explicitly by `cfg.loss_path`.

| Path | SDK call | Cost | When |
|---|---|---|---|
| **Server-side built-in** | `training_client.forward_backward(...)` | Single fused forward+backward on the trainer | Set `cfg.loss_path="builtin"` and use a policy loss registered in `training/utils/rl/builtin_losses.py` — GRPO / DAPO / DRO / GSPO / CISPO / importance_sampling. Requires `cfg.kl_beta == 0` |
| **Client-side custom** | `training_client.forward_backward_custom(...)` | Client evaluates the Python loss closure; this path is slower because it needs logprobs on the client side | Set `cfg.loss_path="client"` for KL penalties, client-only/custom losses, or when you want the conservative path |

Default in stock recipes is `loss_path="client"` for maximum compatibility.
Set `loss_path="builtin"` explicitly when you want the server-side fast path and
your config is eligible. The recipe logs the selected path at startup.

## Cost

Switching to a custom loss roughly doubles the forward cost per step. If you can express the loss as a built-in variant (DAPO / CISPO / GSPO have tuning flags), stay server-side.

## Eligibility checks

There is no silent fallback. `validate_loss_path(...)` raises before rollout if
the requested path is invalid.

- `cfg.policy_loss` not in `BUILTIN_LOSSES` with `loss_path="builtin"` raises.
- `cfg.kl_beta > 0` with `loss_path="builtin"` raises. Built-in datums carry
  `target_tokens` / `logprobs` / `advantages`, not `ref_logprobs`, so the KL term
  would otherwise be dropped.

## Rollout logprobs

`async_rl_loop.py` exposes `use_rollout_logprobs`, matching Slime's flag name.
Here `rollout_logprobs` means sampled-token logprobs recorded during rollout.
Fireworks RL rollouts are expected to use `temperature=1.0`, so do not treat
non-unit temperature behavior as the validation target for this flag. The
important contract is still that loss ratios/TIS use serving's
`sampling_logprob` field, while raw inference logprobs stay out of loss math.
Raw inference logprobs, when present, are observability-only inputs for
`train/inference_*` drift metrics and must not be used as loss/TIS ratios.
For backward compatibility with legacy completions routes that expose only
`logprob`, the SDK may use raw logprobs as `rollout_logprobs` only under
full-distribution sampling (`temperature=1.0`, `top_p=1.0`, `top_k=0`), where
the raw and behavior-policy values are equivalent.

- `False` recomputes old-policy logprobs via `policy.forward(...)`, preserving
  the historical async default.
- `True` reuses `rollout_logprobs` and skips the old-policy forward
  pass. Use it only when sampler/policy drift is known to be negligible.

`rl_loop.py` keeps its existing public interface: `policy_loss="importance_sampling"`
with `separate_tis=False` reuses `rollout_logprobs`; other modes recompute.

## Related

- How to write a custom loss → [`custom-loss.md`](custom-loss.md)
- Why `optim_step` normalization matters → [`gradient-accumulation.md`](gradient-accumulation.md)
- Kernel registry + dispatch logic: `training/utils/rl/losses.py` and
  `training/utils/rl/builtin_losses.py`. Per-algorithm kernels:
  `training/utils/rl/{grpo,dapo,dro,gspo,cispo}.py`.
