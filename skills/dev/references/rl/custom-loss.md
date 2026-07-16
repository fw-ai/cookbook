# RL: writing a custom loss

The generic sync and async recipes call the client-side GRPO builder directly.
Fork the documented forward/backward function and replace that one builder when
you need another Python research loss. See
[`skills/customize-rl-loss/SKILL.md`](../../../customize-rl-loss/SKILL.md).

## Interface

```python
def my_loss(data, logprobs_list):
    # data: dict of aligned tensors (advantages, inference logprobs, prompt lens, etc.)
    # logprobs_list: per-datum training logprobs from the trainer forward pass
    loss = compute_loss(data, logprobs_list)   # scalar torch.Tensor, requires_grad
    return loss, {"loss": float(loss.item())}  # (loss_tensor, metrics_dict)
```

The recipe passes it into `training_client.forward_backward_custom(datums, my_loss).result()`.

## Reference implementation

`cookbook/training/utils/rl/grpo.py` — follow the same shape (advantages +
logprobs + optional KL term, returned as `(scalar, metrics)`). Other direct
builders live beside it (`dapo.py`, `gspo.py`, `cispo.py`, and so on).

## Don'ts

- **Don't reimplement `forward_backward_custom`.** Fork `rl_loop.py` and replace
  the documented direct GRPO builder with another loss closure.
- **Don't add an automatic fallback.** A fork should call either
  `forward_backward` or `forward_backward_custom` explicitly.
- **Don't silently reuse the GRPO reference.** Keep or remove reference
  provisioning explicitly based on what the replacement closure consumes.
- **Don't forget `grad_accumulation_normalization`** — see [`gradient-accumulation.md`](gradient-accumulation.md). Match the mode to whether your loss returns a raw sum or a pre-normalized mean.

## RL-only `Config` fields you commonly touch

All live on `rl_loop.Config`:

| Field | Default | Meaning |
|---|---|---|
| `completions_per_prompt` | `4` | GRPO group size — responses sampled per prompt. |
| `prompt_groups_per_step` | `1` | Number of prompt groups per `forward_backward + optim_step` pair. |
| `kl_beta` | `0.001` | Reference-KL coefficient; `0` skips the reference. |
| `eps_clip`, `eps_clip_high` | `0.2`, `None` | PPO clip for GRPO. |
| `router_replay` | `False` | Record routing at rollout time and replay during training (MoE models). |
| `grad_accumulation_normalization` | `None` | No server-side normalization by default. Use `NUM_LOSS_TOKENS` for raw-sum losses. See [`gradient-accumulation.md`](gradient-accumulation.md). |

Shape-owned fields (`accelerator_type` / `node_count` / `custom_image_tag`) are always populated from the training profile — never hand-set.

## See also

- Built-in GRPO datum preparation: `training/utils/rl/losses.py`.
- Direct client loss builders: `training/utils/rl/{grpo,dapo,dro,gspo,cispo}.py`.
- `forward_backward_custom` signature + behaviour: `fireworks.training.sdk.client.FiretitanTrainingClient.forward_backward_custom` (`pip show fireworks-ai` → find `src/fireworks/training/sdk/client.py` in the installed package).
