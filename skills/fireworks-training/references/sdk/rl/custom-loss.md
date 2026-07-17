# RL: writing a custom loss

Only needed when you cannot express your loss as one of the built-in kernels (GRPO / DAPO / DRO / GSPO / CISPO). Custom losses pay an **extra forward pass** per step (see [`loss-paths.md`](loss-paths.md)); stay on a built-in if you can.

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

`cookbook/training/utils/rl/grpo.py` — follow the same shape (advantages + logprobs + optional KL term, returned as `(scalar, metrics)`).

## Don'ts

- **Don't reimplement `forward_backward_custom`.** Fork `rl_loop.py` and swap only the loss function; the recipe already wires logprob collection, reference-model forward, and client-side loss dispatch.
- **Set dispatch explicitly.** Use `cfg.loss_path="builtin"` only with a registered built-in policy loss, or `cfg.loss_path="client"` for a custom Python loss. There is no silent fallback.
- **Don't forget `grad_accumulation_normalization`** — see [`gradient-accumulation.md`](gradient-accumulation.md). Match the mode to whether your loss returns a raw sum or a pre-normalized mean.

## RL-only `Config` fields you commonly touch

All live on `rl_loop.Config`:

| Field | Default | Meaning |
|---|---|---|
| `loss_path` | `"client"` | Explicitly selects `client` or `builtin` dispatch. |
| `policy_loss` | `"grpo"` | `grpo`, `importance_sampling`, `dapo`, `dro`, `gspo`, `reinforce`, `cispo`. Selects the objective, not the dispatch path. |
| `completions_per_prompt` | `4` | GRPO group size — responses sampled per prompt. |
| `prompt_groups_per_step` | `1` | Number of prompt groups per `forward_backward + optim_step` pair. |
| `kl_beta` | `0.001` | KL-to-reference coefficient. For full-param, the SDK provisions a separate frozen reference trainer and lets backend trainer creation auto-select a LoRA-capable shape unless `cfg.trainer.reference_training_shape_id` pins a LoRA-capable shape. For LoRA, leave it unset — `service.create_reference_client(...)` reuses the policy session with the adapter disabled (the base is the reference). |
| `eps_clip`, `eps_clip_high` | `0.2`, `None` | PPO clip for GRPO. |
| `router_replay` | `False` | Record routing at rollout time and replay during training (MoE models). |
| `grad_accumulation_normalization` | `None` | No server-side normalization by default. Use `NUM_LOSS_TOKENS` for raw-sum losses. See [`gradient-accumulation.md`](gradient-accumulation.md). |

Shape-owned fields (`accelerator_type` / `node_count` / `custom_image_tag`) are always populated from the training profile — never hand-set.

## See also

- Built-in loss registry + per-algorithm kernels: `training/utils/rl/losses.py`, `training/utils/rl/{grpo,dapo,dro,gspo,cispo}.py`.
- `forward_backward_custom` signature + behaviour: `fireworks.training.sdk.client.FiretitanTrainingClient.forward_backward_custom` (`pip show fireworks-ai` → find `src/fireworks/training/sdk/client.py` in the installed package).
