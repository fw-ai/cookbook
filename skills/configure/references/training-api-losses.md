# Training API — losses, datums, and checkpoints

*Source of truth: live [Dedicated Training](https://docs.fireworks.ai/fine-tuning/training-api/dedicated.md), [service client API ref](https://docs.fireworks.ai/fine-tuning/training-api/reference/service-client.md), cookbook recipes. Defer signatures to installed SDK help.*

Use when forking a cookbook recipe, calling `forward_backward` / `forward_backward_custom` directly, or debugging checkpoint promote/resume. Recipe users should start from `references/sdk-recipes.md`.

## Loss paths

| Need | API | Recipe |
|---|---|---|
| SFT next-token | `forward_backward(datums, "cross_entropy")` | `sft_loop` |
| Custom scalar objective | `forward_backward_custom(datums, loss_fn)` | fork RL/DPO recipe |
| Built-in GRPO (trainer) | `forward_backward(grpo_datums, "ppo", loss_fn_config=...)` | fork `rl_loop` / `async_rl_loop` |
| GRPO (client closure) | `forward_backward_custom` + `make_grpo_loss_fn` | stock `rl_loop` default |

RL algorithm policy: `references/rl-custom-loss.md`, `references/rl-loss-paths.md`.

### Built-in `cross_entropy`

Requires `target_tokens` in `loss_fn_inputs`. Weight-based datums from `datum_from_model_input_weights` **fail** with missing `target_tokens`.

```python
result = training_client.forward_backward(datums, "cross_entropy").result()
```

For weight-based SFT objectives, use `forward_backward_custom` with explicit weights — see cookbook `sft_loop` and `references/renderer-verification.md` for token alignment.

### `forward_backward_custom`

1. Trainer forward → logprobs (local tensors, `requires_grad=True`).
2. Your `loss_fn(data, logprobs_list)` returns scalar loss + metrics dict.
3. Trainer backward from `d_loss/d_logprob`.

~1.5× FLOPs vs built-in. Use `torch.dot(logprobs, weights)` for weighted sums.

Embedding objectives: `output="embedding"`, `pooling="mean"|"last"`.

### Gradient accumulation

Call `forward_backward` or `forward_backward_custom` N times, then one `optim_step`. Do **not** use deprecated `gradient_accumulation_steps` on `TrainerJobConfig`. See `references/rl-gradient-accumulation.md`.

### DPO-style custom margin

Pairwise datums + reference logprobs; pattern in cookbook `dpo_loop` and live Dedicated docs quickstart. Prefer the recipe over reimplementing.

## Checkpoints (SDK-level)

Three purposes:

1. **Weight sync / sampling** — `save_weights_for_sampler_ext` → hot-load identity on deployment.
2. **Resume** — `save_state` / `load_state_with_optimizer` (DCP; weights + optimizer).
3. **Promote** — `promote_checkpoint(name=...)` → deployable Fireworks model.

**Sampler snapshot ≠ DCP resume path.** Do not pass sampler paths to `load_state_with_optimizer`.

| `checkpoint_type` | Full-param | LoRA |
|---|---|---|
| `base` | Full weights / adapter; promotable | Full adapter; promotable |
| `delta` | XOR diff; faster sync; **not promotable** | N/A (always full adapter) |

Recipe-driven flow: set `dcp_save_interval`, `output_model_id` on `Config` — see `references/sdk-checkpoints.md` and cookbook `reference#checkpoints`.

### Promote failures

1. `FireworksClient.list_checkpoints(job_id)` — use 4-segment `name` from output.
2. Validate `output_model_id` with `validate_output_model_id` (≤63 chars, `[a-z0-9-]`).
3. Escalate if promotable rows still fail — may need platform re-stage.

### Resume after interrupt

Dedicated: pin same `trainer.job_id`, same `log_path`, auto-resume from newest DCP row. Cross-job init uses `init_from_checkpoint` — see `references/sdk-checkpoints.md`.

## Related

- Checkpoints deep dive: `references/sdk-checkpoints.md`
- Hot-load / incremental snapshots: `references/rl-hotload.md`
- Cookbook tools: `references/sdk-tools.md`
- API signatures: live service-client + fireworks-client docs
