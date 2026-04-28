# Training and deployment shapes — always use a profile

Shapes are the required entry point for both trainer and deployment. Never hand-set `accelerator_type`, `accelerator_count`, `node_count`, or `custom_image_tag` when a shape is in use — the backend will reject or silently ignore them.

## Training shape

Set `cfg.infra.training_shape_id`:

```python
cfg.infra.training_shape_id = "accounts/fireworks/trainingShapes/ts-qwen3-8b-policy"
```

The recipe then does:

```python
profile = rlor_mgr.resolve_training_profile(cfg.infra.training_shape_id)
# profile.training_shape_version
# profile.deployment_shape_version
# profile.max_supported_context_length
# profile.accelerator_type, profile.node_count, ...  (read, do not copy to cfg)
```

See `training/recipes/sft_loop.py` (search `resolve_training_profile`) and `training/recipes/rl_loop.py` (same — called once per policy, once per reference).

## Deployment shape

Do not set `cfg.deployment.deployment_shape` manually. The recipe copies it from the training profile:

```python
if not cfg.deployment.deployment_shape and profile.deployment_shape_version:
    cfg.deployment.deployment_shape = profile.deployment_shape_version
```

That is a **versioned** path (`accounts/fw/deploymentShapes/ds-x/versions/abc123`). The `to_deployment_config` helper in `training/utils/config.py` auto-clears manual accelerator fields whenever a shape is present.

## Reference-model shape (RL / DPO)

For **full-parameter** training with a frozen reference, set `cfg.infra.ref_training_shape_id` explicitly — there is no implicit fallback. It can share the same shape as the policy; the control plane appends `--forward-only` automatically.

For **LoRA** (`lora_rank > 0`), two valid options:
- **Shared session (recommended, saves GPUs)**: leave `ref_training_shape_id` unset. `setup_infra` uses `policy.create_base_reference()` on the policy trainer for reference logprobs — no separate trainer, no extra GPUs.
- **Separate LoRA-capable ref trainer**: set `ref_training_shape_id` to a `LORA_TRAINER` shape (typically the same as the policy shape). `setup_infra` provisions a forward-only LoRA ref trainer (its own GPUs) and forwards `lora_rank` to both the trainer request and the ref client so the gateway infers `trainer_mode=LORA_TRAINER` and matches the shape. CP's V2 DPO auto-resolver picks this path by default for LoRA DPO.

The CI pattern for the saves-GPUs variant is `ref_shape = "" if lora_rank > 0 else <explicit shape>`.

## When to skip validation

`cfg.infra.skip_validations=True` is a superuser-only escape hatch for shapes not yet registered. Agents should not set this unless explicitly told to.

## Listing available shapes

```bash
firectl training-shape list      # alias: firectl ts list
firectl deployment-shape list    # alias: firectl ds list
```

Or programmatically via `FireworksClient` — see the SDK docs linked from the repo README.

## Do not pin a `/versions/<id>`

Pass the bare shape path `accounts/fireworks/trainingShapes/<shape>`. The platform auto-selects the latest validated version for you. Hand-picking a version is almost always wrong:

- The platform only serves validated versions — a versioned ref cannot force an unvalidated one.
- Pinning locks the run to a stale version and prevents the platform from rolling the shape forward when a better-validated image lands.

For the full list of shapes in your account, run `firectl training-shape list` (above).

## When `resolve_training_profile` raises `Failed to resolve latest validated training shape`

This means the shape currently has no validated version at all — usually a transient state right after a shape update. Pinning to an older `/versions/<id>` won't help. Retry after a short wait; if it persists, reach out to Fireworks support.
