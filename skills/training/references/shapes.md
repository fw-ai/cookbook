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

Done in `training/recipes/sft_loop.py` around line 325 and `training/recipes/rl_loop.py` around line 336.

## Deployment shape

Do not set `cfg.deployment.deployment_shape` manually. The recipe copies it from the training profile:

```python
if not cfg.deployment.deployment_shape and profile.deployment_shape_version:
    cfg.deployment.deployment_shape = profile.deployment_shape_version
```

That is a **versioned** path (`accounts/fw/deploymentShapes/ds-x/versions/abc123`). The `to_deployment_config` helper in `training/utils/config.py` auto-clears manual accelerator fields whenever a shape is present.

## Reference-model shape (RL only)

For RL with a frozen reference, set `cfg.infra.ref_training_shape_id` explicitly — there is no implicit fallback. It can share the same shape as the policy; the control plane appends `--forward-only` automatically.

## When to skip validation

`cfg.infra.skip_validations=True` is a superuser-only escape hatch for shapes not yet registered. Agents should not set this unless explicitly told to.

## Listing available shapes

```bash
firectl list training-shapes
firectl list deployment-shapes
```

Or programmatically via `FireworksClient` — see the SDK docs linked from the repo README.
