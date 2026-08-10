# Training and deployment shapes — always use a profile

Shapes are the required entry point for both trainer and deployment. Cookbook
trainer config rejects `accelerator_type` and `accelerator_count`; clients
cannot select trainer accelerators, so set `training_shape_id` instead.
`node_count` and `custom_image_tag` remain advanced controls but should not be
hand-set when a shape is in use.

## Training shape

By default, recipes auto-select the smallest validated training shape that can
fit the configured model and context length:

```python
cfg.trainer.training_shape_id = None
```

Set `cfg.trainer.training_shape_id` only when you need an explicit override:

```python
cfg.trainer.training_shape_id = "accounts/fireworks/trainingShapes/ts-qwen3-8b-policy"
```

The recipe then does:

```python
profile = rlor_mgr.resolve_training_profile(resolved_training_shape_id)
# profile.training_shape_version
# profile.deployment_shape_version
# profile.max_supported_context_length
# profile.accelerator_type, profile.node_count, ...  (read, do not copy to cfg)
```

See `training/recipes/async_rl_loop.py` and
`FiretitanServiceClient.from_firetitan_config(...)` for the refactored
trainer/deployment provisioning path.

## Deployment shape

Do not set `cfg.deployment.deployment_shape` manually. The SDK resolves it from
the requested deployment shape or the selected training profile, and recipes read
the resolved value from the service:

```python
service = build_service_client(...)
training_client = service.create_training_client(...)
deployment_shape = service.deployment_shape
```

That is a **versioned** path (`accounts/fw/deploymentShapes/ds-x/versions/abc123`).
The `to_deployment_config` helper in `training/utils/config.py` auto-clears
manual accelerator fields whenever a shape is present.

## Reference-model shape (RL / DPO)

For **full-parameter** training with a frozen reference, leave `cfg.trainer.reference_training_shape_id` unset to let backend trainer creation auto-select a compatible `LORA_TRAINER` shape. Set it explicitly only when you need an override; it should be a LoRA-capable shape and can share the same shape as the policy.

For **LoRA** (`lora_rank > 0`), two valid options:
- **Shared session (recommended, saves GPUs)**: leave `cfg.trainer.reference_training_shape_id` unset. `service.create_reference_client(...)` reuses the policy session with the adapter disabled for reference logprobs — no separate trainer, no extra GPUs.
- **Separate LoRA-capable ref trainer**: set `cfg.trainer.reference_training_shape_id` to a `LORA_TRAINER` shape (typically the same as the policy shape). The SDK provisions a frozen reference runtime on its own GPUs and requests `trainer_mode=LORA_TRAINER` shape matching.

The CI pattern for the saves-GPUs variant is `ref_shape = "" if lora_rank > 0 else None`, letting backend auto-selection handle the full-parameter reference.

## When to skip validation

`cfg.trainer.skip_validations=True` is a superuser-only escape hatch for shapes not yet registered. Agents should not set this unless explicitly told to.

## Listing available shapes

Read shapes through the **version** collection, not the parent shape resource:

```bash
# Shared catalog entries your account can actually launch on
firectl training-shape-version list --base-model accounts/fireworks/models/<model>

# Every field of the exact version a launch will resolve to
firectl training-shape-version get \
  accounts/fireworks/trainingShapes/<shape>/versions/latest
```

Or programmatically via `FireworksClient.resolve_training_profile(<shape_id>)`, which
hits the same version collection.

### Why not `firectl training-shape list` / `get`

`TrainingShape` (the parent resource) is account-scoped and carries no public
visibility flag, so it is readable only by the account that owns it:

- `firectl training-shape list` accepts no wildcard parent (`ListTrainingShapes`
  rejects `accounts/-` with `InvalidArgument`), so it can only ever list shapes
  **in one account**. For a customer account that is normally empty — it never
  shows the shared `accounts/fireworks/...` catalog.
- `firectl training-shape get accounts/fireworks/trainingShapes/<shape>` is a
  cross-account read into the `fireworks` account and returns
  `rpc error: code = PermissionDenied` for every customer principal.

Neither result depends on Training API entitlement, so **neither command is an
access check**. Public visibility lives on `TrainingShapeVersion.public`, which is
why the version reads above work and the parent reads do not.

## Do not pin a `/versions/<id>`

Pass the bare shape path `accounts/fireworks/trainingShapes/<shape>`. The platform auto-selects the latest validated version for you. Hand-picking a version is almost always wrong:

- The platform only serves validated versions — a versioned ref cannot force an unvalidated one.
- Pinning locks the run to a stale version and prevents the platform from rolling the shape forward when a better-validated image lands.

To see the versions behind a shape, run `firectl training-shape-version list` (above).

## When `resolve_training_profile` raises `Failed to resolve latest validated training shape`

This means the shape currently has no validated version at all — usually a transient state right after a shape update. Pinning to an older `/versions/<id>` won't help. Retry after a short wait; if it persists, reach out to Fireworks support.
