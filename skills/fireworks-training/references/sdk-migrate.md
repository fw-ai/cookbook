# Migrating off the deprecated managed infra (`InfraConfig` / `setup_infra`)

The cookbook moved trainer/deployment provisioning out of the recipe layer and
behind SDK-managed provisioning. Refactored recipes build one SDK-managed
service client and then use Tinker-shaped `service.create_*` calls. Recipe `Config` objects now take
`trainer=TrainerConfig(...)` and `deployment=DeployConfig(...)`. The old
`InfraConfig`, the recipe `infra=` / `weight_sync=` kwargs, and the standalone
`setup_infra` / `ResourceCleanup` / `make_reference_client` /
`create_base_reference` helpers are gone from the recipe surface.

## When to use this

- A script imports `InfraConfig`, `setup_infra`, `ResourceCleanup`,
  `make_reference_client`, or `create_base_reference`.
- A recipe `Config(...)` passes `infra=` or `weight_sync=`.
- `TypeError: __init__() got an unexpected keyword argument 'infra'` (or
  `'weight_sync'`) after a cookbook upgrade.
- `ImportError: cannot import name 'setup_infra'` / `'ResourceCleanup'`.

> **Opt-in.** If a user does not want to migrate, they can simply **not upgrade
> the SDK + cookbook** — pin the current versions and old code keeps working.
> The old and new surfaces do not coexist in one install. Upgrading is
> recommended (one provisioning path, SDK-owned lifecycle, cleaner config) but
> not required.

## Find every call site

```bash
grep -rn -E 'InfraConfig|setup_infra|ResourceCleanup|make_reference_client|create_base_reference|\binfra=|weight_sync=|ref_training_shape_id|trainer_timeout_s|trainer_replica_count|policy_job_id' \
  --include=*.py .
```

## Field / API rename table

| Before (deprecated) | After (current) |
|---|---|
| `Config(infra=InfraConfig(...))` | `Config(trainer=TrainerConfig(...))` |
| `InfraConfig.ref_training_shape_id` | `TrainerConfig.reference_training_shape_id` |
| `InfraConfig.trainer_timeout_s` | `TrainerConfig.timeout_s` |
| `InfraConfig.trainer_replica_count` | `TrainerConfig.replica_count` |
| `Config(weight_sync=WeightSyncConfig(weight_sync_interval=N))` | `Config(weight_sync_interval=N)` (top-level, RL family) |
| `weight_sync.dcp_save_interval=N` | `Config(dcp_save_interval=N)` (top-level, all recipes) |
| top-level `policy_job_id=...` | `TrainerConfig(job_id=...)` |
| `setup_infra(rlor_mgr, deploy_mgr, ...)` | recipe `main(cfg)`, or `FiretitanServiceClient.from_firetitan_config(...)` plus `service.create_*` for raw clients |
| `create_base_reference()` / `make_reference_client()` | `service.create_reference_client(...)` |
| `with ResourceCleanup(...)` | `service.close()` / context-managed cleanup owned by the SDK service lifetime |

`accelerator_type` / `accelerator_count` are deprecated and ignored by the SDK;
the training shape owns accelerator selection. `node_count` and
`custom_image_tag` remain advanced controls; prefer a training shape.

Trainer provisioning has two independent wait budgets. Set
`TrainerConfig.pending_timeout_s` for capacity placement while the job is
`PENDING` (48 hours by default), and `TrainerConfig.timeout_s` for
post-placement startup/readiness (1 hour by default). A long capacity wait no
longer consumes the readiness budget. This requires an SDK version that exposes
`trainer_pending_timeout_s`; upgrade the SDK before using this cookbook version.

## Recipe-launch migration (the common case)

Most scripts just build a recipe `Config` and call `main(cfg)`. Rename the
nested configs:

```python
# BEFORE
from training.utils import InfraConfig, WeightSyncConfig
cfg = rl_loop.Config(
    ...,
    infra=InfraConfig(
        training_shape_id="accounts/fireworks/trainingShapes/qwen3-8b-128k-h200",
        ref_training_shape_id="accounts/fireworks/trainingShapes/qwen3-8b-128k-h200-lora",
        trainer_replica_count=2,
    ),
    weight_sync=WeightSyncConfig(weight_sync_interval=1, dcp_save_interval=10),
)

# AFTER
from training.utils import TrainerConfig
cfg = rl_loop.Config(
    ...,
    trainer=TrainerConfig(
        training_shape_id="accounts/fireworks/trainingShapes/qwen3-8b-128k-h200",
        reference_training_shape_id="accounts/fireworks/trainingShapes/qwen3-8b-128k-h200-lora",
        replica_count=2,
    ),
    weight_sync_interval=1,   # top-level (RL family)
    dcp_save_interval=10,     # top-level (all recipes)
)
```

`main(cfg)` is unchanged — drop any `rlor_mgr=` / `deploy_mgr=` arguments; the
recipe provisions internally.

## API-level migration (`setup_infra` → SDK-managed provisioning)

Scripts that called `setup_infra` directly to get raw policy/reference clients
should prefer a recipe-level config and `main(cfg)`. If a script truly needs
raw RL clients, create one SDK-managed service and consume its resolved
properties:

```python
# BEFORE
from training.utils import InfraConfig, ResourceCleanup
from training.utils.rl import setup_infra
with ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup:
    infra = setup_infra(rlor_mgr=rlor_mgr, deploy_mgr=deploy_mgr, base_model=base_model,
                        infra_cfg=infra_cfg, deploy_cfg=deploy_cfg, lora_rank=0,
                        needs_reference=True, needs_inference=True, role_prefix="grpo",
                        api_key=api_key, cleanup=cleanup)
    policy, reference = infra.policy, infra.reference

# AFTER
from fireworks.training.sdk import FiretitanServiceClient
from training.utils import ReconnectableClient

service = FiretitanServiceClient.from_firetitan_config(
    api_key=api_key, base_url=base_url, additional_headers=None,
    base_model=base_model, tokenizer_model=tokenizer_model, lora_rank=0,
    max_context_length=max_seq_len, learning_rate=lr,
    training_shape_id=..., reference_training_shape_id=...,
    deployment_id=..., cleanup_trainer_on_close=True,
)
try:
    training_client = service.create_training_client(base_model, lora_rank=0)
    policy = ReconnectableClient.from_training_client(
        training_client, base_model=base_model,
        job_id=service.trainer_job_id, service=service)
    if kl_beta > 0:
        reference_client = service.create_reference_client(base_model, lora_rank=0)
    sampler = service.create_deployment_sampler(tokenizer=tokenizer)
    # ... train ...
finally:
    service.close()
```

The SDK owns the reference strategy: LoRA without an explicit
`reference_training_shape_id` reuses the policy session; full-param (or an
explicit reference shape) provisions a separate frozen reference trainer that
`service` owns. When no full-param reference shape is pinned, backend trainer
creation auto-selects a LoRA-capable shape. The reference trainer is torn down per
`TrainerConfig.cleanup_reference_on_close` (default `True`).

## Verify the port

```bash
# No deprecated symbols remain
grep -rn -E 'InfraConfig|setup_infra|ResourceCleanup|create_base_reference|make_reference_client|\binfra=|weight_sync=' --include=*.py .
# Recipes import + Config builds
python -c "import training.recipes.rl_loop as r; r.Config(log_path='x')"
```

`InfraConfig` is still importable for back-compat and emits a
`DeprecationWarning` when constructed, but recipe `Config` objects no longer
accept it.
