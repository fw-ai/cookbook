# fireworks-training-infra

Standalone helpers for spinning up Fireworks training jobs and inference
deployments. Extracted from
[`fireworks-training-cookbook`](https://github.com/fw-ai/cookbook) so that
projects can `pip install fireworks-training-infra` without pulling in the
whole cookbook (datasets, losses, renderers, etc.).

## What you get

- `setup_infra` — one-call provisioning for policy + reference trainers and an
  optional inference deployment. Used by every cookbook recipe.
- `setup_deployment` / `setup_or_reattach_deployment` — single-deployment helpers
  for SFT-style runs.
- `create_trainer_job` / `request_trainer_job` / `wait_trainer_job` — single
  trainer job lifecycle.
- `ReconnectableClient` — thin wrapper over `FiretitanTrainingClient` with
  dispatch-and-wait semantics.
- `InfraConfig`, `DeployConfig`, `WeightSyncScope` — dataclass configs.
- `ResourceCleanup` — context manager that cancels trainers and tears down
  deployments on scope exit.
- `auto_select_training_shape` — pick a validated training shape for a base
  model.

## Install

```bash
pip install fireworks-training-infra
```

## Usage

```python
from fireworks.training.sdk import TrainerJobManager
from fireworks.training.sdk.deployment import DeploymentManager
from fireworks_training_infra import (
    DeployConfig,
    InfraConfig,
    ResourceCleanup,
    setup_infra,
)

rlor_mgr = TrainerJobManager(account_id="my-account", api_key="...")
deploy_mgr = DeploymentManager(account_id="my-account", api_key="...")

with ResourceCleanup(rlor_mgr, deploy_mgr) as cleanup:
    infra = setup_infra(
        rlor_mgr=rlor_mgr,
        deploy_mgr=deploy_mgr,
        base_model="accounts/fireworks/models/qwen3-1p7b",
        infra_cfg=InfraConfig(),
        deploy_cfg=DeployConfig(),
        role_prefix="my-run",
        api_key="...",
        needs_inference=True,
        cleanup=cleanup,
    )
    # infra.policy, infra.reference, infra.inference_model, ...
```
