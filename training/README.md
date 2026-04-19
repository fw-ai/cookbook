# Fireworks Training Cookbook

Ready-to-run training recipes for reinforcement learning, preference optimization, and supervised fine-tuning on [Fireworks](https://fireworks.ai).
Each recipe is a single Python file you can fork and customize.

> **Full documentation**: For detailed guides on each recipe, configuration reference, and the Training SDK, see the [Training SDK documentation](https://docs.fireworks.ai/fine-tuning/training-sdk/introduction).

## Recipes

| Recipe | File | Description |
| --- | --- | --- |
| GRPO / IS / DAPO / DRO / GSPO / CISPO | `recipes/rl_loop.py` | On-policy RL with streaming rollouts. Set `policy_loss="grpo"`, `"importance_sampling"`, `"dapo"`, `"dro"`, `"gspo"`, or `"cispo"`. |
| DPO | `recipes/dpo_loop.py` | Direct preference optimization with cached reference logprobs. |
| ORPO | `recipes/orpo_loop.py` | Odds-ratio preference optimization -- no reference model needed. |
| SFT | `recipes/sft_loop.py` | Supervised fine-tuning with response-only cross-entropy loss. |

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/fw-ai/cookbook.git
cd cookbook/training

# Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install the Fireworks training SDK prerelease that provides
# `fireworks.training.sdk`.
uv pip install --pre "fireworks-ai>=1.0.0a36" tinker-cookbook

# Install this package in editable mode
uv pip install -e .

# If you skip `--pre`, pip may resolve to the stable `0.x` line,
# which does not include `fireworks.training.sdk` and will cause
# imports like `from fireworks.training.sdk import DeploymentManager`
# to fail.
```

### 2. Set your credentials

Create a `.env` file in the `training/` directory (picked up automatically via `python-dotenv`):

```bash
FIREWORKS_API_KEY="your-api-key"
```

Or export it directly:

```bash
export FIREWORKS_API_KEY="..."
```

### 3. Configure your recipe

Each recipe has a `Config` dataclass at the top of the file. Open the recipe you want to run and edit the `if __name__ == "__main__"` block at the bottom. Here are the fields you **must** set:

**All recipes:**

| Field | What to set |
| --- | --- |
| `dataset` | Path to your JSONL training data |
| `base_model` | Fireworks model ID (e.g. `"accounts/fireworks/models/qwen3-8b"`) |
| `max_seq_len` | Max token length for training examples |
| `infra` | `InfraConfig()` to auto-select validated shapes from Fireworks control-plane data, or `InfraConfig(training_shape_id="your-shape")` to override them |

**SFT** (`recipes/sft_loop.py`) -- also requires:

| Field | What to set |
| --- | --- |
| `tokenizer_model` | HuggingFace model name matching your base model (e.g. `"Qwen/Qwen3-8B"`) |

**RL** (`recipes/rl_loop.py`) -- also requires:

| Field | What to set |
| --- | --- |
| `deployment` | `DeployConfig(tokenizer_model="Qwen/Qwen3-8B")` for inference rollouts |
| `weight_sync` | `WeightSyncConfig(weight_sync_interval=1)` to sync weights to the deployment |

When `training_shape_id` is not set, the cookbook auto-selects validated
trainer shapes at runtime from Fireworks control-plane data. RL-family
recipes also auto-select a validated deployment shape, and DPO / KL
flows request a validated forward-only reference shape. Explicit
`training_shape_id`, `ref_training_shape_id`, and deployment-shape
overrides still take precedence.

**DPO / ORPO** -- also requires:

| Field | What to set |
| --- | --- |
| `tokenizer_model` | HuggingFace model name matching your base model |

### 4. Run

```bash
cd cookbook/training
python -m recipes.sft_loop      # or whichever recipe you configured
```

## Useful examples

- `examples/tools/promote_checkpoint.py` reads `checkpoints.jsonl` (produced by cookbook recipes), finds the sampler checkpoint ID and source trainer job, and calls the promotion API to promote it to a deployable Fireworks model. No temporary trainer needed.
- `examples/tools/list_checkpoints.py` lists the server's authoritative view of checkpoints for a trainer job (sampler + DCP, promotable or not). Thin wrapper over `FireworksClient.list_checkpoints()`.
- `examples/tools/reconnect_and_adjust_lr.py` shows how to reconnect to an already-running trainer job and resume training with a different learning rate.

## Serving a promoted full-tune model

SFT and ORPO runs promote their final checkpoint with `kind=HF_BASE_MODEL` (full-parameter). These promoted models are **not** on serverless — to issue an inference call you must create a dedicated deployment whose `base_model` points at the promoted model. Sketch:

```python
from fireworks.training.sdk import DeploymentManager, DeploymentConfig

deploy_mgr = DeploymentManager(api_key=api_key)
deployment = deploy_mgr.create_or_get(DeploymentConfig(
    deployment_id="my-sft-eval",                        # short, [a-z0-9-]
    base_model="accounts/<account>/models/<promoted-id>",
    region=None,                                         # let control plane auto-place
    min_replica_count=0,
    max_replica_count=1,
))
# DeploymentManager methods take the short deployment_id; DeploymentInfo is a
# dataclass with attribute access (deployment.deployment_id, .name, .state).
deploy_mgr.wait_for_ready(deployment.deployment_id, timeout_s=1800)
# ... issue chat-completion against deployment.name ...
deploy_mgr.delete(deployment.deployment_id)              # see note on async delete below
```

**Cold-start latency hazard**: provisioning a fresh dedicated deployment for a full-tune promoted model can take **tens of minutes**, and occasionally much longer under capacity pressure. For a quick "does my fine-tune emit sensible text?" check on a smoke-scale budget, prefer one of:

- Train with `--lora-rank > 0` and hot-load the promoted LoRA adapter onto an existing base-model deployment (see `skills/dev/references/rl/hotload.md`).
- Reuse an already-warm deployment of the same base family if one exists in the account.
- Budget ≥90 minutes if a fresh dedicated full-tune deployment is unavoidable.

**Async cleanup**: `DeploymentManager.delete(deployment_id)` returns before the backing deployed model is fully undeployed; a subsequent `DELETE /models/<id>` can fail with "cannot delete model, found 1 active deployed models" until propagation completes (~30s). Retry the model delete or poll the deployment GET for 404 before deleting the model.

## Serving a promoted LoRA adapter

LoRA adapters promoted from a LoRA-rank run (`kind=HF_PEFT_ADDON`) are served differently from full-tune models — they hot-load onto an existing base deployment rather than requiring a dedicated one. The two-step pattern:

```python
from fireworks import Fireworks

c = Fireworks(api_key=api_key)

# 1. Stand up (or reuse) a base deployment with addon-serving enabled.
#
# Hardware: do not hand-set `accelerator_type`. Either omit it (the control
# plane picks a shape compatible with `base_model`) or pass a versioned
# `deployment_shape="accounts/fireworks/deploymentShapes/<shape>"` — see
# `skills/dev/references/shapes.md` for why shape-first beats raw accelerator.
c.deployments.create(
    deployment_id="my-lora-host",
    base_model="accounts/fireworks/models/qwen3-8b",
    enable_addons=True,               # required for c.lora.load to work
    min_replica_count=1,
)
# 2. Hot-load the promoted adapter onto that deployment.
c.lora.load(
    model="accounts/<account>/models/<promoted-lora-id>",
    deployment="accounts/<account>/deployments/my-lora-host",
    replace_merged_addon=True,
)
# ... issue chat-completions against <base-model>#<deployment-id> ...
```

**Flag mutual exclusivity**: `enable_addons=True` and `enable_hot_reload_latest_addon=True` are two different hot-load regimes and cannot both be set — the API returns `400 EnableAddons and EnableHotReloadLatestAddon cannot both be enabled`. Use `enable_addons=True` with `c.lora.load(..., replace_merged_addon=True)` for explicit-load-by-promoted-model-id (the pattern above); use `enable_hot_reload_latest_addon=True` for the auto-pull-latest-adapter regime.

**Adapter-lifecycle hazard**: a promoted adapter can be garbage-collected within minutes after its training job is cancelled. If you follow the "train + promote + hot-load" pattern, make sure `c.lora.load(...)` completes **before** the trainer is cancelled. For RL recipes, passing `--skip-cleanup` to the driver keeps the trainer alive past the recipe's normal exit so you can complete the hot-load (and any post-training A/B) before cleaning up manually.

## Addressing a trained policy for inference

RL recipes promote the final checkpoint **and** hot-load its weights onto the rollout deployment. To query the trained policy, route requests at the deployment, not at the promoted-model id:

```
model = "<baseModel>#<deployment_id>"
# e.g. accounts/fireworks/models/qwen3-4b#accounts/<account>/deployments/<deployment-id>
```

The promoted model id alone returns 404 unless you create a separate dedicated deployment whose `base_model` points at it. For A/B comparisons against the untrained base, stand up a second deployment of the base model with the same shape — base-family models that are not on serverless for your account require an on-demand deployment.

## Documentation

For detailed guides, configuration reference, and examples, see the official documentation:

- [Introduction & Quickstart](https://docs.fireworks.ai/fine-tuning/training-sdk/introduction)
- [Cookbook recipes (SFT, RL, DPO, ORPO)](https://docs.fireworks.ai/fine-tuning/training-sdk/cookbook/overview)
- [Configuration reference](https://docs.fireworks.ai/fine-tuning/training-sdk/cookbook/reference)

## Directory layout

```
recipes/                 Training loop scripts (fork these)
utils/                   Shared config, data loading, loss functions, metrics
examples/rl/deepmath/    Worked example: math reasoning with GRPO
examples/rl/frozen_lake/ Worked example: Frozen Lake with tool-use RL
examples/orpo/ifeval/    Worked example: IFEval with ORPO
examples/sft/            Worked example: SFT getting started
examples/dpo/            Worked example: DPO
examples/tools/          Standalone utility scripts
tests/                   Unit and end-to-end tests
```

## Tests

```bash
uv pip install -e ".[dev]"
pytest tests/
```

Coverage for the training entrypoints:

```bash
cd training
pytest -q tests/unit tests/test_smoke_imports.py examples/rl/frozen_lake/test_masking.py \
  --cov=. \
  --cov-report=term-missing \
  --cov-report=json:coverage.json
python tests/coverage_summary.py coverage.json
```

See [issues/training-script-coverage-baseline.md](./issues/training-script-coverage-baseline.md)
for the current baseline and
[issues/training-script-coverage-roadmap.md](./issues/training-script-coverage-roadmap.md)
for the expansion plan.
