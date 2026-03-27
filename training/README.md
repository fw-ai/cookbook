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

- `examples/snippets/promote_checkpoint.py` reads `checkpoints.jsonl` (produced by cookbook recipes), finds the sampler checkpoint ID and source trainer job, and calls the promotion API to promote it to a deployable Fireworks model. No temporary trainer needed.
- `examples/snippets/reconnect_and_adjust_lr.py` shows how to reconnect to an already-running trainer job and resume training with a different learning rate.

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
examples/snippets/       Standalone utility scripts
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
