# Fireworks Training Cookbook

Ready-to-run training recipes for reinforcement learning, preference optimization, and supervised fine-tuning on [Fireworks](https://fireworks.ai).
Each recipe is a single Python file you can fork and customize.

## Recipes

| Recipe | File | Description |
| --- | --- | --- |
| GRPO / DAPO / GSPO / CISPO | `recipes/rl_loop.py` | On-policy RL with streaming rollouts. Set `policy_loss="grpo"`, `"dapo"`, `"gspo"`, or `"cispo"`. |
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
FIREWORKS_ACCOUNT_ID="your-account-id"
```

Or export them directly:

```bash
export FIREWORKS_API_KEY="..."
export FIREWORKS_ACCOUNT_ID="..."
```

### 3. Configure your recipe

Each recipe has a `Config` dataclass at the top of the file. Open the recipe you want to run and edit the `if __name__ == "__main__"` block at the bottom. Here are the fields you **must** set:

**All recipes:**

| Field | What to set |
| --- | --- |
| `dataset` | Path to your JSONL training data |
| `base_model` | Fireworks model ID (e.g. `"accounts/fireworks/models/qwen3-8b"`) |
| `max_seq_len` | Max token length for training examples |
| `infra` | `InfraConfig(training_shape_id="your-shape")` for GPU provisioning |

**SFT** (`recipes/sft_loop.py`) -- also requires:

| Field | What to set |
| --- | --- |
| `tokenizer_model` | HuggingFace model name matching your base model (e.g. `"Qwen/Qwen3-8B"`) |

**RL** (`recipes/rl_loop.py`) -- also requires:

| Field | What to set |
| --- | --- |
| `deployment` | `DeployConfig(tokenizer_model="Qwen/Qwen3-8B")` for inference rollouts |
| `hotload` | `HotloadConfig(hot_load_interval=1)` to sync weights to the deployment |

**DPO / ORPO** -- also requires:

| Field | What to set |
| --- | --- |
| `tokenizer_model` | HuggingFace model name matching your base model |

### 4. Run

```bash
cd cookbook/training
python -m recipes.sft_loop      # or whichever recipe you configured
```

## Example: SFT quick start

```python
# At the bottom of recipes/sft_loop.py, edit the __main__ block:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(
        base_model="accounts/fireworks/models/qwen3-8b",
        dataset="path/to/your_data.jsonl",
        tokenizer_model="Qwen/Qwen3-8B",
        max_seq_len=4096,
        max_examples=100,                           # optional: limit for testing
        infra=InfraConfig(
            training_shape_id="your-training-shape",
        ),
        # learning_rate=1e-4,
        # epochs=3,
        # batch_size=32,
        # lora_rank=0,                              # set > 0 for LoRA
        # wandb=WandBConfig(project="my-sft"),
    )
    main(cfg)
```

Then run:

```bash
python -m recipes.sft_loop
```

## Dataset format

All recipes expect JSONL files with OpenAI chat format:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

- **SFT**: trains on assistant response tokens only (prompt tokens are masked).
- **DPO / ORPO**: each row needs `chosen` and `rejected` message lists.
- **RL**: each row needs a `messages` list with at least a user prompt; the reward function scores completions.

## Configuration reference

All recipes use composable dataclass configs:

- **`InfraConfig`** -- region, accelerators, training shapes. When `training_shape_id` is set, `max_seq_len` and GPU config are auto-derived.
- **`DeployConfig`** -- inference deployment for RL rollouts. Set `deployment_id` to reuse an existing deployment, or omit to auto-create.
- **`HotloadConfig`** -- weight sync cadence and checkpoint settings.
- **`WandBConfig`** -- optional Weights & Biases logging.
- **`ResumeConfig`** -- resume training from a checkpoint.

## Directory layout

```
recipes/            Training loop scripts (fork these)
utils/              Shared config, data loading, loss functions, metrics
examples/deepmath/  Worked example: math reasoning with GRPO
tests/              Unit and end-to-end tests
```

## Tests

```bash
uv pip install -e ".[dev]"
pytest tests/
```
