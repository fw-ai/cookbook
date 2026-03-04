# Training Cookbook

This directory hosts the standalone training cookbook code that was moved out of the Python SDK repo.

- Code root: `src/training_cookbook`
- Main docs: `src/training_cookbook/README.md`
- Tests: `src/training_cookbook/tests`

## Quick start

```bash
# 1. From the training/ directory, run setup (installs uv if needed, creates venv, installs deps)
./setup.sh

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Verify the install
python -c "from training_cookbook.recipes.sft_loop import Config; print('OK')"
```

`setup.sh` handles everything: installs [uv](https://docs.astral.sh/uv/) if it isn't already on your PATH, creates a `.venv` virtualenv, and runs `uv pip install -e ".[dev]"` to install the package in editable mode with dev dependencies (pytest, math-verify).

## Environment variables

All recipes require the following environment variables:

```bash
export FIREWORKS_API_KEY=...        # Your Fireworks API key
export FIREWORKS_ACCOUNT_ID=...     # Your Fireworks account ID
```

Optionally:

```bash
export FIREWORKS_BASE_URL=...       # Defaults to https://api.fireworks.ai
```

You can also place these in a `.env` file in the project root — the recipes load it automatically via `python-dotenv`.

## Running a recipe

```python
from training_cookbook.recipes.sft_loop import Config, main
from training_cookbook.utils import InfraConfig, DeployConfig

cfg = Config(
    dataset="path/to/chat_data.jsonl",
    tokenizer_model="Qwen/Qwen3-8B",
    base_model="accounts/fireworks/models/qwen3-8b",
    infra=InfraConfig(region="US_OHIO_1"),
)

main(cfg)
```

See `src/training_cookbook/README.md` for all available recipes (SFT, GRPO, DPO, ORPO) and configuration options.
