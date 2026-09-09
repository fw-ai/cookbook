# Environment Setup

## Install

```bash
git clone https://github.com/fw-ai/cookbook.git
cd cookbook/training

# Option A: conda
conda create -n cookbook python=3.12 -y && conda activate cookbook
python -m pip install -e .

# Option B: uv
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .
```

The cookbook requires `fireworks-ai[training]>=1.2.11,<2`, available as a stable
PyPI release. `--pre` is not required. The legacy `0.19.20` package does not
contain `fireworks.training`; install the cookbook dependencies above to upgrade.
Training requires Python 3.11+ (the setup examples use 3.12). The cookbook declares
recipe-only dependencies such as `tinker-cookbook` directly in `pyproject.toml`.

## Credentials

Set your API key via `.env` (auto-loaded by `python-dotenv`) or environment variable:

```bash
# Option A: .env file in training/
echo 'FIREWORKS_API_KEY="your-api-key"' > .env

# Option B: export
export FIREWORKS_API_KEY="your-api-key"
```

## Verify

```bash
python - <<'PYTHON'
import fireworks
from fireworks.training.sdk import FiretitanServiceClient

print(f"Fireworks SDK {fireworks.__version__}: {FiretitanServiceClient.__name__} is available")
PYTHON
python -c "import training.recipes.rl_loop, training.recipes.dpo_loop; print('Recipes OK')"
```

The recipe import check is intentional: a clean base install should run the
standard DPO/RL recipes without optional example-only packages such as
`eval-protocol`. Install the `dev` extra only when running tests or
eval-protocol examples.

## Dev dependencies (tests, coverage)

```bash
uv pip install -e ".[dev]"   # or: python -m pip install -e ".[dev]"
python -m pytest tests/
```

## Upgrading the SDK

The required SDK version is pinned in `training/pyproject.toml`. To upgrade:

```bash
cd cookbook/training
uv pip install --upgrade -e .
```

Then verify the installed version satisfies the pin:

```bash
grep 'fireworks-ai\[training\]' training/pyproject.toml
pip show fireworks-ai | grep Version
```
