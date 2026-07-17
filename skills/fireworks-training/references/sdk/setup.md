# Environment Setup

## Install

```bash
git clone https://github.com/fw-ai/cookbook.git
cd cookbook/training

# Option A: conda
conda create -n cookbook python=3.12 -y && conda activate cookbook
python -m pip install --pre -e .

# Option B: uv
uv venv --python 3.12 && source .venv/bin/activate
uv pip install --pre -e .
```

`--pre` is required — the SDK (`fireworks-ai[training]`) is a
prerelease. The cookbook declares recipe-only dependencies such as
`tinker-cookbook` directly in `pyproject.toml`.

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
python -c "import fireworks.training.sdk; print('SDK OK')"
python -c "import training.recipes.rl_loop, training.recipes.dpo_loop; print('Recipes OK')"
```

The recipe import check is intentional: a clean base install should run the
standard DPO/RL recipes without optional example-only packages such as
`eval-protocol`. Install the `dev` extra only when running tests or
eval-protocol examples.

## Dev dependencies (tests, coverage)

```bash
uv pip install --pre -e ".[dev]"   # or: python -m pip install --pre -e ".[dev]"
python -m pytest tests/
```

## Upgrading the SDK

The required SDK version is pinned in `training/pyproject.toml`. To upgrade:

```bash
uv pip install --pre --upgrade "fireworks-ai[training]"
```

Then verify the installed version satisfies the pin:

```bash
grep 'fireworks-ai\[training\]' training/pyproject.toml
pip show fireworks-ai | grep Version
```
