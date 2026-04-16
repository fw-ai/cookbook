# Environment Setup

## Install

```bash
git clone https://github.com/fw-ai/cookbook.git
cd cookbook/training

# Option A: conda
conda create -n cookbook python=3.12 -y && conda activate cookbook
pip install --pre -e .

# Option B: uv
uv venv --python 3.12 && source .venv/bin/activate
uv pip install --pre -e .
```

`--pre` is required — the SDK (`fireworks-ai[training]`) is a prerelease. All dependencies (including `tinker-cookbook`) are pulled in automatically via `pyproject.toml`.

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
python -c "from fireworks.training.sdk import TrainerJobManager; print('SDK OK')"
python -c "from training.utils.config import InfraConfig; print('Cookbook OK')"
```

## Dev dependencies (tests, coverage)

```bash
pip install --pre -e ".[dev]"
pytest tests/
```

## Upgrading the SDK

The required SDK version is pinned in `training/pyproject.toml`. To upgrade:

```bash
pip install --pre --upgrade "fireworks-ai[training]"
```

Then verify the installed version satisfies the pin:

```bash
grep 'fireworks-ai\[training\]' training/pyproject.toml
pip show fireworks-ai | grep Version
```
