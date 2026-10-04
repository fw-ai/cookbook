# Environment Setup

## Install

```bash
git clone https://github.com/fw-ai/cookbook.git
cd cookbook/training

# Option A: conda
conda create -n cookbook python=3.12 -y && conda activate cookbook
python -m pip install -e .

# Option B: uv
# If `uv` is not found, install it and load ~/.local/bin:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   source "$HOME/.local/bin/env"
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .
```

The cookbook requires `fireworks-ai[training]>=1.2.11,<2`, available as a stable
PyPI release. `--pre` is not required. The legacy `0.19.20` package does not
contain `fireworks.training`; install the cookbook dependencies above to upgrade.
Training requires Python 3.11+ (the setup examples use 3.12). The cookbook declares
its runtime and recipe dependencies directly in `pyproject.toml`.

## Credentials

Create an account-scoped key at
[app.fireworks.ai/settings/users/api-keys](https://app.fireworks.ai/settings/users/api-keys).
The training examples load `FIREWORKS_API_KEY` from `training/.env` through
`python-dotenv`. `.env` is gitignored. Type the key at the hidden prompt; do
not paste it into chat, a notebook cell, or a committed file.

Run these commands from `cookbook/training` after the install above.

### Linux or any bash shell

```bash
read -rsp "Fireworks API key: " FIREWORKS_API_KEY
echo
printf 'FIREWORKS_API_KEY=%s\n' "$FIREWORKS_API_KEY" > .env
chmod 600 .env
unset FIREWORKS_API_KEY
```

### macOS

macOS Terminal uses zsh. `uv` from the official installer lives in
`~/.local/bin`, which is not on `PATH` until you load it:

```zsh
source "$HOME/.local/bin/env"
```

Store the key in the same `training/.env` file the examples read:

```zsh
cd cookbook/training
read -rs "FIREWORKS_API_KEY?Fireworks API key: "
echo
printf 'FIREWORKS_API_KEY=%s\n' "$FIREWORKS_API_KEY" > .env
chmod 600 .env
unset FIREWORKS_API_KEY
```

To keep the key in the Mac system Keychain and load it in new Terminal
windows, save it once:

```zsh
security add-generic-password -U -a "$USER" -s FIREWORKS_API_KEY -w
```

`security` prompts for the key and does not echo it. Then add these lines to
`~/.zshrc`:

```zsh
if fw_key="$(security find-generic-password -a "$USER" -s FIREWORKS_API_KEY -w 2>/dev/null)" && [[ -n "$fw_key" ]]; then
  export FIREWORKS_API_KEY="$fw_key"
fi
unset fw_key
```

Open a new terminal. macOS may ask once whether Terminal may read that
Keychain item.

`python-dotenv` does not override a variable that is already set, even when it
is empty. If `FIREWORKS_API_KEY` exists in the shell, the cookbook uses it
instead of `training/.env`. Clear an empty or stale value with
`unset FIREWORKS_API_KEY`.

Check both sources without printing the key:

```zsh
echo "shell key length: ${#FIREWORKS_API_KEY}"
python -c 'import os; from dotenv import dotenv_values, load_dotenv; f=dotenv_values(".env").get("FIREWORKS_API_KEY") or ""; load_dotenv(".env"); print(".env key length:", len(f)); print("key loaded" if os.getenv("FIREWORKS_API_KEY") else "key missing")'
```

A shell length of `0` with `key missing` means an empty exported variable is
hiding `.env`; run `unset FIREWORKS_API_KEY`. A `.env` length of `0` means the
key was not saved; rerun the hidden prompt from `training/`.

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
