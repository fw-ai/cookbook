#!/usr/bin/env bash
set -euo pipefail

# Change to the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Install uv if not present
if ! command -v uv >/dev/null 2>&1; then
  echo "[info] 'uv' not found. Installing via Astral..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Ensure typical install location is on PATH for this session
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[info] Using uv at: $(command -v uv)"

# Create the virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "[info] Creating .venv with Python 3.12"
  uv venv --python 3.12
fi

# Activate the environment
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[info] Installing Python dependencies (sagemaker, boto3)"
uv pip install --upgrade pip
uv pip install sagemaker boto3

echo "[info] Done. To activate later: source .venv/bin/activate"


