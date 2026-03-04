#!/usr/bin/env bash
set -e

VENV_DIR=".venv"

# Check for uv, install if missing
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo ""
    echo "uv installed. You may need to restart your shell or run:"
    echo '  source "$HOME/.local/bin/env"'
    echo ""
    # Source it for this session
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Using uv: $(uv --version)"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    uv venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Install the Fireworks SDK with training extras (pre-release required)
echo "Installing fireworks-ai[training] (pre-release)..."
uv pip install --pre 'fireworks-ai[training]'

# Install the cookbook and remaining dependencies
echo "Installing training-cookbook..."
uv pip install -e ".[dev]"

echo ""
echo "Setup complete! Activate your environment with:"
echo ""
echo "  source .venv/bin/activate"
echo ""
echo "(Note: use 'source', not './' — activate must be sourced, not executed directly)"