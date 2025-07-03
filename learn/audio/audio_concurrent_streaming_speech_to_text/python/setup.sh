#!/bin/bash
set -e

echo "🚀 Setting up Fireworks Audio Transcription project..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📥 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Source the cargo environment to make uv available
    if [ -f "$HOME/.cargo/env" ]; then
        source $HOME/.cargo/env
    fi

    # Verify installation worked
    if ! command -v uv &> /dev/null; then
        echo "❌ Failed to install uv. Please install manually:"
        echo "https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
fi

echo "✅ uv found"

# Create virtual environment
echo "📦 Creating virtual environment..."
if ! uv venv; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Install dependencies
echo "📥 Installing dependencies..."
if ! uv pip install -r requirements.txt; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Create .env file and add your Fireworks API key"
echo "2. Activate the environment: source .venv/bin/activate"
echo "3. Run the transcription: python main.py"
echo ""
echo "💡 To deactivate later: deactivate"