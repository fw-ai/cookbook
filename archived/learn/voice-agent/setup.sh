#!/bin/bash

# Setup script for Fireworks Voice Agent development using uv

echo "🚀 Setting up Fireworks Voice Agent development environment with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

echo "✅ uv found"

# Create virtual environment
echo "📦 Creating virtual environment..."
uv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
uv pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Set your API key: by adding a .env file with FIREWORKS_API_KEY='your_key_here'"
echo "2. Activate the environment: source .venv/bin/activate"
echo "3. Run the voice agent: python main.py"
