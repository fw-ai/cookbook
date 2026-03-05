#!/bin/bash

# Exit on any error
set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Check if 'uv' is installed
if ! command_exists uv; then
    echo "'uv' is not installed. Installing 'uv'..."
    # Install 'uv'
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.cargo/env
else
    echo "'uv' is already installed."
fi