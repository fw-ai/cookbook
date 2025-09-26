#!/bin/bash

# Exit on any error
set -e

# 3. Create a virtual environment if not already created
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Creating a new virtual environment..."
    uv venv --python 3.11
else
    echo "Virtual environment already exists."
fi