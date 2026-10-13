#!/bin/bash

REQUIRED_PYTHON="3.9"

# Check Python 3.9 explicitly
if ! command -v python3.9 &> /dev/null; then
    echo "Python $REQUIRED_PYTHON is required but not installed."
    echo "Please install Python $REQUIRED_PYTHON before continuing."
    exit 1
fi

PYTHON_VERSION=$(python3.9 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)

if [ "$PYTHON_VERSION" != "$REQUIRED_PYTHON" ]; then
    echo "Python $REQUIRED_PYTHON required, found: $PYTHON_VERSION"
    exit 1
fi

echo "Python $PYTHON_VERSION detected - OK"

# Create venv with Python 3.9 if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.9..."
    python3.9 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment successfully configured with CUDA support."