#!/bin/bash

# This script sets up the Python virtual environment and installs dependencies.

# Check if we are already in a virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Already in a virtual environment. Skipping venv creation."
else
    echo "Creating virtual environment..."
    python3 -m venv .venv

    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete. If you weren't in a virtual environment, you can activate the new one by running:"
echo "source .venv/bin/activate"