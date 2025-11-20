#!/bin/bash

# Define environment name
ENV_NAME="venv"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Python3 could not be found."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$ENV_NAME" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $ENV_NAME
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source $ENV_NAME/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
fi

echo "Setup complete. To activate the environment, run: source $ENV_NAME/bin/activate"
