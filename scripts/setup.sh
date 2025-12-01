#!/bin/bash

# Get API key from argument, fallback to environment variable
API_KEY=${1:-$WANDB_API_KEY}

if [ -z "$API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY not set. Pass as argument or export it."
    exit 1
fi

echo "=== Installing requirements ==="
pip install -r requirements.txt

echo "=== Logging into Weights & Biases ==="
wandb login "$API_KEY"

echo "=== Done ==="
