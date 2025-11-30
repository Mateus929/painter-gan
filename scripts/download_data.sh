#!/bin/bash

set -e

COMPETITION="gan-getting-started" 
DATA_DIR="data/"
KEY_FILE="monet_jpg"  

usage() {
    echo "Usage: $0 -f /path/to/your/kaggle.json"
    echo ""
    echo "This script downloads and prepares the 'I’m Something of a Painter Myself' data from Kaggle."
    echo ""
    echo "  -f    Path to your kaggle.json API credentials file."
    echo ""
    exit 1
}

if [ "$1" != "-f" ] || [ -z "$2" ]; then
    usage
fi

KAGGLE_JSON_PATH="$2"

if [ ! -f "$KAGGLE_JSON_PATH" ]; then
    echo "Error: kaggle.json not found at '$KAGGLE_JSON_PATH'"
    exit 1
fi

echo "Setting up Kaggle credentials..."
mkdir -p ~/.kaggle
cp "$KAGGLE_JSON_PATH" ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json 

echo "Ensuring data directory exists at '$DATA_DIR'..."
mkdir -p "$DATA_DIR"

if [ -f "$DATA_DIR/$KEY_FILE" ] || [ -d "$DATA_DIR/$KEY_FILE" ]; then
    echo "Data already exists at '$DATA_DIR'. Skipping download and extraction."
else
    echo "Downloading data from Kaggle for competition: '$COMPETITION'..."
    kaggle competitions download -c "$COMPETITION" -p "$DATA_DIR"

    echo "Unzipping files..."
    cd "$DATA_DIR"

    for file in *.zip; do
        if [ -f "$file" ]; then
            unzip -o "$file"
            rm "$file"
        fi
    done
    cd ..
    echo "Data downloaded and extracted successfully to '$DATA_DIR'."
fi
