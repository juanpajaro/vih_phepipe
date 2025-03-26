#!/usr/bin/env bash

# This script checks if conda is properly installed and configured:
# 1. Loads environment variables from .env file
# 2. Verifies conda binaries exist in the cache directory
# 3. Adds conda to PATH if needed
# 4. Exits with status code 0 if everything is OK, 1 if there are issues

export $(cat .env | sed -e /^$/d -e /^#/d | xargs --null)

# Verifies if conda files are installed
if [ ! -d "$CACHE_DIR/conda/bin" ]; then
    echo "Error: Conda is not installed in '$CACHE_DIR/conda/bin'"
    echo "       Please re install it using '/bin/bash scripts/venv-setup-conda.sh'"
    exit 1
fi

if ! command -v conda &> /dev/null; then
    PATH=$PATH:$CACHE_DIR/conda/bin
fi

exit 0
