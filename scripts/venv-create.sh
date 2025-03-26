#!/usr/bin/env bash

# This script creates a conda virtual environment by:
# 1. Running environment checks
# 2. Reading environment variables from .env file
# 3. Setting up conda in PATH if needed
# 4. Creating the conda environment from environment.yml

/bin/bash scripts/venv-check.sh

# Verify if check fails
if [ $? -ne 0 ]; then
    exit 1
fi

echo "Reading environment values"
cp .env.template .env

export $(cat .env | sed -e /^$/d -e /^#/d | xargs --null)

# Verifies if conda is available
if ! command -v conda &> /dev/null; then
    PATH=$PATH:$CACHE_DIR/conda/bin
fi

echo "Creating conda environment"
conda env create -f environment.yml
