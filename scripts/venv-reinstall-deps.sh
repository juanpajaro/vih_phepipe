#!/usr/bin/env bash

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

echo "Reinstalling dependencies (forced)"
conda env create --file environment.yml --force
