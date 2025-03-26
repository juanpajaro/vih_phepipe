#!/usr/bin/env bash

# This script sets up a conda virtual environment by:
# 1. Reading environment variables from .env file
# 2. Creating necessary cache directories
# 3. Downloading and installing miniconda if not present
# 4. Adding conda to PATH if needed
# 5. Creating the virtual environment using venv-create.sh

echo "Reading environment values"
cp .env.template .env

export $(cat .env | sed -e /^$/d -e /^#/d | xargs --null)

# Creates cache directory if doesn't exist
if [ ! -d "$CACHE_DIR" ]; then
    mkdir -p $CACHE_DIR
fi

# Creates conda directory if doesn't exist
if [ ! -d "$CACHE_DIR/conda" ]; then
    mkdir -p $CACHE_DIR/conda
fi

if [ -f "$CACHE_DIR/miniconda.sh" ]; then
    rm $CACHE_DIR/miniconda.sh
fi

echo ""
echo "Downloading installer from $CONDA_INSTALLER_REMOTE_PATH"

wget -x -q $CONDA_INSTALLER_REMOTE_PATH -O $CACHE_DIR/miniconda.sh && \
    /bin/bash $CACHE_DIR/miniconda.sh -b -p $CACHE_DIR/conda -u && \
    rm $CACHE_DIR/miniconda.sh

# Verifies if conda is available
if ! command -v conda &> /dev/null; then
    PATH=$PATH:$CACHE_DIR/conda/bin
    echo "Conda installed successfully"
fi

/bin/bash scripts/venv-create.sh
