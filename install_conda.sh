#!/usr/bin/env bash

export CONDA_ENV_NAME=drone-vo
echo $CONDA_ENV_NAME

# Remove existing environment if it exists, ignore errors if it doesn't
conda remove -n $CONDA_ENV_NAME --all -y || true

# Create the environment with Python and install packages using conda, automatically confirming
# Pulling opencv and flask from conda-forge channel
conda create -n $CONDA_ENV_NAME python=3.9 numpy matplotlib opencv flask -c conda-forge -y

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

which python
which pip

# conda activate drone-vo
