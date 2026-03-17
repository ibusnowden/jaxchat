#!/usr/bin/env bash
#SBATCH --job-name=tinystories-124m
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/inniang/jaxchat/slurm-%A.out
#SBATCH --error=/project/inniang/jaxchat/slurm-%A.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR="/project/inniang/jaxchat"
cd "$PROJECT_DIR"

# Clear any inherited venv so uv uses the project-local .venv
unset VIRTUAL_ENV
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Sync dependencies and run via uv
uv sync

uv run python -u train.py --resume
