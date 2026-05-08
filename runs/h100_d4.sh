#!/usr/bin/env bash
#SBATCH --job-name=jaxchat-d4
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/project/inniang/jaxchat/slurm-%A.out
#SBATCH --error=/project/inniang/jaxchat/slurm-%A.err
#SBATCH --export=ALL

set -euo pipefail
cd /project/inniang/jaxchat

# Wandb credentials are inherited via --export=ALL.  Set WANDB_API_KEY in your
# shell before sbatch.  WANDB_PROJECT / WANDB_ENTITY can override the defaults.
export WANDB_PROJECT="${WANDB_PROJECT:-jaxchat}"
export WANDB_DIR="${WANDB_DIR:-/project/inniang/jaxchat/logs/wandb}"
mkdir -p "$WANDB_DIR"

# Run the full d4 pipeline.
exec bash /project/inniang/jaxchat/runs/speedrun_d4.sh
