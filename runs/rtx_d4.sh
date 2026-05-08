#!/usr/bin/env bash
#SBATCH --job-name=jaxchat-d4-rtx
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=/project/inniang/jaxchat/slurm-%A.out
#SBATCH --error=/project/inniang/jaxchat/slurm-%A.err
#SBATCH --export=ALL

set -euo pipefail
cd /project/inniang/jaxchat

# Run artifacts go to a separate root so this run does not clobber the H100 run.
export D4_ROOT="${D4_ROOT:-/project/inniang/jaxchat/data/d4_speedrun_rtx}"
export WANDB_PROJECT="${WANDB_PROJECT:-jaxchat}"
export WANDB_DIR="${WANDB_DIR:-/project/inniang/jaxchat/logs/wandb}"
mkdir -p "$WANDB_DIR" "$D4_ROOT"

exec bash /project/inniang/jaxchat/runs/speedrun_d4.sh
