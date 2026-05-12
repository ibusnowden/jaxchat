#!/usr/bin/env bash
# Quick wiring/sanity smoke for the 124m presets on a single RTX 6000.
# Goal: confirm loss is finite from step 0 and val_bpb is finite + decreasing on the
# re-tokenized 32k data (catches the GPT-2/32k vocab mismatch and value-embed changes).
#SBATCH --job-name=jaxchat-124m-smoke
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/project/inniang/jaxchat/slurm-smoke-%A.out
#SBATCH --error=/project/inniang/jaxchat/slurm-smoke-%A.err
#SBATCH --export=ALL

set -euo pipefail
cd /project/inniang/jaxchat

export WANDB_PROJECT="${WANDB_PROJECT:-jaxchat}"
export WANDB_DIR="${WANDB_DIR:-/project/inniang/jaxchat/logs/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$WANDB_DIR"
unset VIRTUAL_ENV
export OMP_NUM_THREADS=1

[ -d ".venv" ] || uv venv
uv sync >/dev/null 2>&1 || true
PY="uv run python -u"

SMOKE_ITERS="${SMOKE_ITERS:-250}"
RUN_ROOT="${RUN_ROOT:-/project/inniang/jaxchat/data/124m_smoke}"
mkdir -p "$RUN_ROOT"

for preset in 124m 124m-modern; do
  echo "================ smoke: preset=${preset} iters=${SMOKE_ITERS} ================"
  $PY -m scripts.base_train \
    --preset "${preset}" \
    --run-dir "${RUN_ROOT}/${preset}" \
    --smoke-iters "${SMOKE_ITERS}"
done
echo "================ smoke done ================"
