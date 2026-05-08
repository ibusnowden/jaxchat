#!/usr/bin/env bash
#SBATCH --job-name=jaxchat-1p384b-depth24
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=/project/inniang/jaxchat/slurm-%A.out
#SBATCH --error=/project/inniang/jaxchat/slurm-%A.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR="/project/inniang/jaxchat"
cd "$PROJECT_DIR"

# Clear any inherited venv so uv uses the project-local .venv
unset VIRTUAL_ENV
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false"

# Sync dependencies
uv sync

SPEEDRUN_ROOT="$HOME/.cache/jaxchat/speedrun"
TOKENIZER_DIR="${SPEEDRUN_ROOT}/tokenizer"
DATA_DIR="${SPEEDRUN_ROOT}/data/fineweb32k"
RUN_DIR="${SPEEDRUN_ROOT}/base-depth24"

if ! compgen -G "${DATA_DIR}/fineweb_train_*.bin" > /dev/null || [[ ! -f "${DATA_DIR}/fineweb_val_000000.bin" ]] || [[ ! -f "${TOKENIZER_DIR}/tokenizer.json" ]]; then
  echo "Missing prebuilt speedrun assets." >&2
  echo "Expected tokenizer at ${TOKENIZER_DIR}/tokenizer.json" >&2
  echo "Expected packed bins under ${DATA_DIR}" >&2
  echo "Run bash jobs/speedrun.sh first." >&2
  exit 1
fi

echo "Launching preset: 1p384b-depth24"
uv run python -u -m training.train_base \
  --preset 1p384b-depth24 \
  --input-bin "${DATA_DIR}/fineweb_train_*.bin" \
  --input-val-bin "${DATA_DIR}/fineweb_val_000000.bin" \
  --tokenizer-json "${TOKENIZER_DIR}/tokenizer.json" \
  --run-dir "${RUN_DIR}"
