#!/usr/bin/env bash
#SBATCH --job-name=core1000-eval
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --output=/project/inniang/jaxchat/slurm-core1000-%A_%a.out
#SBATCH --error=/project/inniang/jaxchat/slurm-core1000-%A_%a.err
#SBATCH --export=ALL
#SBATCH --array=0-5%1

set -euo pipefail

cd /project/inniang/jaxchat

unset VIRTUAL_ENV
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false \
  --xla_gpu_enable_cublaslt=True \
  --xla_gpu_autotune_level=4"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/project/inniang/jaxchat/data/124m_rtx_run/uv-cache}"
export WANDB_MODE="${WANDB_MODE:-offline}"

DEPTHS=(10 12 14 16 18 20)
DEPTH="${DEPTHS[${SLURM_ARRAY_TASK_ID:-0}]}"
RUN_DIR="/project/inniang/jaxchat/data/124m_rtx_run/runs/chinchilla/miniseries-v2-d${DEPTH}"
CORE_N="${CORE_N:-1000}"

DATA_DIR_29="/project/inniang/jaxchat/data/fineweb32k_real_29"
DATA_DIR_9="/project/inniang/jaxchat/data/fineweb32k_real"
if [ -f "${DATA_DIR_29}/tokenizer.json" ]; then
  TOKENIZER_JSON="${DATA_DIR_29}/tokenizer.json"
elif [ -f "${DATA_DIR_9}/tokenizer.json" ]; then
  TOKENIZER_JSON="${DATA_DIR_9}/tokenizer.json"
else
  echo "ERROR: no tokenizer.json found under ${DATA_DIR_29} or ${DATA_DIR_9}" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not on PATH." >&2
  exit 1
fi
if [ ! -d ".venv" ] || [ "${SYNC_DEPS:-0}" = "1" ]; then
  uv sync --frozen
fi
PY="uv run --no-sync python -u"

echo "=========================================="
echo "CORE-${CORE_N} eval for ${RUN_DIR}"
echo "Date: $(date)  Host: $(hostname)  GPUs: $(nvidia-smi -L | wc -l)"
echo "Tokenizer: ${TOKENIZER_JSON}"
echo "=========================================="

if [ -f "${RUN_DIR}/base_eval.json" ] && [ ! -f "${RUN_DIR}/base_eval.core200.json" ]; then
  cp "${RUN_DIR}/base_eval.json" "${RUN_DIR}/base_eval.core200.json"
fi

$PY -m scripts.base_eval \
  --run-dir "$RUN_DIR" \
  --core-n "$CORE_N" \
  --skip-generation \
  --tokenizer-json "$TOKENIZER_JSON"

echo "CORE-${CORE_N} eval complete for depth ${DEPTH} — $(date)"
