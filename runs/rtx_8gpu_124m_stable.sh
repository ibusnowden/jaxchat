#!/usr/bin/env bash
#SBATCH --job-name=jaxchat-124m-stable
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --output=/project/inniang/jaxchat/slurm-stable-%A.out
#SBATCH --error=/project/inniang/jaxchat/slurm-stable-%A.err
#SBATCH --export=ALL
#SBATCH --exclusive

# ============================================================================
# Stable baseline — proven working features only for 8×RTX 6000
# ============================================================================

set -euo pipefail
cd /project/inniang/jaxchat

export D4_ROOT="${D4_ROOT:-/project/inniang/jaxchat/data/124m_rtx_run}"
export WANDB_PROJECT="${WANDB_PROJECT:-jaxchat}"
export WANDB_DIR="${WANDB_DIR:-/project/inniang/jaxchat/logs/wandb}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false"

mkdir -p "$WANDB_DIR" "$D4_ROOT"
unset VIRTUAL_ENV
export OMP_NUM_THREADS=1

TOKENIZER_DIR="${D4_ROOT}/tokenizer"
DATA_DIR="${D4_ROOT}/fineweb32k"
BASE_RUN="${D4_ROOT}/runs/stable-base"
mkdir -p "$TOKENIZER_DIR" "$DATA_DIR" "$BASE_RUN"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync

PY="uv run python -u"

echo "=== Stage 1/3: Tokenizer ==="
if [ -f "${TOKENIZER_DIR}/tokenizer.json" ]; then
  echo "[skip]"
else
  cp /project/inniang/jaxchat/data/fineweb32k/tokenizer.json "${TOKENIZER_DIR}/tokenizer.json"
fi

echo "=== Stage 2/3: Data ==="
if compgen -G "${DATA_DIR}/fineweb_train_*.bin" >/dev/null 2>&1; then
  echo "[skip]"
else
  ln -sf /project/inniang/jaxchat/data/fineweb10B/*.bin "${DATA_DIR}/" 2>/dev/null || true
  cp /project/inniang/jaxchat/data/fineweb32k/*.json "${DATA_DIR}/" 2>/dev/null || true
fi

echo "=== Stage 3/3: Base Pretraining (124M) [STABLE] ==="
echo ""
echo "  Features: WSD schedule, DeepNorm init, grad clipping, z-loss,"
echo "            delayed weight tying, seq warmup, GQA, long-short attn,"
echo "            bigram hash, cross-doc mask, skip connections"
echo ""

$PY -m scripts.base_train \
  --preset 124m-modern \
  --input-bin "${DATA_DIR}/fineweb_train_*.bin" \
  --input-val-bin "${DATA_DIR}/fineweb_val_000000.bin" \
  --tokenizer-json "${TOKENIZER_DIR}/tokenizer.json" \
  --run-dir "$BASE_RUN"

echo ""
echo "=========================================="
echo "Complete! Checkpoints: $BASE_RUN"
echo "Wandb: $WANDB_PROJECT"
echo "=========================================="
