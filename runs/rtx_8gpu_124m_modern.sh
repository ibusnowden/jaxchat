#!/usr/bin/env bash
#SBATCH --job-name=jaxchat-124m-modern
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --output=/project/inniang/jaxchat/slurm-%A.out
#SBATCH --error=/project/inniang/jaxchat/slurm-%A.err
#SBATCH --export=ALL
#SBATCH --exclusive

# ============================================================================
# jaxchat 124M Modernized — 8×RTX 6000
# All architectural upgrades enabled for best convergence.
# 
# Active features:
#   architecture: RoPE, QK-Norm, ReLU², RMSNorm, DeepNorm init, GQA
#   optimizer:    MuonAdamW, WSD schedule, grad clipping, z-loss
#   attention:    Long-short hybrid, sliding window, FA3/Pallas/ring
#   residual:     Embedding→every block skip, block 3→6→9 skips, value-path
#   logit head:   Tanh softcap, QK-Norm, z-loss
#   weight tying: Delayed untying at 2/3 training
#   data:         Cross-document loss masking
#   tokens:       Bigram hash embedding
#   compiler:     XLA GPU flags, AOT compilation, persistent cache
# ============================================================================

set -euo pipefail
cd /project/inniang/jaxchat

export D4_ROOT="${D4_ROOT:-/project/inniang/jaxchat/data/124m_rtx_run}"
export WANDB_PROJECT="${WANDB_PROJECT:-jaxchat}"
export WANDB_DIR="${WANDB_DIR:-/project/inniang/jaxchat/logs/wandb}"
# Compute nodes have no/flaky internet -> online wandb.init() hangs in a retry loop.
export WANDB_MODE="${WANDB_MODE:-offline}"

# ---- GPU + JAX Setup ----
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false \
  --xla_gpu_enable_cublaslt=True \
  --xla_gpu_autotune_level=4"

mkdir -p "$WANDB_DIR" "$D4_ROOT"

unset VIRTUAL_ENV
export OMP_NUM_THREADS=1

# ---- Paths ----
# Re-tokenized 32k-BPE FineWeb shards (data/retokenize_bins.py).  The old per-run
# fineweb32k dir held symlinks to GPT-2-tokenized shards -> immediate loss=nan.
DATA_DIR="/project/inniang/jaxchat/data/fineweb32k_real"
TOKENIZER_DIR="$DATA_DIR"
BASE_RUN="${D4_ROOT}/runs/modern-base"
mkdir -p "$BASE_RUN"

# ---- Setup ----
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync

PY="uv run python -u"

echo "=========================================="
echo "jaxchat 124M Modernized — 8×RTX 6000"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPUs: $(nvidia-smi -L | wc -l)"
echo "=========================================="

# ============================================================================
# Stage 1+2: 32K tokenizer + re-tokenized FineWeb-Edu data
# ============================================================================
echo ""
echo "=== Stage 1+2/3: 32K tokenizer + re-tokenized FineWeb-Edu data ==="
if [ ! -f "${DATA_DIR}/fineweb_val_000000.bin" ] || ! compgen -G "${DATA_DIR}/fineweb_train_*.bin" >/dev/null 2>&1; then
  echo "[build] re-tokenizing GPT-2 shards -> 32K BPE into ${DATA_DIR} ..."
  $PY -m data.retokenize_bins \
    --source-dir /project/inniang/jaxchat/data/fineweb10B \
    --output-dir "${DATA_DIR}" \
    --tokenizer-json /project/inniang/jaxchat/data/fineweb32k/tokenizer.json \
    --seq-len 1024 --pack-mode concat --copy-tokenizer
fi
echo "[ok] data at ${DATA_DIR} ($(ls "${DATA_DIR}"/fineweb_train_*.bin | wc -l) train shards)"

# ============================================================================
# Stage 3: Base Pretraining — ALL MODERN FEATURES ENABLED
# ============================================================================
echo ""
echo "=== Stage 3/3: Base Pretraining (124M) [MODERN] ==="
echo ""
echo "  Architecture:  depth=8, d_model=512, n_heads=4, n_kv_heads=2 (GQA)"
echo "  Init:          DeepNorm"
echo "  Optimizer:     MuonAdamW + WSD schedule"
echo "  Attention:     Long-short hybrid + sliding window"
echo "  Skip conns:    Layer 2→5, 5→7"
echo "  Logit head:    Tanh softcap + QK-Norm + z-loss"
echo "  Weight tying:  Delayed (untie at 2/3)"
echo "  Tokens:        Bigram hash embedding"
echo "  Data:          Cross-document loss masking"
echo "  Grad clip:     max_norm=1.0"
echo "  Seq warmup:    512→1024 over 500 steps"
echo ""

$PY -m scripts.base_train \
  --preset 124m-modern \
  --run-dir "$BASE_RUN" \
  --resume

echo ""
echo "=========================================="
echo "✅ Modern 124M pretraining complete!"
echo "  Checkpoints: $BASE_RUN"
echo "  Wandb:       $WANDB_PROJECT"
echo "  Date:        $(date)"
echo "=========================================="

# Print key metrics for quick inspection
echo ""
echo "=== Final Checkpoint Info ==="
ls -lh "${BASE_RUN}/base/"state_step*.pkl 2>/dev/null | tail -1 || echo "No checkpoints found"

echo ""
echo "=== To run eval ==="
echo "  uv run python -m scripts.base_eval --run-dir $BASE_RUN"
echo ""
echo "=== Next steps ==="
echo "  1. Check wandb for loss/val_bpb curves"
echo "  2. Run ablations: compare lr_schedule=linear, init_style=default, etc."
echo "  3. Proceed to SFT + RL pipeline"
