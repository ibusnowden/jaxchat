#!/usr/bin/env bash
#SBATCH --job-name=jaxchat-124m-modern-long
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
# jaxchat 124M Modernized, LONG run — 8×RTX 6000
# Same config as runs/rtx_8gpu_124m_modern.sh, but ~2x the token budget:
#   n_train_iters 2605 -> 5000  (=> ~1.31B tokens, ~1.4 epochs over the
#   ~912M-token fineweb32k_real data).  WSD cool-down is half the run, so a
# longer stable phase + longer cool-down should push val_bpb below the 1.2249
# of the 683M-token modern run.
#
# NOTE: the modern preset stores already-resolved token/iter fields, so
# overriding train_token_ratio alone is a no-op; we set n_train_iters (and the
# matching target_train_tokens for the log line) directly, plus untie_at_step=-1
# so delayed untying re-derives to 2/3 of the new length.
#
# Reference numbers to beat (8×RTX 6000, 124m preset):
#   124m  (linear, no modern) : val_bpb 1.3152 @ 2352 steps  (job 71053)
#   124m-modern (683M tokens)  : val_bpb 1.2249 @ 2604 steps  (job 71056)
# ============================================================================

set -euo pipefail
cd /project/inniang/jaxchat

export D4_ROOT="${D4_ROOT:-/project/inniang/jaxchat/data/124m_rtx_run}"
export WANDB_PROJECT="${WANDB_PROJECT:-jaxchat}"
export WANDB_DIR="${WANDB_DIR:-/project/inniang/jaxchat/logs/wandb}"
# Compute nodes have no/flaky internet -> online wandb.init() hangs in a retry loop.
export WANDB_MODE="${WANDB_MODE:-offline}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false \
  --xla_gpu_enable_cublaslt=True \
  --xla_gpu_autotune_level=4"

mkdir -p "$WANDB_DIR" "$D4_ROOT"
unset VIRTUAL_ENV
export OMP_NUM_THREADS=1

DATA_DIR="/project/inniang/jaxchat/data/fineweb32k_real"
BASE_RUN="${D4_ROOT}/runs/modern-long"
mkdir -p "$BASE_RUN"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
PY="uv run python -u"

echo "=========================================="
echo "jaxchat 124M Modernized [LONG] — 8×RTX 6000"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPUs: $(nvidia-smi -L | wc -l)"
echo "=========================================="

echo ""
echo "=== Data: 32K-BPE re-tokenized FineWeb-Edu ==="
if [ ! -f "${DATA_DIR}/fineweb_val_000000.bin" ] || ! compgen -G "${DATA_DIR}/fineweb_train_*.bin" >/dev/null 2>&1; then
  echo "[build] re-tokenizing GPT-2 shards -> 32K BPE into ${DATA_DIR} ..."
  $PY -m data.retokenize_bins \
    --source-dir /project/inniang/jaxchat/data/fineweb10B \
    --output-dir "${DATA_DIR}" \
    --tokenizer-json /project/inniang/jaxchat/data/fineweb32k/tokenizer.json \
    --seq-len 1024 --pack-mode concat --copy-tokenizer
fi
echo "[ok] data at ${DATA_DIR} ($(ls "${DATA_DIR}"/fineweb_train_*.bin | wc -l) train shards)"

echo ""
echo "=== Base Pretraining (124M) [MODERN, LONG: 5000 steps ~ 1.31B tokens] ==="
$PY -m scripts.base_train \
  --preset 124m-modern \
  --run-dir "$BASE_RUN" \
  --config-override n_train_iters=5000 \
  --config-override target_train_tokens=1310720000 \
  --config-override untie_at_step=-1 \
  --resume

echo ""
echo "=========================================="
echo "✅ Modern-long 124M pretraining complete!"
echo "  Checkpoints: $BASE_RUN"
echo "  Date:        $(date)"
echo "=========================================="
ls -lh "${BASE_RUN}/base/"state_step*.pkl 2>/dev/null | tail -1 || echo "No checkpoints found"
echo ""
echo "=== To run eval ==="
echo "  uv run python -m scripts.base_eval --run-dir $BASE_RUN"
