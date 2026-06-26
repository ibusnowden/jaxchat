#!/usr/bin/env bash
#SBATCH --job-name=chinchilla-sweep
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --output=/project/inniang/jaxchat/slurm-chinchilla-%A_%a.out
#SBATCH --error=/project/inniang/jaxchat/slurm-chinchilla-%A_%a.err
#SBATCH --export=ALL
#SBATCH --exclusive
#SBATCH --array=0-29%1

# ============================================================================
# jaxchat Chinchilla sweep — 8×RTX 6000
#
# 30 tasks driven by scripts/chinchilla_grid.py:
#   tasks  0–23  : IsoFLOP sweep (4 budgets × 6 depths) — Plots 1 + 4
#   tasks 24–29  : depth miniseries d10–d20 at 1.31B tok — Plots 2 + 3
# The fit/plot pipeline in scripts/fit_chinchilla.py consumes both groups.
#
# Total compute ~3.2e19 FLOPs (~120 GPU-node-hours).  --array=...%1 keeps runs
# sequential on a single node; bump the trailing percent if you have multiple
# RTX 6000 nodes free.  All IsoFLOP rows stay within one pass of the 2.94B-token
# FineWeb pool.
#
# Submit with:
#   sbatch /project/inniang/jaxchat/runs/rtx_8gpu_chinchilla_isoflop.sh
# Inspect the grid before submitting:
#   uv run python -m scripts.chinchilla_grid --print-grid
# Submit only the IsoFLOP set (tasks 0-23):
#   sbatch --array=0-23%1 /project/inniang/jaxchat/runs/rtx_8gpu_chinchilla_isoflop.sh
# Submit only the miniseries (tasks 24-29):
#   sbatch --array=24-29%1 /project/inniang/jaxchat/runs/rtx_8gpu_chinchilla_isoflop.sh
# Resubmit just one task (e.g. after a crash):
#   sbatch --array=7 /project/inniang/jaxchat/runs/rtx_8gpu_chinchilla_isoflop.sh
# ============================================================================

set -euo pipefail
cd /project/inniang/jaxchat

export D4_ROOT="${D4_ROOT:-/project/inniang/jaxchat/data/124m_rtx_run}"
export WANDB_PROJECT="${WANDB_PROJECT:-jaxchat}"
export WANDB_DIR="${WANDB_DIR:-/project/inniang/jaxchat/logs/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${D4_ROOT}/uv-cache}"
export CORE_N="${CORE_N:-1000}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false \
  --xla_gpu_enable_cublaslt=True \
  --xla_gpu_autotune_level=4"

mkdir -p "$WANDB_DIR" "$D4_ROOT" "$UV_CACHE_DIR"
unset VIRTUAL_ENV
export OMP_NUM_THREADS=1

DATA_DIR_29="/project/inniang/jaxchat/data/fineweb32k_real_29"
DATA_DIR_9="/project/inniang/jaxchat/data/fineweb32k_real"
if [ -f "${DATA_DIR_29}/fineweb_val_000000.bin" ] && compgen -G "${DATA_DIR_29}/fineweb_train_*.bin" >/dev/null 2>&1; then
  DATA_DIR="$DATA_DIR_29"
elif [ -f "${DATA_DIR_9}/fineweb_val_000000.bin" ] && compgen -G "${DATA_DIR_9}/fineweb_train_*.bin" >/dev/null 2>&1; then
  DATA_DIR="$DATA_DIR_9"
else
  echo "ERROR: no re-tokenized data found at ${DATA_DIR_29} or ${DATA_DIR_9}." >&2
  exit 1
fi
echo "Using data dir: $DATA_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not on PATH; install/sync dependencies before submitting this SLURM job." >&2
  exit 1
fi
if [ ! -d ".venv" ] || [ "${SYNC_DEPS:-0}" = "1" ]; then
  uv sync --frozen
fi
PY="uv run --no-sync python -u"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Pull the per-task knobs from the grid (single source of truth).
GRID_ENV=$($PY -m scripts.chinchilla_grid --task-id "$TASK_ID" --shell)
eval "$GRID_ENV"

BASE_RUN="${D4_ROOT}/runs/chinchilla/${RUN_NAME}"
mkdir -p "$BASE_RUN"

# Per-task group (chinchilla-isoflop-v2 vs chinchilla-miniseries-v2) comes
# straight from chinchilla_grid.py so all sweep rows show up cleanly in W&B.
export WANDB_RUN_GROUP="${WANDB_GROUP_TASK}"
export WANDB_NAME="${RUN_NAME}"

echo "=========================================="
echo "Chinchilla task ${TASK_ID} (${KIND}): ${RUN_NAME}"
echo "  C_target = ${FLOP_BUDGET}   depth = ${DEPTH}   params = ${PARAMS}"
echo "  target_train_tokens = ${TARGET_TRAIN_TOKENS}   iters = ${N_TRAIN_ITERS}"
echo "  actual_train_tokens = ${ACTUAL_TRAIN_TOKENS}   C_actual = ${ACTUAL_FLOPS}"
echo "  run_dir = ${BASE_RUN}"
echo "  W&B group/name = ${WANDB_RUN_GROUP} / ${WANDB_NAME}"
echo "  Date: $(date)  Host: $(hostname)  GPUs: $(nvidia-smi -L | wc -l)"
echo "=========================================="

$PY -m scripts.base_train \
  --preset 124m-modern \
  --run-dir "$BASE_RUN" \
  --input-bin "${DATA_DIR}/fineweb_train_*.bin" \
  --input-val-bin "${DATA_DIR}/fineweb_val_000000.bin" \
  --tokenizer-json "${DATA_DIR}/tokenizer.json" \
  --config-override depth="$DEPTH" \
  --config-override n_kv_heads="$N_KV_HEADS" \
  --config-override target_train_tokens="$TARGET_TRAIN_TOKENS" \
  --config-override n_train_iters="$N_TRAIN_ITERS" \
  --config-override skip_connections="$SKIP_CONNECTIONS" \
  --config-override untie_at_step=-1 \
  --resume

echo ""
echo "=== eval (val_bpb + CORE subset, n=${CORE_N}/task) ==="
$PY -m scripts.base_eval --run-dir "$BASE_RUN" --core-n "$CORE_N" --skip-generation \
  --tokenizer-json "${DATA_DIR}/tokenizer.json" \
  || echo "(base_eval skipped/partial — fit script will fall back to log.txt)"

echo "=========================================="
echo "✅ chinchilla task ${TASK_ID} (${RUN_NAME}) complete — $(date)"
ls -lh "${BASE_RUN}/base/"state_step*.pkl 2>/dev/null | tail -1 || true
