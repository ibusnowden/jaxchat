#!/usr/bin/env bash
#SBATCH --job-name=0p5b-chinchilla
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --output=/project/inniang/jaxchat/slurm-0p5b-chinchilla-%A.out
#SBATCH --error=/project/inniang/jaxchat/slurm-0p5b-chinchilla-%A.err
#SBATCH --export=ALL
#SBATCH --exclusive

# ============================================================================
# jaxchat 0.5B base pretraining — REAL Chinchilla-optimal run — 8×RTX 6000
#
# Architecture is identical to chinchilla sweep run `miniseries-v2-d20`
#   (124m-modern preset, depth=20, d_model=1280, GQA kv=2,
#    skip_connections=((2,5),(5,7)), 529,531,562 params).
# The miniseries trained it to only 1.31B tok (D/N=2.5, undertrained) ->
#   val_bpb 0.8684, worse than the d16 winner (0.7626).
#
# This run trains to the fit-implied Chinchilla optimum for N=529.5M:
#   IsoFLOP fit: N* ∝ C^0.500, D* ∝ C^0.500  =>  D* ≈ 9.5·N
#   D* = 5,018,080,160 target tokens
#   schedule: 19,143 steps × 262,144 tok/step = 5,018,222,592 tokens (D/N≈9.48)
# Wraps the 2.94B-token local FineWeb pool ~1.7x.
#
# ETA ~46h on 8×RTX 6000 (8.29 s/step measured on miniseries-v2-d20).
# train_loop checkpoints every 200 steps; --resume makes it requeue-safe.
# A fresh run dir is used so --resume does NOT pick up the e2e smoke
# checkpoint (data/0p5b_e2e/runs/base/base/state_step000049.pkl).
#
# Submit:   sbatch /project/inniang/jaxchat/runs/rtx_8gpu_0p5b_base_chinchilla.sh
# Resume after a crash: just re-submit the same line (picks up newest ckpt).
# After it finishes, chain SFT+RL with the existing e2e driver:
#   sbatch --export=ALL,SKIP_BASE=1,BASE_RUN=/project/inniang/jaxchat/data/0p5b_e2e/runs/base_chinchilla \
#     /project/inniang/jaxchat/runs/rtx_8gpu_0p5b_e2e.sh
# ============================================================================

set -euo pipefail

PROJECT_DIR="/project/inniang/jaxchat"
cd "$PROJECT_DIR"

unset VIRTUAL_ENV
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false \
  --xla_gpu_enable_cublaslt=True \
  --xla_gpu_autotune_level=4"

export WANDB_PROJECT="${WANDB_PROJECT:-jaxchat}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_DIR}/logs/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"

BASE_RUN="${BASE_RUN:-${PROJECT_DIR}/data/0p5b_e2e/runs/base_chinchilla}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_DIR}/data/0p5b_e2e/uv-cache}"
export UV_CACHE_DIR

BASE_DEPTH="${BASE_DEPTH:-20}"
BASE_TARGET_TOKENS="${BASE_TARGET_TOKENS:-5018080160}"
BASE_N_ITERS="${BASE_N_ITERS:-19143}"
CORE_N="${CORE_N:-1000}"

mkdir -p "$WANDB_DIR" "$UV_CACHE_DIR" "$BASE_RUN"

DATA_DIR_29="${PROJECT_DIR}/data/fineweb32k_real_29"
DATA_DIR_9="${PROJECT_DIR}/data/fineweb32k_real"
if [ -f "${DATA_DIR_29}/fineweb_val_000000.bin" ] && compgen -G "${DATA_DIR_29}/fineweb_train_*.bin" >/dev/null 2>&1; then
  DATA_DIR="$DATA_DIR_29"
elif [ -f "${DATA_DIR_9}/fineweb_val_000000.bin" ] && compgen -G "${DATA_DIR_9}/fineweb_train_*.bin" >/dev/null 2>&1; then
  DATA_DIR="$DATA_DIR_9"
else
  echo "ERROR: no re-tokenized data found at ${DATA_DIR_29} or ${DATA_DIR_9}." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not on PATH; install/sync dependencies before submitting this SLURM job." >&2
  exit 1
fi
if [ ! -d ".venv" ] || [ "${SYNC_DEPS:-0}" = "1" ]; then
  uv sync --frozen
fi
PY="uv run --no-sync python -u"

export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-0p5b-chinchilla}"
export WANDB_NAME="${WANDB_NAME:-0p5b-base-d${BASE_DEPTH}-chinchilla-5p0b}"

echo "=========================================="
echo "jaxchat 0.5B base — REAL Chinchilla run — 8×RTX 6000"
echo "Date: $(date)  Host: $(hostname)  GPUs: $(nvidia-smi -L | wc -l)"
echo "DATA_DIR:  $DATA_DIR"
echo "BASE_RUN:  $BASE_RUN"
echo "Config:    preset=124m-modern depth=${BASE_DEPTH} params=529,531,562"
echo "Schedule:  target_tokens=${BASE_TARGET_TOKENS}  iters=${BASE_N_ITERS}  (~5.02B tok, D/N≈9.5)"
echo "W&B:       ${WANDB_RUN_GROUP} / ${WANDB_NAME}  (mode=${WANDB_MODE})"
echo "=========================================="

$PY -m scripts.base_train \
  --preset 124m-modern \
  --run-dir "$BASE_RUN" \
  --input-bin "${DATA_DIR}/fineweb_train_*.bin" \
  --input-val-bin "${DATA_DIR}/fineweb_val_000000.bin" \
  --tokenizer-json "${DATA_DIR}/tokenizer.json" \
  --config-override depth="$BASE_DEPTH" \
  --config-override n_kv_heads=2 \
  --config-override skip_connections="((2, 5), (5, 7))" \
  --config-override target_train_tokens="$BASE_TARGET_TOKENS" \
  --config-override n_train_iters="$BASE_N_ITERS" \
  --config-override untie_at_step=-1 \
  --resume

echo "=========================================="
echo "Base eval (val_bpb + CORE-${CORE_N})  — compare vs miniseries-v2-d20 @1.31B (0.8684) and d16 winner (0.7626)"
echo "=========================================="
$PY -m scripts.base_eval \
  --run-dir "$BASE_RUN" \
  --core-n "$CORE_N" \
  --skip-generation \
  --tokenizer-json "${DATA_DIR}/tokenizer.json" \
  || echo "(base_eval skipped/partial)"

echo "=========================================="
echo "✅ 0.5B Chinchilla base complete — $(date)"
ls -lh "${BASE_RUN}/base/"state_step*.pkl 2>/dev/null | tail -1 || true
echo "Chain SFT+RL:  sbatch --export=ALL,SKIP_BASE=1,BASE_RUN=${BASE_RUN} ${PROJECT_DIR}/runs/rtx_8gpu_0p5b_e2e.sh"
echo "=========================================="
