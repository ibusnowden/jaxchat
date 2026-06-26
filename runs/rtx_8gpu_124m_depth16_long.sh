#!/usr/bin/env bash
#SBATCH --job-name=jaxchat-depth16-long
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
# jaxchat depth-16 LONG run — 8×RTX 6000
#
# Follow-up to depth-12 SOTA (val_bpb 0.8275 @ 1.31B, job 71178). depth=12
# auto-scales d_model=768 ⇒ 188M params (75M transformer); curve was still
# descending at cutoff, so the lever is "more transformer params on same data".
# depth=16 auto-scales d_model=1024, n_heads=8, head_dim=128 ⇒ ~330M params
# (~165M transformer matrices, 2.2× depth-12). Tokens/transformer-matrix drops
# to ~8 at 1.31B (borderline under-trained), but the descending depth-12 curve
# suggests there's room.
#
# Wall-clock: depth-12 was 2.74 s/step over 5000 steps = 3.96 h. depth-16 is
# ~1.78× cost per layer × 1.33× more layers ≈ 2.4×, so ~9-10 h expected.
#
# Reference numbers to beat (8×RTX 6000, 1.31B tok):
#   modern-long-notie (depth-8,  d=512)  : val_bpb 0.8878 @ 5000 steps  (job 71121)
#   depth12-long      (depth-12, d=768)  : val_bpb 0.8275 @ 5000 steps  (job 71178)
# ============================================================================

set -euo pipefail
cd /project/inniang/jaxchat

export D4_ROOT="${D4_ROOT:-/project/inniang/jaxchat/data/124m_rtx_run}"
export WANDB_PROJECT="${WANDB_PROJECT:-jaxchat}"
export WANDB_DIR="${WANDB_DIR:-/project/inniang/jaxchat/logs/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false \
  --xla_gpu_enable_cublaslt=True \
  --xla_gpu_autotune_level=4"

mkdir -p "$WANDB_DIR" "$D4_ROOT"
unset VIRTUAL_ENV
export OMP_NUM_THREADS=1

# Data dir is resolved by presets.py (prefers fineweb32k_real_29/ → fineweb32k_real/ → legacy);
# this guard just confirms one of the real (re-tokenized) pools exists before launching.
DATA_DIR_29="/project/inniang/jaxchat/data/fineweb32k_real_29"
DATA_DIR_9="/project/inniang/jaxchat/data/fineweb32k_real"
if [ -f "${DATA_DIR_29}/fineweb_val_000000.bin" ] && compgen -G "${DATA_DIR_29}/fineweb_train_*.bin" >/dev/null 2>&1; then
  DATA_DIR="$DATA_DIR_29"
elif [ -f "${DATA_DIR_9}/fineweb_val_000000.bin" ] && compgen -G "${DATA_DIR_9}/fineweb_train_*.bin" >/dev/null 2>&1; then
  DATA_DIR="$DATA_DIR_9"
else
  echo "ERROR: no re-tokenized data found at ${DATA_DIR_29} or ${DATA_DIR_9}; run data/retokenize_bins.py first." >&2
  exit 1
fi
echo "Using data dir: $DATA_DIR"

BASE_RUN="${D4_ROOT}/runs/depth16-long"
mkdir -p "$BASE_RUN"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
PY="uv run python -u"

echo "=========================================="
echo "jaxchat depth=16 LONG — 8×RTX 6000"
echo "Date: $(date)  Host: $(hostname)  GPUs: $(nvidia-smi -L | wc -l)"
echo "Run dir: $BASE_RUN"
echo "=========================================="

$PY -m scripts.base_train \
  --preset 124m-modern \
  --run-dir "$BASE_RUN" \
  --config-override depth=16 \
  --config-override n_train_iters=5000 \
  --config-override target_train_tokens=1310720000 \
  --config-override untie_at_step=-1 \
  --resume

echo ""
echo "=== eval (val_bpb + small CORE; generation skipped) ==="
$PY -m scripts.base_eval --run-dir "$BASE_RUN" --core-n 100 --skip-generation || echo "(base_eval skipped/partial)"
echo "=========================================="
echo "✅ depth-16 long run complete — $(date)"
ls -lh "${BASE_RUN}/base/"state_step*.pkl 2>/dev/null | tail -1 || true
