#!/usr/bin/env bash
#SBATCH --job-name=jaxchat-depth12-long
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
# jaxchat depth-12 LONG run — 8×RTX 6000
#
# Hypothesis: the 124m preset is param-bottlenecked (~23M transformer matrices
# on ~75M of embedding tables; budget sized off the small number). The looped
# transformer ablation (#10 in ablation_notes.md) lost at 1.31B (0.9030 vs
# modern-long-notie's 0.8878), suggesting reusing the same 23M of params more
# times doesn't substitute for adding new ones. depth=12 keeps d_model=512
# (so attention shapes/heads are unchanged) and adds 4 more transformer blocks
# ⇒ ~35M of "real" transformer matrices (1.5× the depth-8 net) for ~12% more
# total params. Token budget held at 1.31B for direct A/B vs modern-long-notie.
#
# Reference numbers to beat (8×RTX 6000):
#   modern-long-notie (1.31B, depth-8) : val_bpb 0.8878 @ 5000 steps  (job 71121)
#   modern-xlong       (2.1B, depth-8) : val_bpb 0.8871 @ 8000 steps  (job 71135)
#   loop-long          (1.31B, n_rec=2): val_bpb 0.9030 @ 5000 steps  (job 71153)
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

DATA_DIR="/project/inniang/jaxchat/data/fineweb32k_real"
BASE_RUN="${D4_ROOT}/runs/depth12-long"
mkdir -p "$BASE_RUN"

if [ ! -f "${DATA_DIR}/fineweb_val_000000.bin" ] || ! compgen -G "${DATA_DIR}/fineweb_train_*.bin" >/dev/null 2>&1; then
  echo "ERROR: re-tokenized data missing at ${DATA_DIR}; run data/retokenize_bins.py first." >&2
  exit 1
fi

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
PY="uv run python -u"

echo "=========================================="
echo "jaxchat depth=12 LONG — 8×RTX 6000"
echo "Date: $(date)  Host: $(hostname)  GPUs: $(nvidia-smi -L | wc -l)"
echo "Run dir: $BASE_RUN"
echo "=========================================="

$PY -m scripts.base_train \
  --preset 124m-modern \
  --run-dir "$BASE_RUN" \
  --config-override depth=12 \
  --config-override n_train_iters=5000 \
  --config-override target_train_tokens=1310720000 \
  --config-override untie_at_step=-1 \
  --resume

echo ""
echo "=== eval (val_bpb + small CORE; generation skipped) ==="
$PY -m scripts.base_eval --run-dir "$BASE_RUN" --core-n 100 --skip-generation || echo "(base_eval skipped/partial)"
echo "=========================================="
echo "✅ depth-12 long run complete — $(date)"
ls -lh "${BASE_RUN}/base/"state_step*.pkl 2>/dev/null | tail -1 || true
