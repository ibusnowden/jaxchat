#!/usr/bin/env bash
#SBATCH --job-name=jaxchat-124m-abl
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --output=/project/inniang/jaxchat/slurm-abl-%A.out
#SBATCH --error=/project/inniang/jaxchat/slurm-abl-%A.err
#SBATCH --export=ALL
#SBATCH --exclusive

# ============================================================================
# jaxchat 124M single-ablation driver — 8×RTX 6000
#
# Runs ONE config (default: the full 124m-modern preset) so the P0-P3 ablations
# can be submitted independently and compared against:
#   124m-modern (683M tokens) : val_bpb 1.2249 @ 2604 steps   (job 71056)
#   124m  (linear, no modern) : val_bpb 1.3152 @ 2352 steps   (job 71053)
#
# Usage:
#   sbatch runs/rtx_124m_ablation.sh <ablation>
# where <ablation> is one of:
#   modern         all features on (control = reproduce 71056)
#   no-wsd         lr_schedule=linear
#   no-clip        max_grad_norm=0.0
#   default-init   init_style=default
#   no-zloss       z_loss_coeff=0.0
#   no-tying       weight_tying=none
#   no-seqwarmup   sequence_warmup_intervals=0  (+ min_seq_len=1024)
#   no-bigram      bigram_hash_embed=False
#   no-longshort   use_long_short_attention=False
#   no-crossdoc    cross_document_mask=False
#   sigmoid-cap    logit_cap_style=sigmoid   (vs the default; modern doesn't set tanh -- kept for completeness)
#   no-skip        skip_connections=()
#   full-mha       n_kv_heads=4              (speed-only control, no BPB change expected)
#
# Plus you can pass extra raw overrides after the name, e.g.:
#   sbatch runs/rtx_124m_ablation.sh custom --config-override muon_base_lr=0.05
# ============================================================================

set -euo pipefail
cd /project/inniang/jaxchat

ABL="${1:-modern}"; shift || true
EXTRA_ARGS=("$@")

export D4_ROOT="${D4_ROOT:-/project/inniang/jaxchat/data/124m_rtx_run}"
export WANDB_PROJECT="${WANDB_PROJECT:-jaxchat-ablation}"
export WANDB_DIR="${WANDB_DIR:-/project/inniang/jaxchat/logs/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false \
  --xla_gpu_enable_cublaslt=True \
  --xla_gpu_autotune_level=4"

mkdir -p "$WANDB_DIR"
unset VIRTUAL_ENV
export OMP_NUM_THREADS=1

DATA_DIR="/project/inniang/jaxchat/data/fineweb32k_real"
if [ ! -f "${DATA_DIR}/fineweb_val_000000.bin" ] || ! compgen -G "${DATA_DIR}/fineweb_train_*.bin" >/dev/null 2>&1; then
  echo "ERROR: re-tokenized data missing at ${DATA_DIR}; run data/retokenize_bins.py first." >&2
  exit 1
fi

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
PY="uv run python -u"

# --- Map the ablation name -> config override(s) ---
OVERRIDE=()
case "$ABL" in
  modern)        ;;  # control: no override
  no-wsd)        OVERRIDE=(--lr-schedule linear) ;;
  no-clip)       OVERRIDE=(--config-override max_grad_norm=0.0) ;;
  default-init)  OVERRIDE=(--config-override init_style=default) ;;
  no-zloss)      OVERRIDE=(--config-override z_loss_coeff=0.0) ;;
  no-tying)      OVERRIDE=(--weight-tying none) ;;
  no-seqwarmup)  OVERRIDE=(--config-override sequence_warmup_intervals=0 --config-override min_seq_len=1024) ;;
  no-bigram)     OVERRIDE=(--config-override bigram_hash_embed=False) ;;
  no-longshort)  OVERRIDE=(--config-override use_long_short_attention=False) ;;
  no-crossdoc)   OVERRIDE=(--config-override cross_document_mask=False) ;;
  sigmoid-cap)   OVERRIDE=(--config-override logit_cap_style=sigmoid) ;;
  no-skip)       OVERRIDE=(--config-override "skip_connections=()") ;;
  full-mha)      OVERRIDE=(--config-override n_kv_heads=4) ;;
  custom)        ;;  # only EXTRA_ARGS
  *) echo "Unknown ablation '$ABL'. See header for the list." >&2; exit 2 ;;
esac

RUN_DIR="${D4_ROOT}/ablation/${ABL}"
mkdir -p "$RUN_DIR"

echo "=========================================="
echo "jaxchat 124M ablation: ${ABL}"
echo "Date: $(date)  Host: $(hostname)  GPUs: $(nvidia-smi -L | wc -l)"
echo "Overrides: ${OVERRIDE[*]:-<none>}  ${EXTRA_ARGS[*]:-}"
echo "Run dir:   ${RUN_DIR}"
echo "=========================================="

$PY -m scripts.base_train \
  --preset 124m-modern \
  --run-dir "$RUN_DIR" \
  --resume \
  "${OVERRIDE[@]}" "${EXTRA_ARGS[@]}"

echo ""
echo "=== ablation ${ABL} done; eval (val_bpb + small CORE; generation skipped for speed) ==="
$PY -m scripts.base_eval --run-dir "$RUN_DIR" --core-n 100 --skip-generation || echo "(base_eval skipped/partial)"
echo "=========================================="
echo "✅ ablation ${ABL} complete — $(date)"
ls -lh "${RUN_DIR}/base/"state_step*.pkl 2>/dev/null | tail -1 || true
