#!/usr/bin/env bash
#SBATCH --job-name=jaxchat-depth16-e2e
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
# jaxchat end-to-end SOTA pipeline — 8×RTX 6000
#
# Same 9 stages as runs/speedrun_d4.sh, but bound to the current SOTA recipe:
#   - 32K BPE tokenizer + fineweb32k_real_29/ shards already on disk → skip
#     tokenizer + data-pack stages.
#   - Base: 124m-modern preset, depth=16, weight_tying=none, 5000 steps /
#     1.31B tok (job 71194 produced val_bpb 0.7662 at data/124m_rtx_run/runs/
#     depth16-long/). Skipped on rerun if a final base checkpoint already
#     exists under $BASE_RUN.
#   - SFT, chat_eval, RL (GRPO), final chat_eval: same hyper-params as the d4
#     speedrun, just pointed at the SOTA base.
#
# Outputs land under data/124m_sota_e2e/ to keep the SOTA base run untouched.
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
mkdir -p "$WANDB_DIR"

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
E2E_ROOT="${E2E_ROOT:-${PROJECT_DIR}/data/124m_sota_e2e}"
SOTA_BASE_RUN="${SOTA_BASE_RUN:-${PROJECT_DIR}/data/124m_rtx_run/runs/depth16-long}"
BASE_RUN="${BASE_RUN:-${SOTA_BASE_RUN}}"
SFT_DATA="${E2E_ROOT}/sft/smoltalk_mini.jsonl"
RL_DATA_TRAIN="${E2E_ROOT}/rl/gsm8k_train.jsonl"
RL_DATA_VAL="${E2E_ROOT}/rl/gsm8k_test.jsonl"
SFT_RUN="${E2E_ROOT}/runs/sft"
RL_RUN="${E2E_ROOT}/runs/rl"

mkdir -p "${E2E_ROOT}/sft" "${E2E_ROOT}/rl" "$SFT_RUN" "$RL_RUN"

# --------------------------------------------------------------------------
# Data sanity: 32K re-tokenized FineWeb pool must exist (preset auto-picks).
# --------------------------------------------------------------------------
DATA_DIR_29="${PROJECT_DIR}/data/fineweb32k_real_29"
DATA_DIR_9="${PROJECT_DIR}/data/fineweb32k_real"
if [ -f "${DATA_DIR_29}/fineweb_val_000000.bin" ]; then
  DATA_DIR="$DATA_DIR_29"
elif [ -f "${DATA_DIR_9}/fineweb_val_000000.bin" ]; then
  DATA_DIR="$DATA_DIR_9"
else
  echo "ERROR: no re-tokenized data found at ${DATA_DIR_29} or ${DATA_DIR_9}; run data/retokenize_bins.py first." >&2
  exit 1
fi

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
PY="uv run python -u"

echo "=========================================="
echo "jaxchat e2e SOTA pipeline — 8×RTX 6000"
echo "Date: $(date)  Host: $(hostname)  GPUs: $(nvidia-smi -L | wc -l)"
echo "E2E_ROOT:   $E2E_ROOT"
echo "BASE_RUN:   $BASE_RUN"
echo "DATA_DIR:   $DATA_DIR"
echo "=========================================="

# --------------------------------------------------------------------------
# Stages 1+2 (tokenizer, data pack) are skipped: the 32K BPE tokenizer and
# fineweb32k_real_29 shards were prepared previously (data/retokenize_bins.py).
# --------------------------------------------------------------------------
echo "[skip] Stage 1/9: tokenizer already at ${DATA_DIR}/tokenizer.json"
echo "[skip] Stage 2/9: data already packed in ${DATA_DIR}/"

# --------------------------------------------------------------------------
# Stage 3: SOTA base pretraining (depth=16). Skipped if already done.
# --------------------------------------------------------------------------
echo "=========================================="
echo "Stage 3/9: base pretraining (124m-modern, depth=16, 1.31B tok)"
echo "=========================================="
mkdir -p "$BASE_RUN"
if [ -f "${BASE_RUN}/latest.txt" ] && compgen -G "${BASE_RUN}/$(cat "${BASE_RUN}/latest.txt")/state_step*.pkl" >/dev/null; then
  echo "[skip] base checkpoint already at ${BASE_RUN}/$(cat "${BASE_RUN}/latest.txt")"
else
  $PY -m scripts.base_train \
    --preset 124m-modern \
    --run-dir "$BASE_RUN" \
    --config-override depth=16 \
    --config-override n_train_iters=5000 \
    --config-override target_train_tokens=1310720000 \
    --config-override untie_at_step=-1 \
    --resume
fi

# --------------------------------------------------------------------------
# Stage 4: base eval (val_bpb + small CORE; skip slow generation eval).
# --------------------------------------------------------------------------
echo "=========================================="
echo "Stage 4/9: base eval (BPB + CORE-100)"
echo "=========================================="
$PY -m scripts.base_eval --run-dir "$BASE_RUN" --core-n 100 --skip-generation \
  || echo "(base_eval skipped/partial)"

# --------------------------------------------------------------------------
# Stage 5: SFT data (synthetic SmolTalk; falls back to embedded set offline).
# --------------------------------------------------------------------------
echo "=========================================="
echo "Stage 5/9: prepare SFT data"
echo "=========================================="
if [ -f "$SFT_DATA" ] && [ "$(wc -l < "$SFT_DATA")" -ge 1000 ]; then
  echo "[skip] SFT data already at $SFT_DATA ($(wc -l < "$SFT_DATA") rows)"
else
  $PY -m dev.synth_smoltalk --out "$SFT_DATA" --n 5000
fi

# --------------------------------------------------------------------------
# Stage 6: supervised fine-tuning on top of the SOTA base.
# --------------------------------------------------------------------------
echo "=========================================="
echo "Stage 6/9: supervised fine-tuning"
echo "=========================================="
$PY -m scripts.chat_sft \
  --base-run-dir "$BASE_RUN" \
  --sft-data "$SFT_DATA" \
  --run-dir "$SFT_RUN" \
  --n-iters 400 \
  --max-seq-len 1024

# --------------------------------------------------------------------------
# Stage 7: chat eval after SFT (CORE + small GSM8K).
# --------------------------------------------------------------------------
echo "=========================================="
echo "Stage 7/9: chat eval after SFT"
echo "=========================================="
$PY -m scripts.chat_eval --run-dir "$SFT_RUN" --core-n 200 --gsm8k-n 50 \
  || echo "(chat_eval skipped/partial)"

# --------------------------------------------------------------------------
# Stage 8: GRPO RL on GSM8K.
# --------------------------------------------------------------------------
echo "=========================================="
echo "Stage 8/9: prepare RL data + GRPO"
echo "=========================================="
$PY -m dev.synth_gsm8k --out "$RL_DATA_TRAIN" --split train --n 2000
$PY -m dev.synth_gsm8k --out "$RL_DATA_VAL"   --split test  --n 200
$PY -m scripts.chat_rl \
  --sft-run-dir "$SFT_RUN" \
  --rl-data "$RL_DATA_TRAIN" \
  --run-dir "$RL_RUN" \
  --n-iters 60 \
  --m-prompts 2 \
  --g-rollouts 2 \
  --max-new-tokens 128

# --------------------------------------------------------------------------
# Stage 9: final chat eval after RL.
# --------------------------------------------------------------------------
echo "=========================================="
echo "Stage 9/9: chat eval after RL"
echo "=========================================="
$PY -m scripts.chat_eval --run-dir "$RL_RUN" --core-n 200 --gsm8k-n 100 \
  || echo "(chat_eval skipped/partial)"

echo
echo "=========================================="
echo "✅ e2e SOTA pipeline complete — $(date)"
echo "Inference targets:"
echo "  CLI:    $PY -m scripts.chat_cli    --run-dir $RL_RUN"
echo "  Web UI: $PY -m scripts.chat_server --run-dir $RL_RUN --port 8000"
echo "=========================================="
