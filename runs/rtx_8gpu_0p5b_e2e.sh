#!/usr/bin/env bash
#SBATCH --job-name=jaxchat-0p5b-e2e
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --output=/project/inniang/jaxchat/slurm-0p5b-e2e-%A.out
#SBATCH --error=/project/inniang/jaxchat/slurm-0p5b-e2e-%A.err
#SBATCH --export=ALL
#SBATCH --exclusive

# ============================================================================
# jaxchat 0.5B end-to-end pipeline — 8×RTX 6000
#
# Scaled from the completed Chinchilla v2 sweep:
#   - IsoFLOP fit: N* ∝ C^0.500, D* ∝ C^0.500
#   - Depth miniseries @ 1.31B tokens: best BPB at depth=16, depth=20 undertrained
#   - 0.5B target: depth=20 => 529,531,562 params
#   - Fit-implied D* for depth=20: 5,018,080,160 target tokens
#   - Rounded schedule: 19,143 steps × 262,144 tok/step = 5,018,222,592 tokens
#
# Full default run intentionally wraps the 2.94B-token local FineWeb pool ~1.7x.
# For a quick pipeline wiring test, override BASE_N_ITERS/SFT_N_ITERS/RL_N_ITERS
# at submit time, e.g.:
#   sbatch --export=ALL,BASE_N_ITERS=50,SFT_N_ITERS=10,RL_N_ITERS=2 \
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

E2E_ROOT="${E2E_ROOT:-${PROJECT_DIR}/data/0p5b_e2e}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${E2E_ROOT}/uv-cache}"
export UV_CACHE_DIR

BASE_RUN="${BASE_RUN:-${E2E_ROOT}/runs/base}"
SFT_RUN="${SFT_RUN:-${E2E_ROOT}/runs/sft}"
RL_RUN="${RL_RUN:-${E2E_ROOT}/runs/rl}"
SFT_DATA="${SFT_DATA:-${E2E_ROOT}/sft/smoltalk_mini.jsonl}"
RL_DATA_TRAIN="${RL_DATA_TRAIN:-${E2E_ROOT}/rl/gsm8k_train.jsonl}"
RL_DATA_VAL="${RL_DATA_VAL:-${E2E_ROOT}/rl/gsm8k_test.jsonl}"

BASE_DEPTH="${BASE_DEPTH:-20}"
BASE_PARAM_COUNT="${BASE_PARAM_COUNT:-529531562}"
BASE_TARGET_TOKENS="${BASE_TARGET_TOKENS:-5018080160}"
BASE_N_ITERS="${BASE_N_ITERS:-19143}"
BASE_ACTUAL_TOKENS="${BASE_ACTUAL_TOKENS:-5018222592}"
BASE_FLOPS="${BASE_FLOPS:-15943843485632692224}"

CORE_N="${CORE_N:-200}"
SFT_N="${SFT_N:-5000}"
SFT_N_ITERS="${SFT_N_ITERS:-400}"
RL_TRAIN_N="${RL_TRAIN_N:-2000}"
RL_VAL_N="${RL_VAL_N:-200}"
RL_N_ITERS="${RL_N_ITERS:-60}"
RL_M_PROMPTS="${RL_M_PROMPTS:-2}"
RL_G_ROLLOUTS="${RL_G_ROLLOUTS:-2}"
RL_MAX_NEW_TOKENS="${RL_MAX_NEW_TOKENS:-128}"

mkdir -p \
  "$WANDB_DIR" "$UV_CACHE_DIR" \
  "$BASE_RUN" "$SFT_RUN" "$RL_RUN" \
  "$(dirname "$SFT_DATA")" "$(dirname "$RL_DATA_TRAIN")" "$(dirname "$RL_DATA_VAL")"

DATA_DIR_29="${PROJECT_DIR}/data/fineweb32k_real_29"
DATA_DIR_9="${PROJECT_DIR}/data/fineweb32k_real"
if [ -f "${DATA_DIR_29}/fineweb_val_000000.bin" ] && compgen -G "${DATA_DIR_29}/fineweb_train_*.bin" >/dev/null 2>&1; then
  DATA_DIR="$DATA_DIR_29"
elif [ -f "${DATA_DIR_9}/fineweb_val_000000.bin" ] && compgen -G "${DATA_DIR_9}/fineweb_train_*.bin" >/dev/null 2>&1; then
  DATA_DIR="$DATA_DIR_9"
else
  echo "ERROR: no re-tokenized data found at ${DATA_DIR_29} or ${DATA_DIR_9}; run data/retokenize_bins.py first." >&2
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

echo "=========================================="
echo "jaxchat 0.5B e2e pipeline — 8×RTX 6000"
echo "Date: $(date)  Host: $(hostname)  GPUs: $(nvidia-smi -L | wc -l)"
echo "DATA_DIR:     $DATA_DIR"
echo "E2E_ROOT:     $E2E_ROOT"
echo "BASE_RUN:     $BASE_RUN"
echo "SFT_RUN:      $SFT_RUN"
echo "RL_RUN:       $RL_RUN"
echo "Base config:  depth=${BASE_DEPTH}, params=${BASE_PARAM_COUNT}, target_tokens=${BASE_TARGET_TOKENS}"
echo "Schedule:     iters=${BASE_N_ITERS}, actual_tokens=${BASE_ACTUAL_TOKENS}, flops=${BASE_FLOPS}"
echo "=========================================="

echo "[skip] Stage 1/9: tokenizer already at ${DATA_DIR}/tokenizer.json"
echo "[skip] Stage 2/9: data already packed in ${DATA_DIR}/"

echo "=========================================="
echo "Stage 3/9: base pretraining (124m-modern, depth=${BASE_DEPTH}, 0.5B)"
echo "=========================================="
if [ "${SKIP_BASE:-0}" = "1" ]; then
  echo "[skip] SKIP_BASE=1; using base checkpoint at $(cat "${BASE_RUN}/base/latest_checkpoint.txt")"
else
  WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-0p5b-e2e}" \
  WANDB_NAME="${WANDB_NAME:-0p5b-base-d${BASE_DEPTH}-${BASE_N_ITERS}steps}" \
  $PY -m scripts.base_train \
    --preset 124m-modern \
    --run-dir "$BASE_RUN" \
    --input-bin "${DATA_DIR}/fineweb_train_*.bin" \
    --input-val-bin "${DATA_DIR}/fineweb_val_000000.bin" \
    --tokenizer-json "${DATA_DIR}/tokenizer.json" \
    --config-override depth="$BASE_DEPTH" \
    --config-override target_train_tokens="$BASE_TARGET_TOKENS" \
    --config-override n_train_iters="$BASE_N_ITERS" \
    --config-override untie_at_step=-1 \
    --resume
fi

echo "=========================================="
echo "Stage 4/9: base eval (BPB + CORE-${CORE_N})"
echo "=========================================="
$PY -m scripts.base_eval \
  --run-dir "$BASE_RUN" \
  --core-n "$CORE_N" \
  --skip-generation \
  --tokenizer-json "${DATA_DIR}/tokenizer.json" \
  || echo "(base_eval skipped/partial)"

echo "=========================================="
echo "Stage 5/9: prepare SFT data"
echo "=========================================="
if [ -f "$SFT_DATA" ] && [ "$(wc -l < "$SFT_DATA")" -ge "$SFT_N" ]; then
  echo "[skip] SFT data already at $SFT_DATA ($(wc -l < "$SFT_DATA") rows)"
else
  $PY -m dev.synth_smoltalk --out "$SFT_DATA" --n "$SFT_N"
fi

echo "=========================================="
echo "Stage 6/9: supervised fine-tuning"
echo "=========================================="
$PY -m scripts.chat_sft \
  --base-run-dir "$BASE_RUN" \
  --sft-data "$SFT_DATA" \
  --run-dir "$SFT_RUN" \
  --n-iters "$SFT_N_ITERS" \
  --max-seq-len 1024 \
  --tokenizer-json "${DATA_DIR}/tokenizer.json"

echo "=========================================="
echo "Stage 7/9: chat eval after SFT"
echo "=========================================="
$PY -m scripts.chat_eval \
  --run-dir "$SFT_RUN" \
  --core-n "$CORE_N" \
  --gsm8k-n 50 \
  --tokenizer-json "${DATA_DIR}/tokenizer.json" \
  || echo "(chat_eval skipped/partial)"

echo "=========================================="
echo "Stage 8/9: prepare RL data + GRPO"
echo "=========================================="
$PY -m dev.synth_gsm8k --out "$RL_DATA_TRAIN" --split train --n "$RL_TRAIN_N"
$PY -m dev.synth_gsm8k --out "$RL_DATA_VAL"   --split test  --n "$RL_VAL_N"
$PY -m scripts.chat_rl \
  --sft-run-dir "$SFT_RUN" \
  --rl-data "$RL_DATA_TRAIN" \
  --run-dir "$RL_RUN" \
  --n-iters "$RL_N_ITERS" \
  --m-prompts "$RL_M_PROMPTS" \
  --g-rollouts "$RL_G_ROLLOUTS" \
  --max-new-tokens "$RL_MAX_NEW_TOKENS" \
  --tokenizer-json "${DATA_DIR}/tokenizer.json"

echo "=========================================="
echo "Stage 9/9: final chat eval after RL"
echo "=========================================="
$PY -m scripts.chat_eval \
  --run-dir "$RL_RUN" \
  --core-n "$CORE_N" \
  --gsm8k-n 100 \
  --tokenizer-json "${DATA_DIR}/tokenizer.json" \
  || echo "(chat_eval skipped/partial)"

echo
echo "=========================================="
echo "0.5B e2e pipeline complete — $(date)"
echo "Inference targets:"
echo "  CLI:    $PY -m scripts.chat_cli    --run-dir $RL_RUN --tokenizer-json ${DATA_DIR}/tokenizer.json"
echo "  Web UI: $PY -m scripts.chat_server --run-dir $RL_RUN --port 8000 --tokenizer-json ${DATA_DIR}/tokenizer.json"
echo "=========================================="
