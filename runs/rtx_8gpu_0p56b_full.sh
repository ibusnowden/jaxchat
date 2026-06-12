#!/usr/bin/env bash
#SBATCH --job-name=jaxchat-0p56b
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --output=/project/inniang/jaxchat/slurm-0p56b-%A.out
#SBATCH --error=/project/inniang/jaxchat/slurm-0p56b-%A.err
#SBATCH --export=ALL
#SBATCH --exclusive

set -euo pipefail

PROJECT_DIR="/project/inniang/jaxchat"
cd "$PROJECT_DIR"

unset VIRTUAL_ENV
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false --xla_gpu_enable_cublaslt=True --xla_gpu_autotune_level=4"
export WANDB_PROJECT="${WANDB_PROJECT:-jaxchat}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_DIR}/logs/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"

ROOT="${ROOT:-${PROJECT_DIR}/data/0p56b_rust65k}"
TOKENIZER_DIR="${ROOT}/tokenizer"
DATA_DIR="${ROOT}"
BASE_RUN="${ROOT}/runs/base"
MIDTRAIN_RUN="${ROOT}/runs/midtrain"
SFT_RUN="${ROOT}/runs/sft"
RL_RUN="${ROOT}/runs/rl"
SFT_DATA="${ROOT}/sft/smoltalk_mini.jsonl"
MIDTRAIN_DATA="${ROOT}/midtrain/mix.jsonl"
RL_TRAIN="${ROOT}/rl/gsm8k_train.jsonl"
RL_VAL="${ROOT}/rl/gsm8k_test.jsonl"
UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT}/uv-cache}"
export UV_CACHE_DIR

mkdir -p "$TOKENIZER_DIR" "$BASE_RUN" "$MIDTRAIN_RUN" "$SFT_RUN" "$RL_RUN" "${ROOT}/sft" "${ROOT}/midtrain" "${ROOT}/rl" "$WANDB_DIR" "$UV_CACHE_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not on PATH." >&2
  exit 1
fi
if [ ! -d ".venv" ] || [ "${SYNC_DEPS:-0}" = "1" ]; then
  uv sync --frozen
fi
PY="uv run --no-sync python -u"

echo "=========================================="
echo "JAXChat 0.56B Rust65K full pipeline"
echo "Root: $ROOT"
echo "Date: $(date) Host: $(hostname) GPUs: $(nvidia-smi -L | wc -l)"
echo "=========================================="

if [ -f "${TOKENIZER_DIR}/tokenizer.pkl" ]; then
  echo "[skip] Rust tokenizer exists at ${TOKENIZER_DIR}/tokenizer.pkl"
else
  $PY -m scripts.tok_train \
    --impl rust \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --vocab-size 65536 \
    --dataset-name "HuggingFaceFW/fineweb-edu,HuggingFaceFW/fineweb" \
    --max-documents "${TOKENIZER_MAX_DOCUMENTS:-1000000}"
fi

if compgen -G "${DATA_DIR}/fineweb_train_*.bin" >/dev/null && [ -f "${DATA_DIR}/fineweb_val_000000.bin" ]; then
  echo "[skip] FineWeb65K bins already exist in ${DATA_DIR}"
else
  $PY -m data.retokenize_bins \
    --source-dir "${SOURCE_DIR:-data/fineweb10B}" \
    --output-dir "$DATA_DIR" \
    --tokenizer-json "${TOKENIZER_DIR}/tokenizer.pkl" \
    --seq-len 2048 \
    --train-target-tokens "${PACK_TRAIN_TOKENS:-0}" \
    --val-target-tokens "${PACK_VAL_TOKENS:-100000000}" \
    --copy-tokenizer
fi

$PY -m scripts.base_train \
  --preset 0p56b-rust65k \
  --run-dir "$BASE_RUN" \
  --resume

$PY -m scripts.base_eval --run-dir "$BASE_RUN" --core-n "${CORE_N:-1000}" --skip-generation

if [ -f "$SFT_DATA" ] && [ "$(wc -l < "$SFT_DATA")" -ge "${SFT_N:-5000}" ]; then
  echo "[skip] SFT data exists at $SFT_DATA"
else
  $PY -m dev.synth_smoltalk --out "$SFT_DATA" --n "${SFT_N:-5000}"
fi

$PY -m dev.synth_midtrain --out "$MIDTRAIN_DATA" --smoltalk "$SFT_DATA" --n "${MIDTRAIN_N:-8000}"

$PY -m scripts.chat_midtrain \
  --base-run-dir "$BASE_RUN" \
  --midtrain-data "$MIDTRAIN_DATA" \
  --run-dir "$MIDTRAIN_RUN" \
  --n-iters "${MIDTRAIN_ITERS:-800}" \
  --max-seq-len 2048

$PY -m scripts.chat_sft \
  --base-run-dir "$MIDTRAIN_RUN" \
  --parent-stage sft \
  --sft-data "$SFT_DATA" \
  --run-dir "$SFT_RUN" \
  --n-iters "${SFT_ITERS:-400}" \
  --max-seq-len 2048

$PY -m scripts.chat_eval --run-dir "$SFT_RUN" --core-n "${CORE_N:-1000}" --gsm8k-n 100 --mmlu-n 200 --humaneval-n 20 || true

if [ "${SKIP_RL:-0}" != "1" ]; then
  $PY -m dev.synth_gsm8k --out "$RL_TRAIN" --split train --n "${RL_TRAIN_N:-2000}"
  $PY -m dev.synth_gsm8k --out "$RL_VAL" --split test --n "${RL_VAL_N:-200}"
  $PY -m scripts.chat_rl \
    --sft-run-dir "$SFT_RUN" \
    --rl-data "$RL_TRAIN" \
    --run-dir "$RL_RUN" \
    --n-iters "${RL_ITERS:-60}" \
    --m-prompts 2 \
    --g-rollouts 2 \
    --max-new-tokens 128
  $PY -m scripts.chat_eval --run-dir "$RL_RUN" --core-n "${CORE_N:-1000}" --gsm8k-n 100 --mmlu-n 200 --humaneval-n 20 || true
fi

$PY -m scripts.report_card --run-root "$ROOT" --out "${ROOT}/report.md"

echo "complete: ${ROOT}/report.md"
