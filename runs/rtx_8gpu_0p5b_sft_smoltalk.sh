#!/usr/bin/env bash
#SBATCH --job-name=0p5b-sft
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --output=/project/inniang/jaxchat/slurm-0p5b-sft-%A.out
#SBATCH --error=/project/inniang/jaxchat/slurm-0p5b-sft-%A.err
#SBATCH --export=ALL
#SBATCH --exclusive

# ============================================================================
# jaxchat 0.5B — smoltalk supervised fine-tuning (base_chinchilla -> sft_chinchilla).
#
# Stage 2 of the post-bigram-fix rebuild (Stage 1 = rtx_8gpu_0p5b_base_chinchilla.sh).
# Produces the smoltalk chat model that (a) is the parent for the math-SFT +
# shaped-GRPO stage (rtx_8gpu_0p5b_mathsft_rl.sh, which reads sft_chinchilla),
# and (b) is itself "the working chatbot" — so its generation sanity GATES the
# chain: a hard regression here exits nonzero and an afterok dependency stops
# before GRPO burns more GPU.
#
# Inputs already on disk (no compute-node network): smoltalk_50k.jsonl.
# Submit (chained after the base job):
#   sbatch --dependency=afterok:<BASEJOBID> \
#     /project/inniang/jaxchat/runs/rtx_8gpu_0p5b_sft_smoltalk.sh
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

BASE_RUN="${BASE_RUN:-${E2E_ROOT}/runs/base_chinchilla}"
SFT_RUN="${SFT_RUN:-${E2E_ROOT}/runs/sft_chinchilla}"
SFT_DATA="${SFT_DATA:-${E2E_ROOT}/sft/smoltalk_50k.jsonl}"

SFT_N_ITERS="${SFT_N_ITERS:-4000}"
SFT_MAX_SEQ_LEN="${SFT_MAX_SEQ_LEN:-1024}"
CORE_N="${CORE_N:-1000}"
RUN_SANITY="${RUN_SANITY:-1}"

DATA_DIR_29="${PROJECT_DIR}/data/fineweb32k_real_29"
DATA_DIR_9="${PROJECT_DIR}/data/fineweb32k_real"
if [ -f "${DATA_DIR_29}/tokenizer.json" ]; then
  DATA_DIR="$DATA_DIR_29"
elif [ -f "${DATA_DIR_9}/tokenizer.json" ]; then
  DATA_DIR="$DATA_DIR_9"
else
  echo "ERROR: no tokenizer.json under ${DATA_DIR_29} or ${DATA_DIR_9}" >&2
  exit 1
fi
TOKENIZER_JSON="${DATA_DIR}/tokenizer.json"

BASE_LATEST="${BASE_RUN}/base/latest_checkpoint.txt"
if [ ! -f "$BASE_LATEST" ]; then
  echo "ERROR: no base checkpoint at ${BASE_LATEST}; run the base job first." >&2
  exit 1
fi
if [ ! -f "$SFT_DATA" ]; then
  echo "ERROR: missing SFT data ${SFT_DATA}" >&2
  exit 1
fi

mkdir -p "$WANDB_DIR" "$UV_CACHE_DIR" "$SFT_RUN"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not on PATH; sync deps before submitting." >&2
  exit 1
fi
if [ ! -d ".venv" ] || [ "${SYNC_DEPS:-0}" = "1" ]; then
  uv sync --frozen
fi
PY="uv run --no-sync python -u"

export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-0p5b-chinchilla}"
export WANDB_NAME="${WANDB_NAME:-0p5b-sft-smoltalk-${SFT_N_ITERS}}"

echo "=========================================="
echo "jaxchat 0.5B smoltalk SFT — 8xRTX 6000"
echo "Date: $(date)  Host: $(hostname)  GPUs: $(nvidia-smi -L | wc -l)"
echo "BASE_RUN:   $BASE_RUN  (ckpt: $(cat "$BASE_LATEST"))"
echo "SFT_RUN:    $SFT_RUN"
echo "SFT_DATA:   $SFT_DATA ($(wc -l < "$SFT_DATA") rows)"
echo "Knobs:      n_iters=${SFT_N_ITERS} max_seq_len=${SFT_MAX_SEQ_LEN} core_n=${CORE_N}"
echo "Tokenizer:  $TOKENIZER_JSON"
echo "=========================================="

echo "Stage 1/3: supervised fine-tuning on smoltalk"
$PY -m scripts.chat_sft \
  --base-run-dir "$BASE_RUN" \
  --sft-data "$SFT_DATA" \
  --run-dir "$SFT_RUN" \
  --n-iters "$SFT_N_ITERS" \
  --max-seq-len "$SFT_MAX_SEQ_LEN" \
  --tokenizer-json "$TOKENIZER_JSON"

echo "=========================================="
echo "Stage 2/3: chat eval after SFT (CORE-${CORE_N} + gsm8k)"
echo "=========================================="
$PY -m scripts.chat_eval \
  --run-dir "$SFT_RUN" \
  --core-n "$CORE_N" \
  --gsm8k-n 50 \
  --tokenizer-json "$TOKENIZER_JSON" \
  || echo "(chat_eval skipped/partial)"

if [ "$RUN_SANITY" = "1" ]; then
  echo "=========================================="
  echo "Stage 3/3: GENERATION SANITY (Engine chat path — gates the chain)"
  echo "=========================================="
  # No '|| true': a hard regression (bigram-leak signature) exits nonzero so an
  # afterok dependency halts before the GRPO stage.
  $PY -m dev.gen_sanity_0p5b --run-dir "$SFT_RUN" --stage sft --tokenizer-json "$TOKENIZER_JSON"
fi

echo
echo "=========================================="
echo "✅ 0.5B smoltalk SFT complete — $(date)"
echo "  eval:   ${SFT_RUN}/chat_eval.json"
echo "  next:   sbatch --export=ALL,RL_PROXIMITY_COEF=0.3 ${PROJECT_DIR}/runs/rtx_8gpu_0p5b_mathsft_rl.sh"
echo "=========================================="
