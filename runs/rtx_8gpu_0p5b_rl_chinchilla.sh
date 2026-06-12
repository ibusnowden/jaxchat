#!/usr/bin/env bash
#SBATCH --job-name=jaxchat-0p5b-rl
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=/project/inniang/jaxchat/slurm-0p5b-rl-%A.out
#SBATCH --error=/project/inniang/jaxchat/slurm-0p5b-rl-%A.err
#SBATCH --export=ALL
#SBATCH --exclusive

# ============================================================================
# jaxchat 0.5B — GRPO/RL stage only, relaunched against the existing SFT
# checkpoint to confirm the depth-20 RL OOM fix on hardware.
#
# Fix under test (scripts/chat_rl.py): the GRPO loss previously held several
# full (B,T,vocab) logits-sized tensors live at once (policy + reference logits
# + softmax/backward) and padded every batch to max_seq_len, which OOM'd the
# first step of job 80387 (34.22 GiB). Now (1) batches crop to their actual
# length (bucketed, mesh-divisible) and (2) the frozen reference log-probs are
# computed in a separate pass so only one logits tensor is live in the
# differentiated update.
#
# Default RL knobs match the run that OOM'd (M=4, G=8, 256 new tokens, 80 iters)
# so a clean finish here is a true confirmation. Override at submit time, e.g.:
#   sbatch --export=ALL,RL_N_ITERS=4,RL_M_PROMPTS=2 \
#     /project/inniang/jaxchat/runs/rtx_8gpu_0p5b_rl_chinchilla.sh
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

# Stage I/O (absolute, so they resolve the same inside the SLURM spool).
SFT_RUN="${SFT_RUN:-${E2E_ROOT}/runs/sft_chinchilla}"
RL_RUN="${RL_RUN:-${E2E_ROOT}/runs/rl_chinchilla}"
RL_DATA_TRAIN="${RL_DATA_TRAIN:-${E2E_ROOT}/rl/gsm8k_train_4k.jsonl}"
RL_DATA_VAL="${RL_DATA_VAL:-${E2E_ROOT}/rl/gsm8k_test_200.jsonl}"

# RL knobs — default to the job-80387 settings that OOM'd.
RL_N_ITERS="${RL_N_ITERS:-80}"
RL_M_PROMPTS="${RL_M_PROMPTS:-4}"
RL_G_ROLLOUTS="${RL_G_ROLLOUTS:-8}"
RL_MAX_NEW_TOKENS="${RL_MAX_NEW_TOKENS:-256}"
CORE_N="${CORE_N:-1000}"
FINAL_GSM8K_N="${FINAL_GSM8K_N:-100}"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-1}"

# Tokenizer / data dir (must match the 32k-vocab tokenizer the base+SFT used).
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

# Preflight: the SFT checkpoint and RL data must exist (RL is not resume-safe,
# but the SFT stage's checkpoints persist, so RL can be relaunched standalone).
SFT_LATEST="${SFT_RUN}/sft/latest_checkpoint.txt"
if [ ! -f "$SFT_LATEST" ]; then
  echo "ERROR: no SFT checkpoint marker at ${SFT_LATEST}; run the SFT stage first." >&2
  exit 1
fi
for f in "$RL_DATA_TRAIN" "$RL_DATA_VAL" "$TOKENIZER_JSON"; do
  if [ ! -f "$f" ]; then echo "ERROR: missing required input ${f}" >&2; exit 1; fi
done

mkdir -p "$WANDB_DIR" "$UV_CACHE_DIR" "$RL_RUN"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not on PATH; sync deps before submitting." >&2
  exit 1
fi
if [ ! -d ".venv" ] || [ "${SYNC_DEPS:-0}" = "1" ]; then
  uv sync --frozen
fi
PY="uv run --no-sync python -u"

echo "=========================================="
echo "jaxchat 0.5B GRPO/RL relaunch — 8xRTX 6000"
echo "Date: $(date)  Host: $(hostname)  GPUs: $(nvidia-smi -L | wc -l)"
echo "SFT_RUN:      $SFT_RUN  (ckpt: $(cat "$SFT_LATEST"))"
echo "RL_RUN:       $RL_RUN"
echo "RL data:      $RL_DATA_TRAIN"
echo "Tokenizer:    $TOKENIZER_JSON"
echo "RL knobs:     iters=${RL_N_ITERS} M=${RL_M_PROMPTS} G=${RL_G_ROLLOUTS} max_new=${RL_MAX_NEW_TOKENS}"
echo "=========================================="

echo "Stage 1/2: GRPO against the SFT policy"
$PY -m scripts.chat_rl \
  --sft-run-dir "$SFT_RUN" \
  --rl-data "$RL_DATA_TRAIN" \
  --run-dir "$RL_RUN" \
  --n-iters "$RL_N_ITERS" \
  --m-prompts "$RL_M_PROMPTS" \
  --g-rollouts "$RL_G_ROLLOUTS" \
  --max-new-tokens "$RL_MAX_NEW_TOKENS" \
  --tokenizer-json "$TOKENIZER_JSON"

if [ "$RUN_FINAL_EVAL" = "1" ]; then
  echo "=========================================="
  echo "Stage 2/2: final chat eval after RL"
  echo "=========================================="
  $PY -m scripts.chat_eval \
    --run-dir "$RL_RUN" \
    --core-n "$CORE_N" \
    --gsm8k-n "$FINAL_GSM8K_N" \
    --tokenizer-json "$TOKENIZER_JSON" \
    || echo "(chat_eval skipped/partial)"
fi

echo
echo "=========================================="
echo "0.5B RL relaunch complete — $(date)"
echo "  CLI:    $PY -m scripts.chat_cli    --run-dir $RL_RUN --tokenizer-json $TOKENIZER_JSON"
echo "  Web UI: $PY -m scripts.chat_server --run-dir $RL_RUN --port 8000 --tokenizer-json $TOKENIZER_JSON"
echo "=========================================="
