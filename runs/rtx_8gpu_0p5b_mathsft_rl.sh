#!/usr/bin/env bash
#SBATCH --job-name=jaxchat-0p5b-mathrl
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=/project/inniang/jaxchat/slurm-0p5b-mathrl-%A.out
#SBATCH --error=/project/inniang/jaxchat/slurm-0p5b-mathrl-%A.err
#SBATCH --export=ALL
#SBATCH --exclusive

# ============================================================================
# jaxchat 0.5B — GSM8K math-SFT warmup -> shaped GRPO -> eval.
#
# WHY: the prior GRPO run (job 100123) completed but learned nothing —
# mean_reward stayed 0.0 for all 40 steps. The smoltalk-only SFT policy never
# emits \boxed{} and never solves a GSM8K problem, so every rollout in a group
# scores 0 under the strict 0/1 reward -> group-relative advantage 0 -> no
# gradient. This run breaks the deadlock with two complementary fixes:
#
#   1. math-SFT warmup: continue SFT from the existing sft_chinchilla ckpt on a
#      mix of smoltalk + GSM8K-with-\boxed{} CoT, so the policy learns the answer
#      format and lands a nonzero pass-rate on some prompts (pass@G > 0).
#   2. shaped GRPO reward: partial credit for emitting \boxed{} (and optionally
#      numeric proximity), so within-group reward variance — hence a gradient —
#      exists even before exact-match works. The *correctness* term is identical
#      to the eval metric, so we still optimize the real objective.
#
# All inputs are built offline from files already on disk (no compute-node
# network). Knobs are env-overridable; e.g. turn on dense proximity shaping:
#   sbatch --export=ALL,RL_PROXIMITY_COEF=0.3 \
#     /project/inniang/jaxchat/runs/rtx_8gpu_0p5b_mathsft_rl.sh
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

# --- Stage I/O (absolute paths; resolve identically inside the SLURM spool) ---
SFT_PARENT_RUN="${SFT_PARENT_RUN:-${E2E_ROOT}/runs/sft_chinchilla}"   # existing smoltalk SFT ckpt (parent)
MATHSFT_RUN="${MATHSFT_RUN:-${E2E_ROOT}/runs/sft_math}"                # continued math-SFT output
RL_RUN="${RL_RUN:-${E2E_ROOT}/runs/rl_math}"                          # GRPO output

SMOLTALK_DATA="${SMOLTALK_DATA:-${E2E_ROOT}/sft/smoltalk_50k.jsonl}"
RL_DATA_TRAIN="${RL_DATA_TRAIN:-${E2E_ROOT}/rl/gsm8k_train_4k.jsonl}"  # {question,answer}, also the math-SFT source
GSM8K_SFT_DATA="${GSM8K_SFT_DATA:-${E2E_ROOT}/sft/gsm8k_sft.jsonl}"    # built here: {messages} boxed CoT
SFT_MIX_DATA="${SFT_MIX_DATA:-${E2E_ROOT}/sft/sft_math_mix.jsonl}"     # built here: smoltalk + gsm8k mix
REBUILD_DATA="${REBUILD_DATA:-0}"                                     # 1 = rebuild even if present

# --- math-SFT knobs ---
MATHSFT_N_ITERS="${MATHSFT_N_ITERS:-2000}"
MATHSFT_LR_SCALE="${MATHSFT_LR_SCALE:-0.1}"   # gentle booster; lower than the 0.2 default to limit forgetting
MATHSFT_MICRO_BS="${MATHSFT_MICRO_BS:-4}"
MATHSFT_GRAD_ACCUM="${MATHSFT_GRAD_ACCUM:-2}"
MIX_N_SMOLTALK="${MIX_N_SMOLTALK:-16000}"
MIX_N_GSM8K="${MIX_N_GSM8K:-4000}"            # up-sampled; ~20% math fraction with the default smoltalk count

# --- GRPO knobs. B = M*G drives memory; B=16 (M=4,G=4) is the known-good
#     depth-20 fit (job 100123). G=8 (B=32) OOM'd even with the crop+ref-split
#     fix (job 99771), so keep B<=16 unless you've shrunk max_new_tokens. ---
RL_N_ITERS="${RL_N_ITERS:-80}"
RL_M_PROMPTS="${RL_M_PROMPTS:-4}"
RL_G_ROLLOUTS="${RL_G_ROLLOUTS:-4}"
RL_MAX_NEW_TOKENS="${RL_MAX_NEW_TOKENS:-256}"
RL_REWARD="${RL_REWARD:-shaped}"
RL_FORMAT_BONUS="${RL_FORMAT_BONUS:-0.1}"
RL_PROXIMITY_COEF="${RL_PROXIMITY_COEF:-0.0}"  # set >0 (e.g. 0.3) for a dense bootstrap signal

# --- eval knobs ---
CORE_N="${CORE_N:-1000}"
FINAL_GSM8K_N="${FINAL_GSM8K_N:-100}"
EVAL_SFT_GSM8K="${EVAL_SFT_GSM8K:-1}"   # gsm8k-only eval after math-SFT, before RL (is pass@1 > 0 yet?)
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-1}"

# --- Tokenizer / data dir (must match the 32k-vocab tokenizer base+SFT used) ---
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

# --- Preflight: parent SFT ckpt + source data must exist ---
SFT_PARENT_LATEST="${SFT_PARENT_RUN}/sft/latest_checkpoint.txt"
if [ ! -f "$SFT_PARENT_LATEST" ]; then
  echo "ERROR: no parent SFT checkpoint at ${SFT_PARENT_LATEST}; run the smoltalk SFT stage first." >&2
  exit 1
fi
for f in "$SMOLTALK_DATA" "$RL_DATA_TRAIN" "$TOKENIZER_JSON"; do
  if [ ! -f "$f" ]; then echo "ERROR: missing required input ${f}" >&2; exit 1; fi
done

mkdir -p "$WANDB_DIR" "$UV_CACHE_DIR" "$MATHSFT_RUN" "$RL_RUN" "$(dirname "$GSM8K_SFT_DATA")"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not on PATH; sync deps before submitting." >&2
  exit 1
fi
if [ ! -d ".venv" ] || [ "${SYNC_DEPS:-0}" = "1" ]; then
  uv sync --frozen
fi
PY="uv run --no-sync python -u"

echo "=========================================="
echo "jaxchat 0.5B math-SFT + shaped GRPO — 8xRTX 6000"
echo "Date: $(date)  Host: $(hostname)  GPUs: $(nvidia-smi -L | wc -l)"
echo "Parent SFT:   $SFT_PARENT_RUN  (ckpt: $(cat "$SFT_PARENT_LATEST"))"
echo "math-SFT out: $MATHSFT_RUN  (iters=${MATHSFT_N_ITERS} lr_scale=${MATHSFT_LR_SCALE})"
echo "RL out:       $RL_RUN"
echo "RL knobs:     iters=${RL_N_ITERS} M=${RL_M_PROMPTS} G=${RL_G_ROLLOUTS} max_new=${RL_MAX_NEW_TOKENS}"
echo "RL reward:    ${RL_REWARD} (format_bonus=${RL_FORMAT_BONUS} proximity_coef=${RL_PROXIMITY_COEF})"
echo "Tokenizer:    $TOKENIZER_JSON"
echo "=========================================="

# --- Stage 0: build GSM8K-SFT + mixed pool (offline, from local files) ---
echo "Stage 0/3: build math-SFT data (offline)"
if [ "$REBUILD_DATA" = "1" ] || [ ! -f "$GSM8K_SFT_DATA" ]; then
  $PY -m dev.synth_gsm8k --mode sft --from-jsonl "$RL_DATA_TRAIN" --out "$GSM8K_SFT_DATA" --n 100000
else
  echo "  reuse $GSM8K_SFT_DATA ($(wc -l < "$GSM8K_SFT_DATA") rows)"
fi
if [ "$REBUILD_DATA" = "1" ] || [ ! -f "$SFT_MIX_DATA" ]; then
  $PY -m dev.build_sft_mix \
    --smoltalk "$SMOLTALK_DATA" \
    --gsm8k-sft "$GSM8K_SFT_DATA" \
    --out "$SFT_MIX_DATA" \
    --n-smoltalk "$MIX_N_SMOLTALK" \
    --n-gsm8k "$MIX_N_GSM8K"
else
  echo "  reuse $SFT_MIX_DATA ($(wc -l < "$SFT_MIX_DATA") rows)"
fi

# --- Stage 1: continued math-SFT from the existing SFT checkpoint ---
echo "=========================================="
echo "Stage 1/3: continued math-SFT (parent-stage sft)"
echo "=========================================="
$PY -m scripts.chat_sft \
  --base-run-dir "$SFT_PARENT_RUN" \
  --parent-stage sft \
  --sft-data "$SFT_MIX_DATA" \
  --run-dir "$MATHSFT_RUN" \
  --n-iters "$MATHSFT_N_ITERS" \
  --micro-batch-size "$MATHSFT_MICRO_BS" \
  --n-grad-accum "$MATHSFT_GRAD_ACCUM" \
  --lr-scale "$MATHSFT_LR_SCALE" \
  --tokenizer-json "$TOKENIZER_JSON"

if [ "$EVAL_SFT_GSM8K" = "1" ]; then
  echo "--- gsm8k-only eval after math-SFT (sanity: is pass@1 > 0 before RL?) ---"
  $PY -m scripts.chat_eval \
    --run-dir "$MATHSFT_RUN" \
    --skip-core --skip-mmlu --skip-humaneval \
    --gsm8k-n "$FINAL_GSM8K_N" \
    --tokenizer-json "$TOKENIZER_JSON" \
    || echo "(sft gsm8k eval skipped/partial)"
fi

# --- Stage 2: GRPO with the shaped reward, against the math-SFT policy ---
echo "=========================================="
echo "Stage 2/3: shaped GRPO against the math-SFT policy"
echo "=========================================="
$PY -m scripts.chat_rl \
  --sft-run-dir "$MATHSFT_RUN" \
  --rl-data "$RL_DATA_TRAIN" \
  --run-dir "$RL_RUN" \
  --n-iters "$RL_N_ITERS" \
  --m-prompts "$RL_M_PROMPTS" \
  --g-rollouts "$RL_G_ROLLOUTS" \
  --max-new-tokens "$RL_MAX_NEW_TOKENS" \
  --reward "$RL_REWARD" \
  --format-bonus "$RL_FORMAT_BONUS" \
  --proximity-coef "$RL_PROXIMITY_COEF" \
  --tokenizer-json "$TOKENIZER_JSON"

# --- Stage 3: final chat eval after RL ---
if [ "$RUN_FINAL_EVAL" = "1" ]; then
  echo "=========================================="
  echo "Stage 3/3: final chat eval after RL"
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
echo "0.5B math-SFT + shaped GRPO complete — $(date)"
echo "  SFT gsm8k eval: ${MATHSFT_RUN}/chat_eval.json"
echo "  RL  full eval:  ${RL_RUN}/chat_eval.json"
echo "  CLI:    $PY -m scripts.chat_cli    --run-dir $RL_RUN --tokenizer-json $TOKENIZER_JSON"
echo "  Web UI: $PY -m scripts.chat_server --run-dir $RL_RUN --port 8000 --tokenizer-json $TOKENIZER_JSON"
echo "=========================================="
