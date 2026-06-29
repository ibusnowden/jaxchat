#!/usr/bin/env bash
#SBATCH --job-name=pbench
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100_80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=/project/inniang/jaxchat/slurm-pbench-%A_%a.out
#SBATCH --error=/project/inniang/jaxchat/slurm-pbench-%A_%a.err
#SBATCH --export=ALL
#SBATCH --array=0-19%1

# ============================================================================
# PretrainingBench — 1×H100 scaling-law sweep (Kaplan / Chinchilla / Muennighoff)
#
# 20 models, even depths 2..40, single clean corpus (FineWeb-Edu 32K-BPE).
# Each array task trains one depth to its Chinchilla-optimal token budget
# (capped by the 2.94B-token pool — the data-constrained regime for small N),
# then evals val_bpb + CORE.  After all tasks finish, run fit_scaling_law to
# fit the three scaling laws and emit plots + CSV + JSON.
#
# Inspect the grid before submitting:
#   uv run python -m scripts.pretrainingbench --print-grid
#   uv run python -m scripts.pretrainingbench --print-grid --vocab 4096 --max-depth 22
#
# Submit the full sweep (sequential on one H100):
#   sbatch /project/inniang/jaxchat/runs/h100_pretrainingbench.sh
# Submit only the 1×H100-feasible subset (depths 2..20, ~15M..~530M):
#   sbatch --array=0-9%1 /project/inniang/jaxchat/runs/h100_pretrainingbench.sh
# Resubmit one task after a crash:
#   sbatch --array=7 /project/inniang/jaxchat/runs/h100_pretrainingbench.sh
#
# Fit the scaling laws once runs land:
#   uv run python -m scripts.fit_scaling_law \
#     --runs-root "${PBENCH_ROOT}/runs" --out-dir "${PBENCH_ROOT}/_fit"
#
# Override the sweep (vocab, depth range, compute budget) with env vars:
#   PBENCH_VOCAB=4096 PBENCH_DEPTHS=2-22:2 sbatch ...
# ============================================================================

set -euo pipefail
cd /project/inniang/jaxchat

export WANDB_PROJECT="${WANDB_PROJECT:-jaxchat}"
export WANDB_DIR="${WANDB_DIR:-/project/inniang/jaxchat/logs/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/project/inniang/jaxchat/.uv-cache}"
export CORE_N="${CORE_N:-1000}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false \
  --xla_gpu_enable_cublaslt=True \
  --xla_gpu_autotune_level=4"

PBENCH_ROOT="${PBENCH_ROOT:-/project/inniang/jaxchat/data/pbench}"
PBENCH_VOCAB="${PBENCH_VOCAB:-32768}"
PBENCH_DEPTHS="${PBENCH_DEPTHS:-2-40:2}"
PBENCH_RATIO="${PBENCH_RATIO:-20}"
PBENCH_DATA_POOL="${PBENCH_DATA_POOL:-2940000000}"

mkdir -p "$WANDB_DIR" "$PBENCH_ROOT" "$PBENCH_ROOT/runs" "$UV_CACHE_DIR"
unset VIRTUAL_ENV
export OMP_NUM_THREADS=1

# --- Data: prefer the 29-shard re-tokenized 32K pool, then the 9-shard, then
# --- the legacy 32K dir.  For PBENCH_VOCAB != 32768 point DATA_DIR at a matching
# --- tokenizer + shards (e.g. fineweb8k for vocab=8192).
DATA_DIR_29="/project/inniang/jaxchat/data/fineweb32k_real_29"
DATA_DIR_9="/project/inniang/jaxchat/data/fineweb32k_real"
DATA_DIR_LEGACY="/project/inniang/jaxchat/data/fineweb32k"
if [ "$PBENCH_VOCAB" = "32768" ]; then
  if [ -f "${DATA_DIR_29}/fineweb_val_000000.bin" ] && compgen -G "${DATA_DIR_29}/fineweb_train_*.bin" >/dev/null 2>&1; then
    DATA_DIR="$DATA_DIR_29"
  elif [ -f "${DATA_DIR_9}/fineweb_val_000000.bin" ] && compgen -G "${DATA_DIR_9}/fineweb_train_*.bin" >/dev/null 2>&1; then
    DATA_DIR="$DATA_DIR_9"
  else
    DATA_DIR="$DATA_DIR_LEGACY"
  fi
  TOKENIZER_JSON="${DATA_DIR}/tokenizer.json"
else
  # Non-32K vocab: caller must point DATA_DIR + TOKENIZER_JSON at a matching shard set.
  DATA_DIR="${PBENCH_DATA_DIR:-/project/inniang/jaxchat/data/fineweb8k}"
  TOKENIZER_JSON="${PBENCH_TOKENIZER_JSON:-${DATA_DIR}/tokenizer.json}"
fi
echo "Using data dir: $DATA_DIR  (vocab=$PBENCH_VOCAB)"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not on PATH; install/sync dependencies before submitting." >&2
  exit 1
fi
if [ ! -d ".venv" ] || [ "${SYNC_DEPS:-0}" = "1" ]; then
  uv sync --frozen
fi
PY="uv run --no-sync python -u"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Pull per-task knobs from the grid (single source of truth).
GRID_ENV=$($PY -m scripts.pretrainingbench --task-id "$TASK_ID" --shell \
  --vocab "$PBENCH_VOCAB" --depths "$PBENCH_DEPTHS" --chinchilla-ratio "$PBENCH_RATIO" \
  --data-pool-tokens "$PBENCH_DATA_POOL")
eval "$GRID_ENV"

BASE_RUN="${PBENCH_ROOT}/runs/${RUN_NAME}"
mkdir -p "$BASE_RUN"
export WANDB_RUN_GROUP="${WANDB_GROUP_TASK}"
export WANDB_NAME="${RUN_NAME}"

echo "=========================================="
echo "PretrainingBench task ${TASK_ID}: ${RUN_NAME}"
echo "  depth=${DEPTH}  d_model=${D_MODEL}  vocab=${VOCAB}  params=${PARAMS}  non_emb=${NON_EMB_PARAMS}"
echo "  target_train_tokens=${TARGET_TRAIN_TOKENS}  iters=${N_TRAIN_ITERS}  tokens/step=${TOKENS_PER_STEP}"
echo "  actual_train_tokens=${ACTUAL_TRAIN_TOKENS}  FLOPs=${ACTUAL_FLOPS}  data_repeats=${DATA_REPEATS}x"
echo "  est_wall=${EST_WALL_HOURS}h  mem=${MEM_GB}G  feasible_1h100=${FEASIBLE_1H100}"
echo "  run_dir=${BASE_RUN}"
echo "  Date: $(date)  Host: $(hostname)  GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "=========================================="

if [ "${FEASIBLE_1H100}" = "0" ]; then
  echo "SKIP: task ${TASK_ID} (${RUN_NAME}) is not feasible on 1×H100 " \
       "(wall=${EST_WALL_HOURS}h or mem=${MEM_GB}G exceeds budget). Re-run with a larger per-model budget or on a multi-GPU node." >&2
  exit 0
fi

# --- Base pretraining (124m-modern feature set, depth/vocab overridden by the grid) ---
$PY -m scripts.base_train \
  --preset 124m-modern \
  --run-dir "$BASE_RUN" \
  --input-bin "${DATA_DIR}/fineweb_train_*.bin" \
  --input-val-bin "${DATA_DIR}/fineweb_val_000000.bin" \
  --tokenizer-json "$TOKENIZER_JSON" \
  --vocab-size "$VOCAB" \
  --config-override depth="$DEPTH" \
  --config-override n_kv_heads="$N_KV_HEADS" \
  --config-override skip_connections="$SKIP_CONNECTIONS" \
  --config-override target_train_tokens="$TARGET_TRAIN_TOKENS" \
  --config-override n_train_iters="$N_TRAIN_ITERS" \
  --config-override untie_at_step=-1 \
  --resume

echo ""
echo "=== eval (val_bpb + CORE subset, n=${CORE_N}/task) ==="
$PY -m scripts.base_eval --run-dir "$BASE_RUN" --core-n "$CORE_N" --skip-generation \
  --tokenizer-json "$TOKENIZER_JSON" || echo "(base_eval failed for ${RUN_NAME})"

echo "=== task ${TASK_ID} (${RUN_NAME}) complete ==="

# --- After the LAST array task, fit the scaling laws. ---
# SLURM doesn't easily give us "after the whole array", so run fit_scaling_law
# unconditionally; it's read-only and idempotent and skips missing runs.
echo "=== fitting scaling laws (Kaplan / Chinchilla / Muennighoff) ==="
$PY -m scripts.fit_scaling_law \
  --runs-root "${PBENCH_ROOT}/runs" \
  --out-dir "${PBENCH_ROOT}/_fit" \
  --vocab "$PBENCH_VOCAB" \
  --depths "$PBENCH_DEPTHS" \
  --chinchilla-ratio "$PBENCH_RATIO" \
  --data-pool-tokens "$PBENCH_DATA_POOL" \
  || echo "(fit_scaling_law skipped/partial — re-run once all tasks land)"
