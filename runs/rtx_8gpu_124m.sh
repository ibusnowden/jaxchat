#!/usr/bin/env bash
#SBATCH --job-name=jaxchat-124m-rtx8
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

set -euo pipefail
cd /project/inniang/jaxchat

# Run artifacts go to a separate root so this run does not clobber the H100 run.
export D4_ROOT="${D4_ROOT:-/project/inniang/jaxchat/data/124m_rtx_run}"
export WANDB_PROJECT="${WANDB_PROJECT:-jaxchat}"
export WANDB_DIR="${WANDB_DIR:-/project/inniang/jaxchat/logs/wandb}"
# Compute nodes have no/flaky internet -> online wandb.init() hangs in a retry loop.
export WANDB_MODE="${WANDB_MODE:-offline}"

# GPU setup — JAX picks up all 8 devices automatically.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false"

mkdir -p "$WANDB_DIR" "$D4_ROOT"

# Exec the speedrun pipeline adapted for the 124M+32K-vocab preset.
# Instead of calling speedrun_d4.sh (which uses the 8K-vocab d4 preset),
# we inline the stages with the 124m preset here.
set -euo pipefail

unset VIRTUAL_ENV
export OMP_NUM_THREADS=1

# Re-tokenized 32k-BPE FineWeb shards (produced by data/retokenize_bins.py).  These are
# the *real* 32k data -- the old per-run fineweb32k dir held symlinks to GPT-2-tokenized
# shards (vocab 50257), which silently produced loss=nan on a 32k-vocab model.
DATA_DIR="/project/inniang/jaxchat/data/fineweb32k_real"
TOKENIZER_DIR="$DATA_DIR"
SFT_DATA="${D4_ROOT}/sft/smoltalk_mini.jsonl"
RL_DATA_TRAIN="${D4_ROOT}/rl/gsm8k_train.jsonl"
RL_DATA_VAL="${D4_ROOT}/rl/gsm8k_test.jsonl"
BASE_RUN="${D4_ROOT}/runs/base"
SFT_RUN="${D4_ROOT}/runs/sft"
RL_RUN="${D4_ROOT}/runs/rl"

mkdir -p "$BASE_RUN" "$SFT_RUN" "$RL_RUN" "${D4_ROOT}/sft" "${D4_ROOT}/rl"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync

PY="uv run python -u"

# ------------------------------------------------------------------
# Stage 1+2/7: 32K tokenizer + re-tokenized FineWeb shards
# ------------------------------------------------------------------
echo "=========================================="
echo "Stage 1+2/7: 32K tokenizer + re-tokenized FineWeb data"
echo "=========================================="
if [ ! -f "${DATA_DIR}/fineweb_val_000000.bin" ] || ! compgen -G "${DATA_DIR}/fineweb_train_*.bin" >/dev/null; then
  echo "[build] re-tokenizing GPT-2 shards -> 32K BPE into ${DATA_DIR} ..."
  $PY -m data.retokenize_bins \
    --source-dir /project/inniang/jaxchat/data/fineweb10B \
    --output-dir "${DATA_DIR}" \
    --tokenizer-json /project/inniang/jaxchat/data/fineweb32k/tokenizer.json \
    --seq-len 1024 --pack-mode concat --copy-tokenizer
fi
echo "[ok] data at ${DATA_DIR} ($(ls "${DATA_DIR}"/fineweb_train_*.bin | wc -l) train shards)"

# ------------------------------------------------------------------
# Stage 3/7: base pretraining (124M preset; data/tokenizer come from the preset)
# ------------------------------------------------------------------
echo "=========================================="
echo "Stage 3/7: base pretraining (124M)"
echo "=========================================="
if [ -f "${BASE_RUN}/latest.txt" ] && compgen -G "${BASE_RUN}/$(cat "${BASE_RUN}/latest.txt")/state_step*.pkl" >/dev/null; then
  echo "[skip] base checkpoint already at ${BASE_RUN}/$(cat "${BASE_RUN}/latest.txt")"
else
  $PY -m scripts.base_train \
    --preset 124m \
    --lr-schedule linear \
    --weight-tying none \
    --run-dir "$BASE_RUN"
fi

# ------------------------------------------------------------------
# Stage 4/7: base eval
# ------------------------------------------------------------------
echo "=========================================="
echo "Stage 4/7: base eval (BPB + CORE-subset)"
echo "=========================================="
$PY -m scripts.base_eval --run-dir "$BASE_RUN" --core-n 200 || echo "(base_eval skipped/partial)"

# ------------------------------------------------------------------
# Stage 5/7: supervised fine-tuning
# ------------------------------------------------------------------
echo "=========================================="
echo "Stage 5/7: supervised fine-tuning"
echo "=========================================="
if [ -f "$SFT_DATA" ] && [ "$(wc -l < "$SFT_DATA")" -ge 1000 ]; then
  echo "[skip] SFT data already at $SFT_DATA ($(wc -l < "$SFT_DATA") rows)"
else
  mkdir -p "$(dirname "$SFT_DATA")"
  # Copy existing SFT data if available, otherwise synthesize
  if [ -f /project/inniang/jaxchat/data/d4_speedrun/sft/smoltalk_mini.jsonl ]; then
    cp /project/inniang/jaxchat/data/d4_speedrun/sft/smoltalk_mini.jsonl "$SFT_DATA"
    echo "[copied] SFT data from d4_speedrun"
  else
    $PY -m dev.synth_smoltalk --out "$SFT_DATA" --n 5000
  fi
fi

$PY -m scripts.chat_sft \
  --base-run-dir "$BASE_RUN" \
  --sft-data "$SFT_DATA" \
  --run-dir "$SFT_RUN" \
  --n-iters 400 \
  --max-seq-len 1024

# ------------------------------------------------------------------
# Stage 6/7: chat eval after SFT
# ------------------------------------------------------------------
echo "=========================================="
echo "Stage 6/7: chat eval after SFT"
echo "=========================================="
$PY -m scripts.chat_eval --run-dir "$SFT_RUN" --core-n 200 --gsm8k-n 50 || echo "(chat_eval skipped/partial)"

# ------------------------------------------------------------------
# Stage 7/7: RL (GRPO)
# ------------------------------------------------------------------
echo "=========================================="
echo "Stage 7/7: GRPO reinforcement learning"
echo "=========================================="
if [ -f /project/inniang/jaxchat/data/d4_speedrun/rl/gsm8k_train.jsonl ]; then
  cp /project/inniang/jaxchat/data/d4_speedrun/rl/gsm8k_train.jsonl "$RL_DATA_TRAIN"
  cp /project/inniang/jaxchat/data/d4_speedrun/rl/gsm8k_test.jsonl "$RL_DATA_VAL"
  echo "[copied] RL data from d4_speedrun"
else
  $PY -m dev.synth_gsm8k --out "$RL_DATA_TRAIN" --split train --n 2000
  $PY -m dev.synth_gsm8k --out "$RL_DATA_VAL"   --split test  --n 200
fi

$PY -m scripts.chat_rl \
  --sft-run-dir "$SFT_RUN" \
  --rl-data "$RL_DATA_TRAIN" \
  --run-dir "$RL_RUN" \
  --n-iters 60 \
  --m-prompts 4 \
  --g-rollouts 4 \
  --max-new-tokens 128 \
  --kl-beta 0.01 \
  --lr-scale 0.02 \
  --clip-eps 0.2

# ------------------------------------------------------------------
# Stage 8/8 (bonus): chat eval after RL
# ------------------------------------------------------------------
echo "=========================================="
echo "Stage 8/8: chat eval after RL"
echo "=========================================="
$PY -m scripts.chat_eval --run-dir "$RL_RUN" --core-n 200 --gsm8k-n 100 || echo "(chat_eval skipped/partial)"

echo
echo "=========================================="
echo "All stages finished!"
echo "=========================================="
echo "Timing results written to slurm output."
echo "  CLI:    $PY -m scripts.chat_cli    --run-dir $RL_RUN"
echo "  Web UI: $PY -m scripts.chat_server --run-dir $RL_RUN --port 8000"
