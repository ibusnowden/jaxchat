#!/usr/bin/env bash
# Reference 124m base-pretraining run on 8x RTX 6000 (plain preset: linear LR, no
# modern features) on the re-tokenized 32k FineWeb data.  This is the baseline that
# 124m-modern and the ablations are measured against.
#SBATCH --job-name=jaxchat-124m-base
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --output=/project/inniang/jaxchat/slurm-base-%A.out
#SBATCH --error=/project/inniang/jaxchat/slurm-base-%A.err
#SBATCH --export=ALL
#SBATCH --exclusive

set -euo pipefail
cd /project/inniang/jaxchat

export D4_ROOT="${D4_ROOT:-/project/inniang/jaxchat/data/124m_rtx_run}"
export WANDB_PROJECT="${WANDB_PROJECT:-jaxchat}"
export WANDB_DIR="${WANDB_DIR:-/project/inniang/jaxchat/logs/wandb}"
# Compute nodes have no/flaky internet -> online wandb.init() hangs in a retry loop.
# Log offline; sync later from the login node with `wandb sync logs/wandb/...`.
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false"
mkdir -p "$WANDB_DIR" "$D4_ROOT"
unset VIRTUAL_ENV
export OMP_NUM_THREADS=1

DATA_DIR="/project/inniang/jaxchat/data/fineweb32k_real"
BASE_RUN="${D4_ROOT}/runs/base-linear"
mkdir -p "$BASE_RUN"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
PY="uv run python -u"

# Data: re-tokenized 32k shards (built once by data/retokenize_bins.py).
if [ ! -f "${DATA_DIR}/fineweb_val_000000.bin" ] || ! compgen -G "${DATA_DIR}/fineweb_train_*.bin" >/dev/null 2>&1; then
  echo "[build] re-tokenizing GPT-2 shards -> 32K BPE into ${DATA_DIR} ..."
  $PY -m data.retokenize_bins \
    --source-dir /project/inniang/jaxchat/data/fineweb10B \
    --output-dir "${DATA_DIR}" \
    --tokenizer-json /project/inniang/jaxchat/data/fineweb32k/tokenizer.json \
    --seq-len 1024 --pack-mode concat --copy-tokenizer
fi
echo "[ok] data: $(ls "${DATA_DIR}"/fineweb_train_*.bin | wc -l) train shards in ${DATA_DIR}"

echo "================ 124m base reference (linear, no modern features) ================"
$PY -m scripts.base_train \
  --preset 124m \
  --lr-schedule linear \
  --weight-tying none \
  --run-dir "$BASE_RUN"

echo "================ base eval (BPB + CORE-subset) ================"
$PY -m scripts.base_eval --run-dir "$BASE_RUN" --core-n 200 || echo "(base_eval skipped/partial)"

echo "================ done ================"
ls -lh "${BASE_RUN}/base/"state_step*.pkl 2>/dev/null | tail -1 || true
