#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="/project/inniang/jaxchat"
cd "$PROJECT_DIR"

export OMP_NUM_THREADS=1
SPEEDRUN_ROOT="$HOME/.cache/jaxchat/speedrun"
TOKENIZER_DIR="${SPEEDRUN_ROOT}/tokenizer"
DATA_DIR="${SPEEDRUN_ROOT}/data/fineweb32k"
RUN_DIR="${SPEEDRUN_ROOT}/base-depth24"
DATASET_NAME="HuggingFaceFW/fineweb-edu,HuggingFaceFW/fineweb"

mkdir -p "$TOKENIZER_DIR" "$DATA_DIR" "$RUN_DIR"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
source .venv/bin/activate

echo "Stage 1/5: tokenizer training"
python -m jaxchat.tokenizer \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --dataset-name "$DATASET_NAME"

echo "Stage 2/5: tokenizer evaluation"
python -m training.eval_tokenizer \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --dataset-name "$DATASET_NAME"

echo "Stage 3/5: dataset packing"
python -m data.cached_fineweb \
  --dataset-name "$DATASET_NAME" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --output-dir "$DATA_DIR" \
  --train-tokenizer \
  --train-target-tokens 14533312248 \
  --val-target-tokens 100000000

echo "Stage 4/5: base pretraining"
python -m training.train_base \
  --preset 1p384b-depth24 \
  --input-bin "${DATA_DIR}/fineweb_train_*.bin" \
  --input-val-bin "${DATA_DIR}/fineweb_val_000000.bin" \
  --tokenizer-json "${TOKENIZER_DIR}/tokenizer.json" \
  --run-dir "${RUN_DIR}"

echo "Stage 5/5: base evaluation"
python -m training.eval_base \
  --run-dir "${RUN_DIR}"
