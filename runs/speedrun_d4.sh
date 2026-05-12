#!/usr/bin/env bash
# End-to-end d4 (~10M-param) pipeline: tokenizer -> pretrain -> SFT -> RL -> inference.
# Designed to run on a single H100 in ~2-4 hours.

set -euo pipefail

PROJECT_DIR="/project/inniang/jaxchat"
cd "$PROJECT_DIR"

unset VIRTUAL_ENV
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false"

D4_ROOT="${D4_ROOT:-$PROJECT_DIR/data/d4_speedrun}"
TOKENIZER_DIR="${D4_ROOT}/tokenizer"
DATA_DIR="${D4_ROOT}/fineweb8k"
SFT_DATA="${D4_ROOT}/sft/smoltalk_mini.jsonl"
RL_DATA_TRAIN="${D4_ROOT}/rl/gsm8k_train.jsonl"
RL_DATA_VAL="${D4_ROOT}/rl/gsm8k_test.jsonl"
BASE_RUN="${D4_ROOT}/runs/base"
SFT_RUN="${D4_ROOT}/runs/sft"
RL_RUN="${D4_ROOT}/runs/rl"

mkdir -p "$TOKENIZER_DIR" "$DATA_DIR" "$BASE_RUN" "$SFT_RUN" "$RL_RUN" "${D4_ROOT}/sft" "${D4_ROOT}/rl"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync

PY="uv run python -u"

echo "=========================================="
echo "Stage 1/9: tokenizer (vocab=8192)"
echo "=========================================="
if [ -f "${TOKENIZER_DIR}/tokenizer.json" ]; then
  echo "[skip] tokenizer already at ${TOKENIZER_DIR}/tokenizer.json"
else
  $PY -m scripts.tok_train \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --dataset-name "HuggingFaceFW/fineweb-edu" \
    --vocab-size 8192 \
    --max-documents 200000
fi

echo "=========================================="
echo "Stage 2/9: pack fineweb at 8k vocab"
echo "=========================================="
if compgen -G "${DATA_DIR}/fineweb_train_*.bin" >/dev/null && [ -f "${DATA_DIR}/fineweb_val_000000.bin" ]; then
  echo "[skip] fineweb shards already in ${DATA_DIR}"
else
  $PY -m data.cached_fineweb \
    --dataset-name "HuggingFaceFW/fineweb-edu,HuggingFaceFW/fineweb" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --output-dir "$DATA_DIR" \
    --train-target-tokens 150000000 \
    --val-target-tokens 1000000
fi

echo "=========================================="
echo "Stage 3/9: base pretraining (d4)"
echo "=========================================="
if [ -f "${BASE_RUN}/latest.txt" ] && compgen -G "${BASE_RUN}/$(cat "${BASE_RUN}/latest.txt")/state_step*.pkl" >/dev/null; then
  echo "[skip] base checkpoint already at ${BASE_RUN}/$(cat "${BASE_RUN}/latest.txt")"
else
  $PY -m scripts.base_train \
    --preset d4 \
    --input-bin "${DATA_DIR}/fineweb_train_*.bin" \
    --input-val-bin "${DATA_DIR}/fineweb_val_000000.bin" \
    --tokenizer-json "${TOKENIZER_DIR}/tokenizer.json" \
    --run-dir "$BASE_RUN"
fi

echo "=========================================="
echo "Stage 4/9: base eval (BPB + CORE-subset)"
echo "=========================================="
$PY -m scripts.base_eval --run-dir "$BASE_RUN" --core-n 200 || echo "(base_eval skipped/partial)"

echo "=========================================="
echo "Stage 5/9: prepare SFT data"
echo "=========================================="
if [ -f "$SFT_DATA" ] && [ "$(wc -l < "$SFT_DATA")" -ge 1000 ]; then
  echo "[skip] SFT data already at $SFT_DATA ($(wc -l < "$SFT_DATA") rows)"
else
  $PY -m dev.synth_smoltalk --out "$SFT_DATA" --n 5000
fi

echo "=========================================="
echo "Stage 6/9: supervised fine-tuning"
echo "=========================================="
$PY -m scripts.chat_sft \
  --base-run-dir "$BASE_RUN" \
  --sft-data "$SFT_DATA" \
  --run-dir "$SFT_RUN" \
  --n-iters 400 \
  --max-seq-len 1024

echo "=========================================="
echo "Stage 7/9: chat eval after SFT"
echo "=========================================="
$PY -m scripts.chat_eval --run-dir "$SFT_RUN" --core-n 200 --gsm8k-n 50 || echo "(chat_eval skipped/partial)"

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
  --m-prompts 4 \
  --g-rollouts 4 \
  --max-new-tokens 128

echo "=========================================="
echo "Stage 9/9: chat eval after RL"
echo "=========================================="
$PY -m scripts.chat_eval --run-dir "$RL_RUN" --core-n 200 --gsm8k-n 100 || echo "(chat_eval skipped/partial)"

echo
echo "All stages finished. Inference:"
echo "  CLI:    $PY -m scripts.chat_cli    --run-dir $RL_RUN"
echo "  Web UI: $PY -m scripts.chat_server --run-dir $RL_RUN --port 8000"
