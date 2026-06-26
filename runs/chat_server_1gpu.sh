#!/usr/bin/env bash
#SBATCH --job-name=jaxchat-chat-1gpu
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/project/inniang/jaxchat/slurm-chat-1gpu-%A.out
#SBATCH --error=/project/inniang/jaxchat/slurm-chat-1gpu-%A.err
#SBATCH --export=ALL

# ============================================================================
# jaxchat inference server — 1 GPU, single-device, new web UI.
#
# Serves the last RL checkpoint (data/0p5b_e2e/runs/rl_chinchilla, step 79,
# 529.5M params) with scripts.chat_server's FastAPI web UI + SSE streaming.
#
# The checkpoint was trained across an 8-GPU mesh (mesh_shape=(8,)), so we pass
# --single-device to collapse it to a 1-device mesh for inference on one card.
# Binds 0.0.0.0:8000 on the compute node. SSH-tunnel from a workstation:
#     ssh -L 8000:<compute-node>:8000 <login-host>
# then open http://localhost:8000
#
# Override at submit time:
#   sbatch --export=ALL,PORT=8123,RUN_DIR=.../runs/sft_chinchilla runs/chat_server_1gpu.sh
#   sbatch --export=ALL,STAGE=sft runs/chat_server_1gpu.sh
# Cancel with: scancel <jobid>
# ============================================================================

set -euo pipefail

PROJECT_DIR="/project/inniang/jaxchat"
cd "$PROJECT_DIR"

unset VIRTUAL_ENV
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Last RL run by default; allow override.
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/data/0p5b_e2e/runs/rl_chinchilla}"
STAGE="${STAGE:-rl}"
TOKENIZER="${TOKENIZER:-${PROJECT_DIR}/data/fineweb32k_real_29/tokenizer.json}"
PORT="${PORT:-8000}"
MAXNEW="${MAXNEW:-256}"

echo "=== jaxchat 1-GPU chat server ==="
echo "Host: $(hostname)  GPUs: $(nvidia-smi -L 2>/dev/null || echo 'n/a')"
echo "Run dir: ${RUN_DIR}  (stage=${STAGE})"
echo "Tokenizer: ${TOKENIZER}"
echo "Port: ${PORT}  max_new_tokens=${MAXNEW}"
echo "================================="

uv sync >/dev/null 2>&1

exec uv run --no-sync python -u -m scripts.chat_server \
  --run-dir "${RUN_DIR}" \
  --stage "${STAGE}" \
  --tokenizer-json "${TOKENIZER}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max-new-tokens "${MAXNEW}" \
  --single-device
