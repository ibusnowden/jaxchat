#!/usr/bin/env bash
#SBATCH --job-name=jaxchat-chat-server
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_5000:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=/project/inniang/jaxchat/slurm-chat-%A.out
#SBATCH --error=/project/inniang/jaxchat/slurm-chat-%A.err

# Launch the FastAPI chat server on a SOTA chat checkpoint (defaults to the
# SFT stage of the depth-16 e2e run — RL added no learning signal). Binds
# 0.0.0.0:8000 on the compute node so the login node can curl it directly
# over the cluster network and a user can SSH-tunnel from a workstation.
# Override the served checkpoint with ``RUN_DIR=<...> sbatch ...``.

set -euo pipefail

PROJECT_DIR="/project/inniang/jaxchat"
cd "$PROJECT_DIR"

unset VIRTUAL_ENV
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/data/124m_sota_e2e/runs/sft}"
PORT="${PORT:-8000}"

echo "Host: $(hostname)  Port: $PORT  Run dir: $RUN_DIR"
echo "GPU(s): $(nvidia-smi -L)"

uv sync >/dev/null 2>&1
exec uv run python -u -m scripts.chat_server \
  --run-dir "$RUN_DIR" \
  --host 0.0.0.0 \
  --port "$PORT"
