#!/usr/bin/env bash
#SBATCH --job-name=jaxchat-0p5b-diag
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=00:45:00
#SBATCH --output=/project/inniang/jaxchat/slurm-0p5b-diag-%A.out
#SBATCH --error=/project/inniang/jaxchat/slurm-0p5b-diag-%A.err
#SBATCH --export=ALL
#SBATCH --exclusive

# End-to-end chat smoke test for the 0.5B pipeline: loads sft_math2 + rl_math2
# checkpoints, chats, and prints raw GSM8K generations (frac_format diagnosis).
# Tokenizer passed explicitly — the fineweb data dirs that used to carry a
# tokenizer.json copy were deleted 2026-06-12; the canonical one lives in
# data/124m_rtx_run/tokenizer/.

set -euo pipefail

PROJECT_DIR="/project/inniang/jaxchat"
cd "$PROJECT_DIR"

unset VIRTUAL_ENV
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false \
  --xla_gpu_enable_cublaslt=True \
  --xla_gpu_autotune_level=4"

export UV_CACHE_DIR="${PROJECT_DIR}/data/0p5b_e2e/uv-cache"

"${PROJECT_DIR}/.venv/bin/python" "${PROJECT_DIR}/dev/diag_backend_bisect.py"
