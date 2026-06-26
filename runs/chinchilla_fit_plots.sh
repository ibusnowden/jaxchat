#!/usr/bin/env bash
#SBATCH --job-name=chinchilla-fit
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/project/inniang/jaxchat/slurm-chinchilla-fit-%j.out
#SBATCH --error=/project/inniang/jaxchat/slurm-chinchilla-fit-%j.err
#SBATCH --export=ALL

set -euo pipefail

cd /project/inniang/jaxchat

unset VIRTUAL_ENV
export OMP_NUM_THREADS=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/project/inniang/jaxchat/data/124m_rtx_run/uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/jaxchat-matplotlib}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not on PATH." >&2
  exit 1
fi
PY="uv run --no-sync python -u"

$PY -m scripts.fit_chinchilla \
  --runs-root /project/inniang/jaxchat/data/124m_rtx_run/runs/chinchilla \
  --out-dir /project/inniang/jaxchat/data/124m_rtx_run/runs/chinchilla/_fit

echo "Chinchilla fit/plots regenerated — $(date)"
