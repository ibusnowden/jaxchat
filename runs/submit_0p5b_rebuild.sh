#!/usr/bin/env bash
# ============================================================================
# Submit the post-bigram-fix 0.5B rebuild as an afterok dependency chain:
#
#   A  base pretrain (~46h)   rtx_8gpu_0p5b_base_chinchilla.sh  -> base_chinchilla
#        (+ base generation sanity GATE — the bigram-fix acceptance test)
#   B  smoltalk SFT (~3h)     rtx_8gpu_0p5b_sft_smoltalk.sh     -> sft_chinchilla
#        (+ chat sanity GATE — this IS the working chatbot)
#   C  math-SFT + shaped GRPO rtx_8gpu_0p5b_mathsft_rl.sh        -> sft_math, rl_math
#        (RL_PROXIMITY_COEF=0.3 = the job-114571 live-gradient recipe; + sanity report)
#
# B starts only if A succeeds (afterok), and A's tail exits nonzero on the
# broken-Engine generation signature — so a still-leaky base halts the chain
# instead of wasting GPU on SFT/GRPO. Likewise C waits on B.
#
# Prereq: the FineWeb pool must be rebuilt first (deleted 2026-06-12). Build it
# on the LOGIN node (compute nodes are offline):
#   .venv/bin/python -m dev.build_fineweb_pool \
#     --output-dir data/fineweb32k_real_29 \
#     --tokenizer-json data/124m_rtx_run/tokenizer/tokenizer.json \
#     --shards 30 --shard-tokens 100000000 --seq-len 1024
#
# Usage:  bash runs/submit_0p5b_rebuild.sh
# ============================================================================

set -euo pipefail

PROJECT_DIR="/project/inniang/jaxchat"
cd "$PROJECT_DIR"
RUNS="${PROJECT_DIR}/runs"
DATA_DIR="${PROJECT_DIR}/data/fineweb32k_real_29"

# --- Preflight: data pool must be present and non-trivial ---
if [ ! -f "${DATA_DIR}/fineweb_val_000000.bin" ] || ! compgen -G "${DATA_DIR}/fineweb_train_*.bin" >/dev/null 2>&1; then
  echo "ERROR: pretraining pool not found at ${DATA_DIR}." >&2
  echo "       Rebuild it on the login node first (see header of this script)." >&2
  exit 1
fi
if [ ! -f "${DATA_DIR}/tokenizer.json" ]; then
  echo "ERROR: ${DATA_DIR}/tokenizer.json missing (base/eval read it)." >&2
  exit 1
fi
N_TRAIN=$(compgen -G "${DATA_DIR}/fineweb_train_*.bin" | wc -l)
echo "Data pool OK: ${N_TRAIN} train shards + 1 val shard at ${DATA_DIR}"

# --- Submit the chain ---
JOB_A=$(sbatch --parsable "${RUNS}/rtx_8gpu_0p5b_base_chinchilla.sh")
echo "A  base               -> job ${JOB_A}"

JOB_B=$(sbatch --parsable --dependency="afterok:${JOB_A}" "${RUNS}/rtx_8gpu_0p5b_sft_smoltalk.sh")
echo "B  smoltalk SFT       -> job ${JOB_B}  (afterok:${JOB_A})"

# GRPO live-gradient recipe (from job 114571): proximity bootstrap + a stronger
# 50%-math warmup so the (now-coherent, post-bigram-fix) rollouts can actually
# emit \boxed{} for the format/correctness reward to fire. Fresh _v3 dirs keep
# the old invalid-model outputs (sft_math/rl_math) for comparison.
JOB_C=$(sbatch --parsable --dependency="afterok:${JOB_B}" \
  --export=ALL,RL_PROXIMITY_COEF=0.3,REBUILD_DATA=1,MIX_N_SMOLTALK=8000,MIX_N_GSM8K=8000,MATHSFT_LR_SCALE=0.2,MATHSFT_N_ITERS=3000,SFT_MIX_DATA=${PROJECT_DIR}/data/0p5b_e2e/sft/sft_math_mix_v3.jsonl,MATHSFT_RUN=${PROJECT_DIR}/data/0p5b_e2e/runs/sft_math_v3,RL_RUN=${PROJECT_DIR}/data/0p5b_e2e/runs/rl_math_v3 \
  "${RUNS}/rtx_8gpu_0p5b_mathsft_rl.sh")
echo "C  math-SFT + GRPO    -> job ${JOB_C}  (afterok:${JOB_B})"

echo
echo "Chain submitted: ${JOB_A} -> ${JOB_B} -> ${JOB_C}"
echo "Watch:  squeue -u \$USER ; tail -f slurm-0p5b-chinchilla-${JOB_A}.out"
echo "Gates:  A fails if base generation shows the bigram-leak signature;"
echo "        B fails if the SFT chat model does. Either halts the rest."
