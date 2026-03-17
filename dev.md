# 1. see nanochat file structure and single file implementation
# 2. go see the jax implementation and docs and write code by hand for a mental model
# 3. claude/codex review and help debug, no rush , no yolo ,we are here to learn.

Phase A — 124M speedrun in JAX

  - bf16 everywhere sensible
  - packed tokenized dataset
  - tiktoken now
  - RoPE + RMSNorm + QK norm + ReLU² + untied head
  - XLA Triton GEMM + latency-hiding flags
  - PGLE after baseline is stable
  - one-process-per-GPU distributed run on 8 H100s.

Phase B — 1.3B nanochat-style

   - keep the same training harness
   - increase scale only after you have stable throughput metrics
   - add sliding-window attention
   - then consider custom Pallas kernels where profiling shows a real hotspot.

Phase C — post-pretraining

   - instruction tuning
   - then RL -> GRPO
   - keep tiktoken for inference and serving unless you have a very specific deployment reason to replace it.

Notes:
Start with a 124M JAX speedrun harness that measures:
  - tokens/sec
  - step time
  - compile time
  - MFU proxy -> model flop utilization.
  - dataloader idle time
  - eval cadence cost
