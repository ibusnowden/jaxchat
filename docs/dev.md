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

# test the inference stack
Prometheus metrics to expose:
  - inference_requests_total (counter, by model/status)
  - inference_tokens_per_second (gauge)
  - kv_cache_utilization (gauge, 0-1)
  - request_queue_depth (gauge)
  - time_to_first_token_seconds (histogram)
  - inter_token_latency_seconds (histogram)
  - batch_size (histogram)
  
Logging: structured JSON logs per request
  { request_id, prompt_tokens, completion_tokens, latency_ms, model }

## Runtime layout

- `engine.py`
  Shared inference core: checkpoint loading, prompt encoding, KV-cache sessions, cached decode, uncached reference path.
- `inference.py`
  CLI-only entrypoint backed by `InferenceEngine`.
- `server.py`
  FastAPI server with `/healthz`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`, and a browser chat at `/`.
- `benchmark_inference.py`
  Cached vs uncached latency comparison helper.

## Commands

- CLI:
  `python inference.py`
- FastAPI server:
  `python server.py --ckpt ./checkpoints/124m-tinystories`
- Benchmark:
  `JAX_PLATFORMS=cpu python benchmark_inference.py --max-tokens 48 --runs 3`

## Cached inference notes

- v1 uses learned positional embeddings plus a hard max-context cutoff.
- There is no continuous batching or speculative decoding yet.
- The current server is single-model and serializes active generations with one async lock per worker.

## Benchmark note

- Local CPU sanity run on a tiny random model (`d_model=32`, `n_layers=2`, `max_seq_len=16`) produced:
  `uncached_mean_s=0.4133`, `cached_mean_s=0.4112`, `speedup_x=1.01`, `completion_tokens=8`.
- Treat that as a wiring check only. Real gains need to be measured on the actual checkpoint and target hardware.
