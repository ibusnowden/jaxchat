# jaxchat — 124M ablation log & idea list

A running log of experiments on the `124m` preset (8×RTX 6000), in the spirit of
[nanochat's `dev/LOG.md`](https://github.com/karpathy/nanochat/blob/master/dev/LOG.md)
and [discussion #481 "Test ideas that did not work"](https://github.com/karpathy/nanochat/discussions/481).
Goal (`program.md`): lower validation BPB / faster convergence for a ~98M-param model trained
through the jaxchat pipeline.

**The setup.** `124m` ⇒ depth 8, `d_model=512`, `n_heads=4`, `n_kv_heads=2` (GQA), `vocab=32768`,
`n_value_layers=2`. Param split: ~67M embedding tables (`wte` 16.8M + `value_embeds` 33.6M + `lm_head` 16.8M),
~23M trainable transformer matrices, +8.4M bigram-hash table. So **the model is ~70% lookup tables and only
~23M of "real" transformer** — and the Chinchilla-ish token budget is sized off `total − wte − lm_head`
(`train_token_ratio=10.5` ⇒ ~683M tokens). Data: FineWeb-Edu re-tokenized to the 32K BPE
(`data/fineweb32k_real/`, ~912M train / ~101M val tokens, disjoint). All numbers are `val_bpb`
(true bits-per-byte; ~4.2 bytes/token); the val set is small (65 536 tokens, 1 batch) so treat ±0.02 as noise.
Throughput: ~1.2 s/step, `mfu_proxy ≈ 0.016` — every attention layer falls back to SDPA (the Pallas/FA3
GPU kernels are rejected on RTX 6000 for these shapes), and the optimizer state is replicated across the
8 ranks rather than sharded. (Heads-up: `steady_state_step_s` / `mfu_proxy` in the "Timing summary" line are
divided by `n_train_iters`, so they look ~2.5× better than reality on a *resumed* run — use wall-clock.)

## Results so far

| # | run | preset / change | tokens (steps) | weight_tying | val_bpb | notes |
|---|---|---|---|---|---|---|
| 1 | baseline | `124m`, linear LR, no modern feats | 617M (2352) | none | **1.3152** | the reference |
| 2 | modern | `124m-modern` (all feats, WSD) | 683M (2604) | delayed | **1.2249** | had the untie spike (#A) |
| 3 | modern, no-WSD | + `lr_schedule=linear` | 683M (2604) | delayed | **1.2273** | — |
| 4 | modern, default-init | + `init_style=default` (drop DeepNorm) | 683M (2604) | delayed | **1.1966** | (had untie spike too) |
| 5 | modern, **long** | `124m-modern` | **1.31B (5000)** | delayed | **1.1068** | spike@3333 cost ~1000 steps |
| 6 | modern, long, **no-tie** | `124m-modern` + `weight_tying=none` | **1.31B (5000)** | none | **0.8878** | clean monotone descent |
| 7 | modern, **xlong** | `124m-modern` (none) | 2.1B (8000) | none | _in progress_ | ~2.3 epochs |
| 8 | **124m-loop** | `n_recurrence=2` (eff. depth 16) | 683M (2605) | none | _in progress_ | weight-shared loop |
| 9 | modern, no-tie @683M | matched control for #8 | 683M (2605) | none | _in progress_ | |

## What we've learned

### ✅ Worked / mattered
- **Don't use `weight_tying=delayed`.** Re-seeding `lm_head ← wteᵀ` at 2/3 of training (`train_base.py`)
  spikes `val_bpb` from ~1.27 → **~2.93** and the run never fully recovers when the LR there is still
  meaningful (observed at both untie@1736 in #3 and untie@3333 in #5). Switching to plain `weight_tying=none`
  (independent `lm_head` from init) is a clean monotone descent and won the head-to-head by **−0.22 BPB at
  identical budget** (#5 → #6: 1.1068 → 0.8878). → `PRESET_124M_MODERN` default changed `delayed`→`none`.
- **More tokens dominates everything else.** 683M → 1.31B (with `none`) took `val_bpb` 1.22 → 0.89, and the
  curve was still falling ~0.06/1000 steps at the budget cutoff. The token budget here is the lever; the
  bottleneck on it is throughput (see ideas).

### ❌ Didn't help (or barely)
- **WSD vs. linear LR**: ≈ **0.002 BPB** (#2 vs #3). `program.md` predicted 0.05–0.10. Both schedules have an
  annealing tail; at this scale the WSD-specific shape isn't the win it's billed as.
- **DeepNorm init vs. default init**: ≈ **noise** (#2 vs #4 — default was actually *slightly* better).
  Predicted +0.03–0.08. With QK-norm + RMSNorm everywhere + grad-clip + z-loss already on, the init style
  doesn't move the needle.
- Net: the "modern feature set" as a bundle is worth ~0.1 BPB over plain `124m` (#1 vs #2), and most of that
  isn't the headline features — it's likely the value-embeds + bigram + GQA bookkeeping. Worth re-checking
  each against the new `none` baseline before claiming anything.

## Ideas to try (roughly priority order)

1. **Looped / recursive transformer** — ✅ *implemented this round* (`Config.n_recurrence`, `PRESET_124M_LOOP`).
   Apply the block stack `n_recurrence` times, weight-shared, with a per-loop "timestep" embedding added to the
   residual stream (the Saunshi "Reasoning with Latent Thoughts" / Kevin671 2410.01405 recipe — the timestep
   embed breaks the fixed-point symmetry that otherwise collapses a reused block). Effective depth = `n_layers ×
   n_recurrence` at `n_layers` params. This is the most natural fit for *this* model: it's FLOP-cheap and
   param-bottlenecked (~23M transformer, budget sized off that), so recurrence buys "free compute per param".
   First test: `n_recurrence=2` (eff. depth 16) vs the matched non-looped control at 683M (#8 vs #9). If it
   wins, re-run at 1.31B+ and make it the default; then sweep `n_recurrence ∈ {3,4}` and try
   per-loop LR / per-effective-layer `resid_lambdas`.
2. **Throughput pass** (biggest indirect win on `val_bpb`, since more tokens = lower loss):
   - Get a flash-attention kernel that actually engages on RTX 6000 for these shapes (head_dim=128, n_heads=4,
     sliding-window / long-short combos currently force SDPA) — cuDNN SDPA flash path or a Pallas/Triton kernel.
   - ZeRO-1 optimizer-state sharding (one slice of Muon/AdamW state per rank) — currently fully replicated.
   - Re-test `--xla_gpu_enable_triton_gemm=True` on RTX 6000 Ada + JAX 0.9.1 (was disabled for breaking autotune).
   - Target: `mfu_proxy` ≫ 0.016, then spend the saved wall-clock on tokens.
3. **depth=12** — `d_model=768`, ~78M transformer matrices (≈3.4× the depth-8 net), token budget auto-grows to
   ~1.5B. Conventional "more capacity" bet; downside is it re-grows `wte`/`lm_head` to ~25M each. Run after the
   loop result so we know whether recurrence or width is the better use of the FLOPs.
4. **muP-style LR sweep** on the locked config — a few % of steps could come off; the optimizer LRs
   (`muon_base_lr`, `embed_lr_base`, `lm_head_lr_base`, …) were never tuned for this scale.
5. **Re-run the P0–P3 single-feature ablations against the new `weight_tying=none` baseline** so the
   contribution table is honest (`runs/rtx_124m_ablation.sh <name>` does one per job: `no-wsd`, `no-clip`,
   `default-init`, `no-zloss`, `no-tying`, `no-seqwarmup`, `no-bigram`, `no-longshort`, `no-crossdoc`,
   `sigmoid-cap`, `no-skip`, `full-mha`).
6. **Curriculum / data ordering** — e.g. a difficulty-sorted warmup (train a tiny doc-difficulty classifier on
   an LLM-labelled subset, then sample easy→hard for the first 10–20%). Holding model+optimizer+compute fixed,
   how much variance is data ordering vs. seed? (Worth a seed-variance run first to know what "significant" means.)
7. **Optimizer swaps** — Dion (Microsoft) instead of Muon; NorMuon / SOAP are already implemented but unbenchmarked here.
8. **Low-rank value embeds** — `value_embeds` is 33.6M (the second-biggest table); `(n_value_layers, vocab, r)·(n_value_layers, r, d_model)` with `r=d_model//4` would reclaim ~25M of params for the transformer (or shrink the model).

## Tooling added this round
- `scripts/base_train.py --config-override KEY=VALUE` — repeatable, type-coerced from the `Config` dataclass.
  (Gotcha: the modern preset stores *resolved* `target_train_tokens`/`n_train_iters`, so overriding
  `train_token_ratio` alone is a no-op — set `n_train_iters` directly + `untie_at_step=-1`.)
- `runs/rtx_124m_ablation.sh <name>` — single-ablation driver: one `--preset 124m-modern` run + `base_eval`.
- `training/eval_base.py` / `scripts/base_eval.py` — fixed the post-training crash (DP-on-batch-axis sharding
  spec → `P(*config.activation_sharding)`; eager-sampling mesh-context → `with mesh, jax.set_mesh(mesh)`);
  added `--skip-generation` (eager autoregressive decode re-traces per token across 8 GPUs → minutes).

---

## Reference notes (from nanochat discussion #481 / the original modernization plan)

# https://github.com/karpathy/nanochat/blob/master/dev/LOG.md
# Compares ideas with our original baseline  baseline vs single ablation.
Think about optimizing this pretraining code for rtx gpu
https://github.com/karpathy/nanochat/discussions/481
- Trition-jax attention
- FA4 -> https://github.com/karpathy/nanochat/pull/609
- Long tokenizer from google deepresearch char skill2.md
- Attention residual -> https://github.com/karpathy/nanochat/pull/646
- ZeRO-3 style sharding: each rank owns a slice of optimizer state
- Fitting scaling laws (124M for ablations before testing on 1.5B target eval core bpb+ wall clock instead of val/loss)

These are the ones I didn’t keep/implement (might still work though):

- Value embeddings - couldn’t get either the vanilla implementation or the U-Net to work without a major slowdown earlier.
- Block sliding window - I was lazy and didn’t want to implement sliding window attention, though it’s certainly possible.
- FP8 - Not available on TPU v6e (though v7p will likely have it). I am considering int8 training for future versions, given that the flops and bytes per second both double.
- Batched Muon - I was lazy and did not implement this.
- Custom hardware/PyTorch optimizations - Skipped custom communication strategies and other parallelism tricks from the official speedrun.
- mup sweep -> More hparam tuning could shave off some steps.

Questions?
- The question remains: why is HBM utilization only ~50%? Some of the candidates could be suboptimal overlapping, kernel launch overhead, or communication issues not fully saturating the memory bus.

- The official speedrun saw some benefit from manual communication primitives. This could be explored in JAX.

- Custom kernels in Pallas (lowers to Mosaic on TPUs and Triton on GPUs):
    - A Pallas kernel for block-sparse flex attention could be useful. I wasted a lot of time trying to integrate a kernel that claimed   to do this but wasn’t actually block-sparse.
    - I tried integrating existing Flash/Splash Attention implementations, but they didn’t work on the first attempt, so I moved on. It would be great if someone could get these working.
    - A custom Pallas kernel for the cut cross entropy loss could help improve MFU (according to someone whose MFU on GPUs got fixed after using a CUDA kernel for that).
- The computation for the optimizers likely happens on every PyTree leaf individually and with replicated computation across shards. This may or may not a bottleneck right now, and could be helpful to keep in mind (to avoid extraneous computation/unfused ops). Sharding computation requires using an all-gather which incurs overhead and should be measured, as with everything else.
- Using Microsoft’s Dion optimizer instead of Muon could be interesting.
- Going a bit further, trying out other parallelism strategies like FSDP/TP/hybrid strategies could be looked into.

- holding architecture + optimizer + compute fixed, have you ever tried systematically varying the data curation / filtering regime, rather than the training stack, and measuring how much variance shows up downstream?

- The first thing that came to my mind was to maybe try curriculum learning. It would look something like this :

  1 - Fix everything (seed, model depth, token budget, eval interval).
  2 - Compare:

Baseline random sampling
Curriculum warmup (first 10–20% tokens) then baseline sampling
I assume this would involve training a classifier first to score documents from Fineweb-Edu according to their "difficulty" (score categories could be 0-5). For this we can build a tiny subset of documents annotated by a LLM and train the classifier on it.n

- Have you ever tried to train the same (small) model with different training data permutations, and maybe also with different random seeds. I wonder what the random variance is for the same architecture, and whether minor differences in validation loss of different architectural tweaks are really meaningful.


Also from nanochat discussion 481
Test ideas that did not work.
