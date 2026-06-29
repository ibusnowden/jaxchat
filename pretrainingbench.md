# PretrainingBench — 1×H100 data-constrained pretraining sweep

> **What this eval measures.** LLM *pretraining ability*: the agent must reason
> about LLM training, diagnose training bottlenecks, make useful changes to the
> training loop, tune hyperparameters, and produce a training script that
> reaches a target validation objective **efficiently** on a single H100 —
> balancing model quality, training time, and compute usage.  A strong model
> diagnoses what is limiting convergence and improves empirical performance.
>
> **The central question (not just curve-fitting).** The point is *not* to
> re-fit Kaplan/Chinchilla on a standard web corpus.  Existing data-constrained
> scaling laws — Muennighoff et al. (2023) and Lovelace et al. — are built from
> **empirical curve fitting**: it is still unclear *why* they should take
> exactly these functional forms and *why each free parameter is needed*.  This
> sweep is set up to probe that gap on the **NVIDIA Nemotron** corpus, in the
> repeated-data (multi-epoch) regime where those forms actually bind — treating
> the fitted exponents not as the deliverable but as evidence about the
> *mechanism* of data-constrained scaling, and pointing toward more theoretical
> work on why the laws have the shape they do.

This file is the plan + runbook for the jaxchat PretrainingBench effort.
The code lives in:

| File | Role |
|------|------|
| `scripts/pretrainingbench.py` | Grid generator: 20 models, Chinchilla-optimal token budgets, 1×H100 feasibility flags. |
| `scripts/fit_scaling_law.py`   | Fits Kaplan, Chinchilla (IsoFLOP), and Muennighoff data-constrained laws to the sweep results. |
| `runs/h100_pretrainingbench.sh` | 1×H100 SLURM array driver: train → eval → fit, one task per depth. |
| `scripts/chinchilla_grid.py`   | The existing 8×RTX-6000 IsoFLOP sweep (124m-modern family, depths 4–20). |
| `scripts/fit_chinchilla.py`    | The existing IsoFLOP fit/plot pipeline (consumed by `fit_scaling_law` for log parsing). |

---

## 1. Setup

* **Hardware:** 1 × H100 80 GB (`--gres=gpu:h100_80gb:1`).
* **Corpus:** the NVIDIA **Nemotron pre-training datasets**
  (<https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets>) —
  specifically a clean 32 K-BPE re-tokenization of **Nemotron-CC** (the curated
  Common-Crawl web split) into `data/nemotron_cc_32k` (~2.94 B tokens for this
  sweep), with the **Nemotron-CC-Math** and **Nemotron-Pretraining** code/SFT
  splits available for mixing.  A single vocab is held fixed across the sweep so
  the embedding tax is constant and width is the only free axis (Kaplan 2020,
  Chinchilla 2022).  Pass `--vocab 4096` with a matching 4 K shard set to push
  the floor from ~15 M toward ~2 M.
* **Architecture:** a decoder-only Transformer (Vaswani et al., 2017) in the
  `124m-modern` preset, updated to a Gemma-3-style **periodic local/global
  attention** layout with **dense SwiGLU** feed-forward blocks (we stay dense —
  see *Architecture details* below).  RMSNorm (Zhang & Sennrich, 2019) is applied
  at both the input and the output of every block, immediately before the
  residual add (Gemma Team 2025); no biases anywhere; input and output embeddings
  are tied.  Retains the SOTA feature set: WSD schedule, DeepNorm init, grad-clip
  1.0, z-loss 1e-4, bigram-hash embed, cross-document mask, tanh logit cap.  Only
  `depth` (and thus `d_model = depth*64`) and `vocab` vary.
* **Optimizer:** MuonAdamW (the default; the SOAP/NorMuon bugs fixed in
  `jaxchat/optimizer.py` are prerequisites for any SOAP/NorMuon ablation).

### Architecture details

**Tokenizer.** Production uses the `o200k_base` tokenizer (vocab 200,019;
OpenAI 2022, Hurst et al. 2024) for compatibility with existing in-house tools.
The sub-billion sweep instead holds a fixed *small* vocab (32 K, or 4 K via
`--vocab`) so the embedding tax is constant and width is the only free axis —
the 200 K production vocab would dominate the parameter count below ~100 M and
contaminate the scaling fit.

**Attention.** Following Gemma 3, we pair **5 local attention layers** (Beltagy
et al., 2020) with **1 global attention layer**, which sharply reduces attention
compute in training and KV-cache size at inference. Local layers use RoPE
(Su et al., 2023) with a **512 sliding window** and base frequency 10,000;
global layers use **no position encoding** (NoPE; Kazemnejad et al., 2023),
which matches RoPE while being cheaper. We use **group-query attention**
(Ainslie et al., 2023) with a per-head dim of 128 and apply **RMSNorm to the
queries and keys** (QK-norm; Henry 2020, Dehghani 2023, Wortsman 2023). These
standard components let us leverage FlashAttention-4 and Ulysses-style context
parallelism (Jacobs et al., 2023) for efficient long-context training. *(In the
preset this is the `use_long_short_attention` path; the sweep keeps the existing
GQA KV-head count.)*

**Feed-forward — dense (no MoE).** We evaluated the Nemotron interleaved layout
that alternates **dense FFN and MoE** blocks — pairing a high-sparsity layer
with a zero-sparsity (dense) layer, which scales comparably to medium-sparsity
MoE throughout while being more wall-clock efficient under both iso-active and
iso-total parameter settings. That design keeps the **first FFN dense**
(DeepSeek 2025, Kimi 2026, Dai et al. 2024), uses SwiGLU (Shazeer 2020) for both
dense and MoE blocks, adopts the **LatentMoE** shared down-/up-projection around
the all-to-all (NVIDIA 2025) with routing on the *uncompressed* representation
(top-8 of 512 experts, softmax gating), **global-batch load balancing**
(GShard-style loss with empirical expert frequencies aggregated across DP workers
and micro-batches — the aggregation matters far more than the loss type) and a
fully **dropless** MoE to avoid capacity-dependent conclusions.

**For this sweep we stay with a purely dense SwiGLU model.** Dense keeps the
scaling axis clean (params ∝ depth only), removes routing/capacity/expert-balance
confounds from the Kaplan/Chinchilla/Muennighoff fits, and fits comfortably on a
single H100. The interleaved-MoE layout is recorded here as the *production*
target, not exercised in the sub-billion scaling sweep.

## 2. The 20-model grid

`python -m scripts.pretrainingbench --print-grid`

| task | depth | d_model | params | non_emb | tokens | iters | epochs | FLOPs | wall | mem | 1×H100 |
|----:|----:|----:|-----:|-----:|-----:|----:|-----:|-----:|----:|----:|:--:|
| 0 | 2 | 128 | 15 M | 6.7 M | 134 M | 511 | 0.05× | 1.2e16 | 0.5 m | 0.1 G | ✓ |
| 1 | 4 | 256 | 41 M | 24 M | 483 M | 1841 | 0.16× | 1.2e17 | 4.4 m | 0.2 G | ✓ |
| 2 | 6 | 384 | 66 M | 41 M | 818 M | 3121 | 0.28× | 3.2e17 | 12 m | 0.4 G | ✓ |
| 3 | 8 | 512 | 99 M | 65 M | 1.30 B | 4961 | 0.44× | 7.7e17 | 29 m | 0.6 G | ✓ |
| 4 | 10 | 640 | 137 M | 95 M | 1.90 B | 7251 | 0.65× | 1.6e18 | 58 m | 0.8 G | ✓ |
| 5 | 12 | 768 | 189 M | 138 M | 2.77 B | 10561 | 0.94× | 3.1e18 | 2.0 h | 1.1 G | ✓ |
| 6 | 14 | 896 | 248 M | 189 M | 2.94 B | 11216 | 1.00× | 4.4e18 | 2.7 h | 1.5 G | ✓ |
| 7 | 16 | 1024 | 327 M | 260 M | 2.94 B | 11216 | 1.00× | 5.8e18 | 3.6 h | 2.0 G | ✓ |
| 8 | 18 | 1152 | 414 M | 339 M | 2.94 B | 11216 | 1.00× | 7.3e18 | 4.6 h | 2.5 G | ✓ |
| 9 | 20 | 1280 | 530 M | 446 M | 2.94 B | 11216 | 1.00× | 9.3e18 | 5.8 h | 3.2 G | ✓ |
| 10–19 | 22–40 | 1408–2560 | 0.65–3.05 B | … | 2.94 B | 11216 | 1.00× | 1.1e19–5.4e19 | 7–34 h | 3.9–18 G | ✗ |

* **Token budget:** `D = min(20 × non_embedding_params, 2.94 B)` — Chinchilla-optimal
  (20× ratio) until the corpus is exhausted, then data-constrained (Muennighoff).
* **`epochs`** = `D / 2.94 B`.  From depth 14 on, the small models repeat the
  corpus (≥1 epoch) — this is the data-constrained regime the Muennighoff fit
  targets.
* **`wall`** = `6·N·D / (0.45 × 989 TFLOP/s)`; **`mem`** ≈ `6 × N` bytes
  (bf16 params + MuonAdamW state).  Points marked ✗ exceed the 6 h / 1.5 B-param
  1×H100 budget and are skipped by the driver (run them on a multi-GPU node).
* The 10 feasible points (15 M → 530 M, depths 2–20) cover the “1 M → 0.5 B”
  target on a single H100 in ≈ 20 h of wall time.

### Reaching toward 1 M

At vocab 32 K the floor is ~15 M (depth 2).  To extend toward ~1–2 M, use a
smaller tokenizer/data shard:

```
python -m scripts.pretrainingbench --print-grid --vocab 4096 --max-depth 22
```

| task | depth | params | tokens | wall | 1×H100 |
|----:|----:|-----:|-----:|----:|:--:|
| 0 | 2 | 4.1 M | 61 M | 0.1 m | ✓ |
| … | … | … | … | … | … |
| 10 | 22 | 490 M | 2.94 B | 5.4 h | ✓ |

(Requires a 4 K-vocab Nemotron-CC shard; build one with the data-prep helper
`--tokenizer-vocab-size 4096 …` pointed at the **Nemotron-CC** HF source — no
FineWeb-Edu is used anywhere in this sweep.)

## 3. Scaling laws fit

`python -m scripts.fit_scaling_law --runs-root <runs> --out-dir <out>`

Fits three laws to the completed (N, D, val_bpb) points:

1. **Kaplan** (arXiv:2001.08361):  `L(N, D) = E + A·N^{-α} + B·D^{-β}`,
   joint non-linear least squares (`scipy.optimize.curve_fit`).
   Reference: α ≈ 0.076, β ≈ 0.095.
2. **Chinchilla** (arXiv:2203.15556):  bin points into IsoFLOP buckets, fit a
   parabola in `log10(N)` per bucket, read `N*(C)` and `D*(C)`, then fit power
   laws `N* ∝ C^{a_N}`, `D* ∝ C^{a_D}`.  Reference: both ≈ 0.50.
   *(A single-vocab depth sweep gives one (N, D) per depth, so the IsoFLOP
   analysis is supplementary; Kaplan + Muennighoff are the primary laws here.)*
3. **Muennighoff data-constrained** (arXiv:2305.16264):  when `D > pool`,
   `D_eff = D_u + R_n·D_r` with `R_n = 1 − (1 − D_r/D_u)^n`, and we fit
   `L = E + A·N^{-α} + B·D_eff^{-β}`.  This down-weights repeated tokens and
   is the right law for the depth-14+ points that exhaust the corpus.  We also
   fit the **Lovelace et al.** data-constrained form for comparison.

Kaplan/Chinchilla on under-1-epoch points are the **baseline/control**; the
deliverable is the multi-epoch behaviour, so the data-constrained fits are the
primary objects of study here.

**Why these forms? (the actual research question).** Both the Muennighoff and
Lovelace data-constrained laws are **empirical curve fits** — the `R_n`
repeat-decay term and the `D_eff` substitution are chosen because they fit, not
derived from first principles, and it is unclear why each free parameter
(`E`, `A`, `B`, `α`, `β`, the repeat half-life in `R_n`) is *needed* rather than
an artifact of the fitting family.  So beyond reporting exponents we probe the
*mechanism*:

* **Parameter ablation / identifiability.**  Drop or fix each free parameter and
  measure the hit to held-out fit RMSE on the Nemotron multi-epoch points — a
  parameter that can be removed without cost is not load-bearing.  This directly
  tests "why is each free parameter needed."
* **Form falsification.**  Compare Muennighoff `R_n` decay vs the Lovelace form
  vs a plain `D_eff = D` (no repeat penalty) on the *same* points; if the curated
  Nemotron corpus repeats more gracefully than CommonCrawl, the fitted repeat
  half-life should move — evidence the parameter tracks a real data property, not
  the functional family.
* **Toward theory.**  Record where the empirical forms break (deepest /
  most-repeated points) as the concrete anchor for *why* a data-constrained law
  should have this shape — the open theoretical question motivating the sweep.

Outputs: `fit.csv` (all points), `scaling_laws.json` (fitted exponents +
predictions + per-parameter ablation deltas), and plots (`plot_kaplan.png`,
`plot_chinchilla.png`, `plot_data_constrained.png`, `plot_loss_vs_flops.png`).

**Verified on synthetic data:** with `L = 1.5 + 6/N^0.1 + 4/D_eff^0.3` the
fitter recovers `α = 0.100 ± 0.000`, `β = 0.300 ± 0.000`, `rmse = 0.0000`
for both Kaplan and Muennighoff.

## 4. Running the sweep

```bash
# Inspect the grid
uv run python -m scripts.pretrainingbench --print-grid

# Submit the 10 feasible tasks (depths 2–20, ~15 M–~530 M) on one H100
sbatch --array=0-9%1 runs/h100_pretrainingbench.sh

# Or the full 20-task sweep (tasks 10–19 self-skip on 1×H100)
sbatch runs/h100_pretrainingbench.sh

# Fit the laws once runs land
uv run python -m scripts.fit_scaling_law \
  --runs-root data/pbench/runs --out-dir data/pbench/_fit
```

Each array task: `base_train` (124m-modern preset, depth/vocab/tokens overridden
by the grid) → `base_eval` (val_bpb + CORE) → `fit_scaling_law` (read-only,
skips missing runs).  Overrides: `PBENCH_VOCAB`, `PBENCH_DEPTHS`,
`PBENCH_RATIO`, `PBENCH_DATA_POOL`, `PBENCH_ROOT`.

## 5. Hypotheses & what the sweep answers

* **Pretraining ability (the eval signal).**  Reaching the target val_bpb at the
  least wall-time/compute requires diagnosing the actual bottleneck per point
  (data-bound vs param-bound vs optimizer/LR vs throughput) and changing the
  training loop accordingly — that diagnosis, not the curve, is what the eval
  scores.
* **Why the data-constrained law has its form.**  On the Nemotron multi-epoch
  points, the per-parameter ablation (§3) should show which of `R_n`'s free
  parameters are load-bearing and which are fitting-family slack — the concrete
  handle on "why exactly this form / why each parameter."
* **Compute-optimal N*(C).**  The Chinchilla fit gives `N* ∝ C^{a_N}`; we
  expect `a_N ≈ 0.5` for the under-1-epoch points and a steeper slope once the
  corpus is exhausted (data-constrained optimum shifts to larger N).
* **Data-constrained decay.**  The depth-14+ points repeat the corpus; the
  Muennighoff `D_eff` fit should have a smaller `β` than the Kaplan `β` on raw
  `D`, quantifying how much repeat tokens are worth (the “scaling with repeated
  data” result, Muennighoff §3).
* **Loss vs FLOPs frontier.**  `plot_loss_vs_flops` shows, for a fixed compute
  budget, which depth is optimal — the direct answer to “what size model should
  I train for this H100-hour budget?”
* **Target validation objective.**  The Kaplan/Muennighoff fit predicts the
  val_bpb reachable at any (N, D) inside the sweep, so the “efficient” target
  is the point on the loss-vs-FLOPs frontier at the chosen budget.

## 6. References

* Kaplan et al. 2020 — *Scaling Laws for Neural Language Models* (arXiv:2001.08361)
* Hoffmann et al. 2022 — *Training Compute-Optimal LLMs* (Chinchilla) (arXiv:2203.15556)
* Muennighoff et al. 2023 — *Scaling Data-Constrained Language Models* (arXiv:2305.16264)
* Lovelace et al. — *data-constrained scaling law form* (compared against Muennighoff here)
* Bi & Lin 2024 — *PretrainingBench* (arXiv:2506.10972)
* Lilian Weng — *Scaling Laws* (2026-06-24) https://lilianweng.github.io/posts/2026-06-24-scaling-laws/
* NVIDIA — *Nemotron pre-training datasets* (collection) https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets
* Gemma Team 2025 — *Gemma 3* (periodic local/global attention, dual RMSNorm)
* Beltagy et al. 2020 — *Longformer* (sliding-window local attention) (arXiv:2004.05150)
* Su et al. 2023 — *RoFormer / RoPE* (arXiv:2104.09864)
* Kazemnejad et al. 2023 — *The Impact of Positional Encoding (NoPE)* (arXiv:2305.19466)
* Ainslie et al. 2023 — *GQA: Grouped-Query Attention* (arXiv:2305.13245)
* Shazeer 2020 — *GLU Variants Improve Transformer (SwiGLU)* (arXiv:2002.05202)
* Zhang & Sennrich 2019 — *RMSNorm* (arXiv:1910.07467)
* NVIDIA et al. 2025 — *LatentMoE / Nemotron architecture* (interleaved dense+MoE, dropless)
