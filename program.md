# jaxchat — Convergence Program for 8×RTX 6000 (124M)

**One goal**: lower validation BPB, faster time-to-target-loss on 8×RTX 6000 GPUs.

This document defines the experiment configuration, the complete feature set, ablation priorities, and the SLURM job to submit.

---

## Target: 124M Param Model on 8×RTX 6000

| Property | Value |
|----------|-------|
| Preset | `124m` (125.8M params) |
| Hardware | 8 × RTX 6000 (48 GB each) |
| Data | FineWeb-Edu, 32K vocab, ~900M tokens |
| Token budget | ~1.3B (Chinchilla, 10.5× param count) |
| Baseline steps | ~5,000 |
| Expected wall time | ~4-8 hours |

---

## Full Feature Set (All ON)

Every feature below is enabled for the 124m run. After we have a baseline, we ablate individual features (see Ablation Plan).

### Architecture

| Feature | Config Field | Value |
|---------|-------------|-------|
| RoPE | `rope_base` | 10000.0 |
| QK-Norm (RMSNorm) | — | Always on (after RoPE) |
| ReLU² activation | — | Always on |
| RMSNorm | — | Always on (pre-attn, pre-mlp, final) |
| DeepNorm init | `init_style` | `"deepnorm"` — zero-init output projections, scaled residual |
| Embedding scaling | `scale_embedding` | `True` — `wte *= sqrt(d_model)` |
| GQA (Grouped-Query Attention) | `n_kv_heads` | `4` (n_heads=8 for depth=12) → 2× fewer KV heads |

### Optimizer

| Feature | Config Field | Value |
|---------|-------------|-------|
| MuonAdamW | `optimizer` | `"muon_adamw"` — AdamW for embed/scalars/lm_head, Muon for matrices |
| Schedules | `lr_schedule` | `"wsd"` — Warmup-Stable-Decay |
| Gradient clipping | `max_grad_norm` | `1.0` — prevents Muon instability |
| Z-loss | `z_loss_coeff` | `1e-4` — keeps logits bounded |

### Attention

| Feature | Config Field | Value |
|---------|-------------|-------|
| Long-short hybrid | `use_long_short_attention` | `True` — even layers: short window, odd layers: full context |
| Sliding window | `sliding_window_pattern` | `(1024, 1024, 1024, 2048)` |
| Pallas/FA3 backend | `use_pallas_attention` | `True` — Triton-based attention on GPU |
| Ring attention | `use_ring_attention` | `True` — multi-GPU ring |
| GQA | `n_kv_heads` | `4` — repeat KV heads to 8 Q heads |

### Context Scheduling

| Feature | Config | Value |
|---------|--------|-------|
| Seq len warmup | `sequence_warmup_intervals` | `500` — linear ramp from `min_seq_len` to `max_seq_len` |
| Min seq len | `min_seq_len` | `512` |
| Max seq len | `max_seq_len` | `1024` |
| Tokens per step | `tokens_per_step` | `262144` |
| Joint schedule | `joint_schedule_points` | `()` — not needed for 124m |

### Residual & Skip Connections

| Feature | Config | Value |
|---------|--------|-------|
| Embedding→every block | `x0_lambdas` | Learned per-layer skip from embedding |
| Block skip | `skip_connections` | `((3, 6), (6, 9))` — skip from block 3→6, 6→9; ReZero init (0) |
| Value-path augmentation | — | Gated value embeddings at even layers |

### Logit Head

| Feature | Config | Value |
|---------|--------|-------|
| Softcap style | `logit_cap_style` | `"tanh"` — `softcap * tanh(x/softcap)` smoother gradients |
| Softcap value | `logit_softcap` | `15.0` |
| QK-Norm for head | `normalize_logits` | `True` — normalize x and lm_head weight before dot product |
| Z-loss | `z_loss_coeff` | `1e-4` |

### Weight Tying

| Feature | Config | Value |
|---------|--------|-------|
| Mode | `weight_tying` | `"delayed"` — tied for first 2/3, then untied |
| Untie step | `untie_at_step` | Auto: `int(0.66 * n_train_iters)` |

### Data & Document Handling

| Feature | Config | Value |
|---------|--------|-------|
| Cross-doc masking | `cross_document_mask` | `True` — mask loss at document boundaries |
| Doc sep token | `doc_sep_id` | `1` (BOS token used as separator in packed data) |

### Token Feature Enrichment

| Feature | Config | Value |
|---------|--------|-------|
| Bigram hash embed | `bigram_hash_embed` | `True` — extra embedding from hash(token_i, token_{i+1}) |
| Bigram buckets | `bigram_hash_buckets` | `16384` |

### Compiler & Runtime

| Feature | Value |
|---------|-------|
| XLA GPU flags | `--xla_gpu_enable_triton_gemm=True --xla_gpu_enable_cublaslt=True --xla_gpu_autotune_level=4` |
| AOT compilation | All unique shapes pre-compiled before training |
| Persistent cache | `/tmp/jax_cache` |

---

## Expected Impact

### vs Original 124m Baseline

| Metric | Baseline | With All Features | Source |
|--------|----------|-------------------|--------|
| Final Val BPB | ~1.45 | **~1.25-1.30** | WSD + init + gradient clipping + z-loss |
| Steps to target BPB (1.35) | ~full run | **~70% of tokens** | Seq warmup + bigram + better init |
| Training throughput | baseline | **~5-10% faster** | XLA flags + GQA |
| Stability | occasional spikes | **stable** | Grad clipping + z-loss + DeepNorm |

### Feature Contribution to Convergence (Estimated)

```
WSD schedule          ████████████████████░░  0.05-0.10 BPB
DeepNorm init         ██████████████░░░░░░░░  0.03-0.08 BPB
Grad clipping         ████████████░░░░░░░░░░  0.02-0.05 BPB
Z-loss                ████████████░░░░░░░░░░  0.02-0.05 BPB
Seq len warmup        ████████████░░░░░░░░░░  0.03-0.06 BPB
Delayed weight tying  ██████████░░░░░░░░░░░░  0.02-0.05 BPB
Bigram hash           ████████░░░░░░░░░░░░░░  0.02-0.04 BPB (early)
Long-short attn       ████████░░░░░░░░░░░░░░  0.02-0.04 BPB
Cross-doc mask        ██████░░░░░░░░░░░░░░░░  0.01-0.03 BPB
Tanh softcap          ████░░░░░░░░░░░░░░░░░░  0.01-0.03 BPB
QK-Norm for head      ████░░░░░░░░░░░░░░░░░░  0.01-0.02 BPB
Skip connections      ████░░░░░░░░░░░░░░░░░░  0.01-0.02 BPB (depth=12)
GQA speed             speed only, not loss
```

---

## Per-Layer Config for 124m (depth=12)

Since depth=12:
- `d_model = 12 × 64 = 768`
- `n_heads = 768 / 128 = 6` (wait — let me recalculate)

Actually for the 124m preset: `depth=8`, `d_model=512`, `n_heads=4`.

For the 124m preset specifically:
- `depth=8`, `d_model=512`, `n_heads=4`
- `n_kv_heads=2` (GQA with 2 KV heads, repeated to 4 Q heads) — **correction**
- `n_value_layers=4`

Long-short pairs:
- Layers 0,2,4,6: short window (1024)
- Layers 1,3,5,7: full context (1024)

Skip connections:
- Layer 2 → Layer 5
- Layer 5 → Layer 7

---

## SLURM Job: 8×RTX 6000

### Submit Script: `runs/rtx_8gpu_124m_modern.sh`

```bash
#!/usr/bin/env bash
#SBATCH --job-name=jaxchat-124m-modern
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_6000:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --output=/project/inniang/jaxchat/slurm-%A.out
#SBATCH --error=/project/inniang/jaxchat/slurm-%A.err
#SBATCH --export=ALL
#SBATCH --exclusive

set -euo pipefail
cd /project/inniang/jaxchat

export D4_ROOT="${D4_ROOT:-/project/inniang/jaxchat/data/124m_rtx_run}"
export WANDB_PROJECT="${WANDB_PROJECT:-jaxchat}"
export WANDB_DIR="${WANDB_DIR:-/project/inniang/jaxchat/logs/wandb}"

# GPU + JAX setup
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false \
  --xla_gpu_enable_triton_gemm=True \
  --xla_gpu_enable_cublaslt=True \
  --xla_gpu_autotune_level=4"

mkdir -p "$WANDB_DIR" "$D4_ROOT"

unset VIRTUAL_ENV
export OMP_NUM_THREADS=1

TOKENIZER_DIR="${D4_ROOT}/tokenizer"
DATA_DIR="${D4_ROOT}/fineweb32k"
BASE_RUN="${D4_ROOT}/runs/base"

mkdir -p "$TOKENIZER_DIR" "$DATA_DIR" "$BASE_RUN"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync

PY="uv run python -u"

# ── Stage 1: Tokenizer (copy existing 32K) ──
if [ -f "${TOKENIZER_DIR}/tokenizer.json" ]; then
  echo "[skip] tokenizer exists"
else
  cp /project/inniang/jaxchat/data/fineweb32k/tokenizer.json "${TOKENIZER_DIR}/tokenizer.json"
fi

# ── Stage 2: Data ──
if compgen -G "${DATA_DIR}/fineweb_train_*.bin" >/dev/null 2>&1 && [ -f "${DATA_DIR}/fineweb_val_000000.bin" ]; then
  echo "[skip] data exists"
else
  ln -sf /project/inniang/jaxchat/data/fineweb10B/*.bin "${DATA_DIR}/" 2>/dev/null || true
  # Also copy base tokenizer files
  cp /project/inniang/jaxchat/data/fineweb32k/*.json "${DATA_DIR}/" 2>/dev/null || true
fi

# ── Stage 3: Base pretraining (124M) WITH ALL MODERN FEATURES ──
$PY -m scripts.base_train \
  --preset 124m \
  --input-bin "${DATA_DIR}/fineweb_train_*.bin" \
  --input-val-bin "${DATA_DIR}/fineweb_val_000000.bin" \
  --tokenizer-json "${TOKENIZER_DIR}/tokenizer.json" \
  --run-dir "$BASE_RUN" \
  --optimizer muon_adamw \
  --lr-schedule wsd \
  --weight-tying delayed

echo "=========================================="
echo "Base pretraining complete!"
echo "Checkpoints in: $BASE_RUN"
echo "Wandb project: $WANDB_PROJECT"
echo "=========================================="
```

### Submit It

```bash
sbatch runs/rtx_8gpu_124m_modern.sh
```

---

## Ablation Plan

After the all-features run completes, we isolate individual contributions:

### Priority Order (by expected impact)

| Priority | Ablation | Hypothesis | Expected BPB shift |
|----------|----------|------------|-------------------|
| P0 | `lr_schedule="linear"` (no WSD) | WSD gives the biggest single win | +0.05-0.10 BPB (worse) |
| P0 | `max_grad_norm=0` (no clipping) | Loss spikes without clipping | +0.02-0.05 BPB |
| P1 | `init_style="default"` (no DeepNorm) | DeepNorm stabilizes early training | +0.03-0.08 BPB |
| P1 | `z_loss_coeff=0` (no z-loss) | Logit saturation without z-loss | +0.02-0.05 BPB |
| P1 | `weight_tying="none"` (no delay) | Delayed tying helps both phases | +0.02-0.05 BPB |
| P1 | `sequence_warmup_intervals=0` (no warmup) | Warmup helps early convergence | +0.03-0.06 BPB |
| P2 | `bigram_hash_embed=False` | Small early boost | +0.02-0.04 BPB |
| P2 | `use_long_short_attention=False` | Local context helps | +0.02-0.04 BPB |
| P2 | `cross_document_mask=False` | Cleaner signal at boundaries | +0.01-0.03 BPB |
| P2 | `logit_cap_style="sigmoid"` | Tanh smoother than sigmoid | +0.01-0.03 BPB |
| P3 | `normalize_logits=False` | Small head stability | +0.01-0.02 BPB |
| P3 | `skip_connections=()` (no skips) | Small residual benefit | +0.01-0.02 BPB |
| P3 | `n_kv_heads=4` (full MHA, no GQA) | GQA only speeds up, doesn't affect loss | no BPB change |

### Ablation Run Template

```bash
# Do one ablation at a time. Each gets a unique run-dir.

# Ablation: no WSD (use linear)
$PY -m scripts.base_train \
  --preset 124m \
  --input-bin "${DATA_DIR}/fineweb_train_*.bin" \
  --input-val-bin "${DATA_DIR}/fineweb_val_000000.bin" \
  --tokenizer-json "${TOKENIZER_DIR}/tokenizer.json" \
  --run-dir "${BASE_RUN}-no-wsd" \
  --lr-schedule linear \
  --weight-tying delayed
```

### Minimal Config (all features OFF for true baseline comparison)

```python
# Build by editing presets.py or passing dataclasses.replace args.
# Everything below is the original 124m behavior:
Config(
    lr_schedule="linear",
    max_grad_norm=0.0,
    z_loss_coeff=0.0,
    init_style="default",
    weight_tying="none",
    sequence_warmup_intervals=0,
    bigram_hash_embed=False,
    use_long_short_attention=False,
    cross_document_mask=False,
    logit_cap_style="sigmoid",
    normalize_logits=False,
    skip_connections=(),
)
```

---

## Monitoring & Success Criteria

### During Training (wandb)

Watch these metrics:

```
loss          → should drop smoothly, no spikes
val_bpb       → final target < 1.30 (aiming for 1.25)
grad_norm     → should stay < 10 with clipping=1.0
step_time_s   → should be consistent across shapes
```

### Defining "Better Convergence"

| Criterion | Measurement |
|-----------|-------------|
| Lower final BPB | `val_bpb` at last step vs baseline |
| Faster to target | Steps to reach `val_bpb=1.35` |
| Stability | Number of loss spikes > 2σ from rolling mean |
| Step consistency | Variance in per-step timing |

---

## File Map

| File | Purpose |
|------|---------|
| `jaxchat/optimizer.py` | MuonAdamW, NorMuon, SOAP factories |
| `jaxchat/schedules.py` | Seq len warmup, batch/joint scheduling |
| `jaxchat/data_mixer.py` | Cross-doc masking, dataset schedule |
| `jaxchat/token_features.py` | Bigram hash, PKO |
| `jaxchat/fa3.py` | Long-short attention, backend dispatch |
| `jaxchat/model.py` | Config with all new fields, init, forward |
| `training/train_base.py` | Training loop with new CLI args |
| `runs/rtx_8gpu_124m.sh` | **Original** baseline SLURM script |
| `runs/rtx_8gpu_124m_modern.sh` | **To create** — modernized SLURM script |
| `program.md` | This file |
| `TODO.md` | Full implementation status |

---

## Quick Reference: New CLI Args

```bash
python -m training.train_base \
  --preset 124m \
  --optimizer {muon_adamw,normuon,soap} \
  --lr-schedule {linear,cosine,wsd} \
  --weight-tying {none,full,delayed}
```

For full per-feature control, edit the Config in `presets.py` or pass `dataclasses.replace(PRESETS["124m"], lr_schedule="wsd", ...)` when constructing.
