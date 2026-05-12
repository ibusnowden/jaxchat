"""Central preset registry for jaxchat training stages.

Each preset is a fully-formed :class:`jaxchat.model.Config`.  They are exposed
from a single module so that scripts/, training/, and runs/ all pick them up
the same way and a new preset only needs to be added in one place.
"""

from __future__ import annotations

import dataclasses
import os

from jaxchat.model import Config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 32k-BPE FineWeb assets.  ``fineweb32k_real`` holds shards that were tokenized with
# the 32k tokenizer (see ``data/retokenize_bins.py``).  The older ``fineweb32k`` dir was
# a symlink to GPT-2-tokenized shards (vocab 50257) -- feeding those into a 32k-vocab
# model silently corrupts the embedding gather and labels, i.e. immediate loss=nan.
# Prefer the re-tokenized shards when present; fall back to the legacy dir otherwise.
_FINEWEB_32K_REAL_DIR = os.path.join(PROJECT_ROOT, "data", "fineweb32k_real")
_FINEWEB_32K_LEGACY_DIR = os.path.join(PROJECT_ROOT, "data", "fineweb32k")
FINEWEB_32K_DIR = (
    _FINEWEB_32K_REAL_DIR
    if os.path.isfile(os.path.join(_FINEWEB_32K_REAL_DIR, "fineweb_val_000000.bin"))
    else _FINEWEB_32K_LEGACY_DIR
)
FINEWEB_TRAIN_GLOB = os.path.join(FINEWEB_32K_DIR, "fineweb_train_*.bin")
FINEWEB_VAL_BIN = os.path.join(FINEWEB_32K_DIR, "fineweb_val_000000.bin")
FINEWEB_TOKENIZER_JSON = os.path.join(FINEWEB_32K_DIR, "tokenizer.json")

# Single-H100 d4 assets.  Tokenizer/data are produced by the d4 speedrun.
FINEWEB_8K_DIR = os.path.join(PROJECT_ROOT, "data", "fineweb8k")
FINEWEB_8K_TRAIN_GLOB = os.path.join(FINEWEB_8K_DIR, "fineweb_train_*.bin")
FINEWEB_8K_VAL_BIN = os.path.join(FINEWEB_8K_DIR, "fineweb_val_000000.bin")
FINEWEB_8K_TOKENIZER_JSON = os.path.join(FINEWEB_8K_DIR, "tokenizer.json")


DEFAULT_CONFIG = Config(
    input_bin=FINEWEB_TRAIN_GLOB,
    input_val_bin=FINEWEB_VAL_BIN,
    tokenizer_json=FINEWEB_TOKENIZER_JSON,
)

SMOKE = dataclasses.replace(
    DEFAULT_CONFIG,
    depth=4,
    vocab_size=4096,
    tokenizer_json="",
    min_seq_len=512,
    max_seq_len=512,
    tokens_per_step=8192,
    micro_batch_size=4,
    target_train_tokens=262144,
    val_tokens=8192,
    val_loss_every=16,
    save_every=0,
    use_pallas_attention=False,
    use_ring_attention=False,
)

# 1p384b-depth24: ~1.38B params, the original 24-depth preset.  (8 GPUs recommended.)
PRESET_1P384B_DEPTH24 = dataclasses.replace(DEFAULT_CONFIG, depth=24)

# 124m: depth-8 (d_model=512) transformer, sized for an 8-GPU RTX 6000 node.  Note: with
# vocab=32768 the input/output embeddings already cost ~33.5M params; ``n_value_layers=2``
# keeps the value-embedding tables at ~33.5M (vs ~67M at the depth//2=4 default) so the
# model isn't ~80% lookup tables.  The token budget is derived from non-(wte+lm_head) params.
_124m = Config(
    input_bin=FINEWEB_TRAIN_GLOB,
    input_val_bin=FINEWEB_VAL_BIN,
    tokenizer_json=FINEWEB_TOKENIZER_JSON,
    depth=8,
    n_value_layers=2,
    min_seq_len=1024,
    max_seq_len=1024,
    tokens_per_step=262144,
    micro_batch_size=4,
    target_train_tokens=0,
    val_loss_every=50,
    val_tokens=131072,
    save_every=200,
    activation_sharding=(None, None, None),
    use_pallas_attention=False,
    use_ring_attention=False,
)
PRESET_124M = _124m

# d4: ~11M params, sized to fit a single H100 from pretrain through RL.
D4 = dataclasses.replace(
    DEFAULT_CONFIG,
    input_bin=FINEWEB_8K_TRAIN_GLOB,
    input_val_bin=FINEWEB_8K_VAL_BIN,
    tokenizer_json=FINEWEB_8K_TOKENIZER_JSON,
    tokenizer_name="fineweb8k",
    depth=4,
    vocab_size=8192,
    min_seq_len=1024,
    max_seq_len=1024,
    tokens_per_step=32768,
    micro_batch_size=4,
    target_train_tokens=115_000_000,
    val_loss_every=50,
    val_tokens=131072,
    save_every=200,
    use_pallas_attention=False,
    use_ring_attention=False,
)

# 124m-modern: Proven stable features for best convergence.
_124m_modern = Config(
    input_bin=FINEWEB_TRAIN_GLOB,
    input_val_bin=FINEWEB_VAL_BIN,
    tokenizer_json=FINEWEB_TOKENIZER_JSON,
    depth=8,
    min_seq_len=512,
    max_seq_len=1024,
    sequence_warmup_intervals=500,
    tokens_per_step=262144,
    micro_batch_size=4,
    target_train_tokens=0,
    val_loss_every=50,
    val_tokens=65536,
    save_every=200,
    activation_sharding=(None, None, None),
    use_pallas_attention=False,
    use_ring_attention=False,
    n_value_layers=2,
    optimizer="muon_adamw",
    lr_schedule="wsd",
    init_style="deepnorm",
    max_grad_norm=1.0,
    z_loss_coeff=1e-4,
    # weight_tying: "delayed" re-seeds lm_head=wte^T at 2/3 training, which spikes
    # val_bpb ~1.27->2.93 and burns ~1000 steps recovering whenever the LR is still
    # meaningful there (observed on the 5000-step run, jobs 71107/71113).  Plain
    # "none" (independent lm_head from init) is a clean monotone descent and won the
    # A/B by ~0.22 BPB at equal budget (1.1068 delayed -> 0.8878 none).
    weight_tying="none",
    n_kv_heads=2,
    use_long_short_attention=True,
    bigram_hash_embed=True,
    cross_document_mask=True,
    doc_sep_id=0,  # packed sequences start with <|bos|> == id 0
    skip_connections=((2, 5), (5, 7)),
)
PRESET_124M_MODERN = _124m_modern

# 124m-loop: looped/weight-shared transformer — the 8 blocks (d_model=512) are applied
# twice, so effective depth 16 with depth-8 params (~2x FLOPs/token, ~negligible extra
# params: just a 2x512 timestep embedding).  Aimed at the "124m" pathology: ~98M params
# but only ~23M of trainable transformer matrices, and a token budget sized off that
# small number — i.e. the net is FLOP-cheap and param-bottlenecked, exactly where adding
# recurrence (free compute per param) should pay.  Same data/budget as 124m-modern.
PRESET_124M_LOOP = dataclasses.replace(_124m_modern, n_recurrence=2)


PRESETS: dict[str, Config] = {
    "default": DEFAULT_CONFIG,
    "smoke": SMOKE,
    "d4": D4,
    "1p384b-depth24": PRESET_1P384B_DEPTH24,
    "124m": PRESET_124M,
    "124m-modern": PRESET_124M_MODERN,
    "124m-loop": PRESET_124M_LOOP,
}


__all__ = [
    "DEFAULT_CONFIG",
    "SMOKE",
    "PRESET_1P384B_DEPTH24",
    "PRESET_124M",
    "PRESET_124M_MODERN",
    "PRESET_124M_LOOP",
    "D4",
    "PRESETS",
    "FINEWEB_32K_DIR",
    "FINEWEB_TRAIN_GLOB",
    "FINEWEB_VAL_BIN",
    "FINEWEB_TOKENIZER_JSON",
    "FINEWEB_8K_DIR",
    "FINEWEB_8K_TRAIN_GLOB",
    "FINEWEB_8K_VAL_BIN",
    "FINEWEB_8K_TOKENIZER_JSON",
]
