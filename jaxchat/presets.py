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

# Default speedrun assets (legacy 8-GPU path).
FINEWEB_32K_DIR = os.path.join(PROJECT_ROOT, "data", "fineweb32k")
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

PRESET_1P384B_DEPTH24 = dataclasses.replace(DEFAULT_CONFIG, depth=24)

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


PRESETS: dict[str, Config] = {
    "default": DEFAULT_CONFIG,
    "smoke": SMOKE,
    "d4": D4,
    "1p384b-depth24": PRESET_1P384B_DEPTH24,
}


__all__ = [
    "DEFAULT_CONFIG",
    "SMOKE",
    "PRESET_1P384B_DEPTH24",
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
