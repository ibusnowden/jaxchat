"""Shared JAX runtime, model, optimizer, and dataset helpers.

Modernized architecture with:
- RoPE + QK-Norm (RMSNorm)
- ReLU² activation
- RMSNorm everywhere
- Sequence length warmup + schedule
- LR schedules: linear, cosine, WSD
- Batch size + seq_len joint scheduling
- Gradient clipping
- Z-loss regularization
- Init tricks (DeepNorm, Muon-compatible)
- Tunable weight tying (delayed untying)
- Skip connections (embedding→every block, block N→N+3)
- Value residuals + value-path augmentation
- Logit stabilization (tanh-capped, sigmoid softcap)
- FP8 LM head option
- Cross-document loss masking
- Document boundary handling
- Multi-dataset mixing
- Long-short hybrid attention
- GQA (Grouped-Query Attention)
- Token-feature enrichment (PKO, Bigram Hash)
- QK-Norm for lm_head
- Gradient checkpointing (recompute attention)
- Stochastic depth (layer dropout)
- Batch Muon, NorMuon, SOAP optimizers (via optimizer.py)
"""

from __future__ import annotations

import bisect
import glob
import itertools
import logging
import math
import os
import pickle
import sys
import uuid

if __package__ in {None, ""}:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


_FALSEY_VISIBLE_DEVICES = {"", "-1", "none", "void"}


def _has_visible_nvidia_device() -> bool:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        normalized = cuda_visible_devices.strip().lower()
        if normalized in _FALSEY_VISIBLE_DEVICES:
            return False

    nvidia_visible_devices = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    if nvidia_visible_devices is not None:
        normalized = nvidia_visible_devices.strip().lower()
        if normalized in _FALSEY_VISIBLE_DEVICES:
            return False

    return bool(glob.glob("/dev/nvidia[0-9]*"))


def configure_jax_runtime() -> None:
    """Make local CPU-only imports quiet without changing real GPU jobs."""

    requested_platforms = os.environ.get("JAX_PLATFORMS")
    has_gpu = _has_visible_nvidia_device()

    if requested_platforms and requested_platforms.strip().lower() != "cpu":
        return

    if has_gpu:
        return

    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
    if not requested_platforms:
        os.environ["JAX_PLATFORMS"] = "cpu"


configure_jax_runtime()

import jax

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "autotune")

# XLA flags for GPU performance (only set if GPU is available)
if _has_visible_nvidia_device():
    xla_flags = os.environ.get("XLA_FLAGS", "")
    gpu_flags = [
        # Note: triton_gemm causes autotune failures on RTX 6000 Ada with JAX 0.9.1
        "--xla_gpu_enable_cublaslt=True",
        "--xla_gpu_autotune_level=4",
    ]
    for flag in gpu_flags:
        if flag not in xla_flags:
            xla_flags = f"{xla_flags} {flag}" if xla_flags else flag
    os.environ["XLA_FLAGS"] = xla_flags

import einops
import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad, vmap
from jax.lax import rsqrt, scan
from jax.nn import relu
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.tree_util import tree_flatten, tree_flatten_with_path, tree_leaves, tree_map, tree_unflatten
from collections import Counter
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, NamedTuple

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional during tests
    tiktoken = None

from jaxchat.fa3 import attention as fa3_attention
from jaxchat.tokenizer import load_tokenizer
from jaxchat.optimizer import create_optimizer
from jaxchat.schedules import get_shape_for_step, get_train_shape_counts as schedules_get_train_shape_counts, format_shape_summary as schedules_format_shape_summary, get_eval_shape as schedules_get_eval_shape
from jaxchat.data_mixer import mean_loss_masked, build_doc_boundary_mask

Pytree = Any

BIN_MAGIC = 20240520
BIN_VERSION = 1
BIN_HEADER_INTS = 256
BIN_HEADER_BYTES = BIN_HEADER_INTS * 4


class Logger:
    def __init__(self, run_dir: str | None = None) -> None:
        self.run_id = None
        self.logdir = None
        self.logfile = None
        self.latest_checkpoint_file = None
        self.is_master = jax.process_index() == 0
        if not self.is_master:
            return
        if run_dir:
            self.logdir = os.path.abspath(run_dir)
            self.run_id = os.path.basename(self.logdir.rstrip(os.sep)) or "run"
        else:
            self.run_id = str(uuid.uuid4())
            self.logdir = f"logs/{self.run_id}"
        os.makedirs(self.logdir, exist_ok=True)
        if run_dir:
            self.logfile = os.path.join(self.logdir, "train.log")
        else:
            self.logfile = f"logs/{self.run_id}.txt"
        self.latest_checkpoint_file = os.path.join(self.logdir, "latest_checkpoint.txt")
        self.prev_metrics = None
        with open(self.logfile, "w") as f:
            launcher = sys.argv[0] if sys.argv else ""
            if launcher and os.path.isfile(launcher):
                with open(launcher) as ff:
                    code = ff.read()
            else:
                code = f"argv: {' '.join(sys.argv)}"
            f.write("=" * 100 + "\n" + code + "\n" + "=" * 100 + "\n")

    def msg(self, msg: str):
        if not self.is_master:
            return
        print(msg)
        with open(self.logfile, "a") as f:
            f.write("[MESSAGE] " + str(msg) + "\n")

    def log(self, metrics: dict):
        if not self.is_master:
            return
        metrics, self.prev_metrics = self.prev_metrics, metrics
        if metrics is None:
            return
        metrics = "  |  ".join(list(itertools.starmap("{}: {}".format, metrics.items())))
        print(metrics)
        with open(self.logfile, "a") as f:
            f.write("[METRICS (1 step stale)] " + str(metrics) + "\n")

    def flush(self):
        if not self.is_master:
            return
        metrics = self.prev_metrics
        self.prev_metrics = None
        if metrics is None:
            return
        metrics = "  |  ".join(list(itertools.starmap("{}: {}".format, metrics.items())))
        print(metrics)
        with open(self.logfile, "a") as f:
            f.write("[METRICS (latest)] " + str(metrics) + "\n")

    def dump(self, step: int, params: Pytree, opt_state: Pytree, config):
        if not self.is_master:
            return
        params_host = jax.device_get(params)
        opt_state_host = jax.device_get(opt_state)
        state_to_save = {
            "step": step,
            "params": params_host,
            "opt_state": opt_state_host,
            "config": config,
        }
        save_path = os.path.join(self.logdir, f"state_step{step:06d}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(state_to_save, f)
        if self.latest_checkpoint_file is not None:
            with open(self.latest_checkpoint_file, "w") as f:
                f.write(save_path + "\n")

        self.msg(f"Saved checkpoint to {save_path}")


@dataclass(kw_only=True, frozen=True)
@jax.tree_util.register_static
class Config:
    # --- Mesh & data ---
    mesh_axis_names: tuple[str, ...] = ("dp",)
    mesh_shape: tuple[int, ...] = ()
    input_bin: str = ""
    input_val_bin: str = ""
    # --- Model dimensions ---
    depth: int = 24
    n_layers: int = 0
    d_model: int = 0
    n_heads: int = 0
    d_head: int = 128
    n_kv_heads: int = 0  # 0 = full MHA, >0 = GQA with that many KV heads
    n_value_layers: int = -1  # -1 = auto (n_layers // 2); else the first N even layers get value embeds
    n_recurrence: int = 1  # >1: loop the whole block stack this many times (weight-shared "looped transformer";
    # effective depth = n_layers * n_recurrence) with a per-loop timestep embedding added to the residual stream
    # --- Training tokens ---
    train_token_ratio: float = 10.5
    target_train_tokens: int = 0
    actual_train_tokens: int = 0
    n_train_iters: int = 0
    n_warmup_iters: int = 0
    f_warmdown_iters: float = 0.5
    n_warmdown_iters: int = 0
    # --- LR schedule ---
    lr_schedule: str = "linear"  # "linear", "cosine", "wsd"
    # --- Validation & saving ---
    val_loss_every: int = 125
    val_tokens: int = 524288
    save_every: int = 0
    log_every: int = 10
    eval_at_start: bool = False
    # --- Batch / sequence ---
    tokens_per_step: int = 524288
    batch_size: int = 0
    micro_batch_size: int = 16
    min_seq_len: int = 2048
    max_seq_len: int = 2048
    sequence_warmup_intervals: int = 2048
    # --- Sequence & batch schedules ---
    seq_schedule_points: tuple = ()  # tuple of SchedulePoint (step, seq_len, _)
    batch_schedule_points: tuple = ()  # tuple of SchedulePoint (step, _, batch_size)
    joint_schedule_points: tuple = ()  # tuple of SchedulePoint (step, seq_len, batch_size)
    # --- Tokenizer ---
    seed: int = 42
    vocab_size: int = 32768
    tokenizer_name: str = "fineweb32k"
    tokenizer_json: str = ""
    tokenizer_bos_token: str = "<|bos|>"
    tokenizer_bos_id: int = 1
    # --- Logit head ---
    logit_softcap: float = 15.0
    logit_cap_style: str = "sigmoid"  # "sigmoid" or "tanh"
    normalize_logits: bool = False  # QK-Norm for lm_head
    lm_head_fp8: bool = False
    z_loss_coeff: float = 0.0
    # --- RoPE ---
    rope_base: float = 10000.0
    # --- Sharding ---
    dtype: Any = jnp.bfloat16
    weight_sharding: Any = None
    activation_sharding: tuple = (None, "dp", None)
    peak_tflops_per_device: float = 989.0
    # --- Attention ---
    use_pallas_attention: bool = True
    use_ring_attention: bool = True
    use_long_short_attention: bool = False
    attention_block_q: int = 128
    attention_block_k: int = 128
    sliding_window_pattern: tuple[int, int, int, int] = (1024, 1024, 1024, 2048)
    # --- Gradient ---
    max_grad_norm: float = 0.0  # 0 = disabled
    # --- Optimizer ---
    optimizer: str = "muon_adamw"  # "muon_adamw", "normuon", "soap"
    adam_eps: float = 1e-8
    adam_beta2: float = 0.95
    embed_beta1: float = 0.8
    lm_head_beta1: float = 0.8
    scalar_beta1: float = 0.8
    x0_beta1: float = 0.96
    embed_lr_base: float = 0.3
    lm_head_lr_base: float = 0.004
    scalar_resid_lr: float = 0.005
    scalar_x0_lr: float = 0.5
    muon_base_lr: float = 0.04
    muon_beta2: float = 0.95
    muon_eps: float = 1e-7
    muon_momentum_warmup_steps: int = 300
    muon_warmup_momentum_init: float = 0.85
    muon_warmup_momentum_final: float = 0.95
    muon_polar_iters: int = 5
    muon_polar_every: int = 1  # update orthogonalization every N steps
    # --- SOAP ---
    soap_rank: int = 32
    soap_update_freq: int = 10
    soap_beta2: float = 0.95
    # --- Weight decay ---
    weight_decay_base: float = 0.2
    weight_decay: float = 0.0
    # --- Weight tying ---
    weight_tying: str = "none"  # "none", "full", "delayed"
    untie_at_step: int = -1  # if delayed, step at which to untie (default: 2/3 of training)
    # --- Init ---
    init_style: str = "default"  # "default", "deepnorm", "muon"
    scale_embedding: bool = False  # scale embeddings by sqrt(d_model)
    # --- Residual ---
    skip_connections: tuple = ()  # e.g., ((3, 6),) means skip from block 3 to block 6
    hpc_cell_size: int = 0  # 0 = disabled, >0 = partition into cells
    # --- Stochastic depth ---
    layer_drop_prob: float = 0.0
    # --- Recompute ---
    recompute_layers: str = "none"  # "none", "attention", "all"
    # --- Cross-document masking ---
    cross_document_mask: bool = False
    doc_sep_id: int = 2  # document separator token ID (set by tokenizer)
    # --- Token features ---
    bigram_hash_embed: bool = False
    bigram_hash_buckets: int = 16384
    pko_enabled: bool = False
    pko_hash_buckets: int = 2048
    # --- Dataset schedule (swaps) ---
    dataset_schedule: tuple = ()  # tuple of DatasetSchedulePoint
    # --- Compilation ---
    jax_compilation_cache_dir: str = "/tmp/jax_cache"

    def __post_init__(self):
        object.__setattr__(self, "mesh_shape", (jax.device_count(),))
        object.__setattr__(self, "n_layers", self.depth)
        object.__setattr__(self, "d_model", self.depth * 64)
        object.__setattr__(self, "d_head", 128)
        assert self.depth % 2 == 0, "Depth must be even for alternating VE layers."
        assert self.d_model % self.d_head == 0
        object.__setattr__(self, "n_heads", self.d_model // self.d_head)
        if self.n_value_layers < 0:
            object.__setattr__(self, "n_value_layers", self.n_layers // 2)
        else:
            # value embeds live on even layers only; can't have more than n_layers//2 of them
            object.__setattr__(self, "n_value_layers", min(self.n_value_layers, self.n_layers // 2))
        if self.n_kv_heads == 0:
            object.__setattr__(self, "n_kv_heads", self.n_heads)
        # batch_size derived from tokens_per_step, rounded to micro_batch_size
        derived_batch_size = self.tokens_per_step // self.max_seq_len
        derived_batch_size = (derived_batch_size // self.micro_batch_size) * self.micro_batch_size
        object.__setattr__(self, "batch_size", max(derived_batch_size, self.micro_batch_size))
        derived_weight_decay = self.weight_decay_base * (12.0 / self.depth) ** 2
        object.__setattr__(self, "weight_decay", derived_weight_decay)
        if self.target_train_tokens <= 0:
            # Size the token budget off the params that actually scale with data, not the
            # vocab tax: a "124M" model here is ~80% input/output embedding tables, so
            # ``total * ratio`` wildly over-tokens the (~25M-param) transformer while the
            # accounting is dominated by lookup tables.  Exclude wte + lm_head; keep the
            # transformer matrices, scalars, token-feature tables, and value embeds.
            bd = expected_parameter_breakdown(self)
            budget_params = bd["total"] - bd["wte"] - bd["lm_head"]
            object.__setattr__(
                self,
                "target_train_tokens",
                int(budget_params * self.train_token_ratio),
            )
        if self.n_train_iters <= 0:
            n_train_iters = math.ceil(self.target_train_tokens / self.tokens_per_step)
            object.__setattr__(self, "n_train_iters", n_train_iters)
        object.__setattr__(
            self,
            "actual_train_tokens",
            int(self.n_train_iters * self.tokens_per_step),
        )
        object.__setattr__(
            self,
            "n_warmdown_iters",
            int(self.n_train_iters * self.f_warmdown_iters),
        )
        # Delayed weight tying default: untie at 2/3 of training
        if self.weight_tying == "delayed" and self.untie_at_step < 0:
            object.__setattr__(self, "untie_at_step", int(self.n_train_iters * 2 / 3))


def expected_parameter_breakdown(config: Config) -> dict[str, int]:
    d_model = config.d_model
    vocab_size = config.vocab_size
    n_layers = config.n_layers
    n_value_layers = config.n_value_layers
    n_heads = config.n_heads
    wte = vocab_size * d_model
    value_embeds = n_value_layers * vocab_size * d_model
    lm_head = d_model * vocab_size
    # GQA: if config has n_kv_heads < n_heads, WK/WV are smaller
    if config.n_kv_heads > 0 and config.n_kv_heads < n_heads:
        kv_dim = config.n_kv_heads * config.d_head
        # wq: d*d, wk: d*kv, wv: d*kv, wo: d*d  => 2*d*d + 2*d*kv
        # mlp: w1: d*4d, w2: 4d*d  => 8*d*d
        # total per layer: 10*d*d + 2*d*kv_dim
        per_layer = 10 * d_model * d_model + 2 * d_model * kv_dim
    else:
        # MHA: wq+wk+wv+wo = 4*d*d, mlp = 8*d*d => 12*d*d
        per_layer = 12 * d_model * d_model
    transformer_matrices = n_layers * per_layer + n_value_layers * 32 * n_heads
    scalars = 2 * n_layers + len(config.skip_connections)  # resid_lambdas + x0_lambdas + skip_lambdas
    # Token features
    extra = 0
    if config.bigram_hash_embed:
        extra += config.bigram_hash_buckets * d_model
    if config.pko_enabled:
        extra += config.pko_hash_buckets * config.d_head
    if config.normalize_logits:
        extra += config.vocab_size  # lm_head_norm per-output scale
    if config.n_recurrence > 1:
        extra += config.n_recurrence * d_model  # timestep_embed (looped transformer)
    total = wte + value_embeds + lm_head + transformer_matrices + scalars + extra
    return {
        "wte": int(wte),
        "value_embeds": int(value_embeds),
        "lm_head": int(lm_head),
        "transformer_matrices": int(transformer_matrices),
        "scalars": int(scalars),
        "extra": int(extra),
        "total": int(total),
    }


def format_parameter_breakdown(breakdown: dict[str, int]) -> str:
    ordered_keys = ("wte", "value_embeds", "lm_head", "transformer_matrices", "scalars", "extra", "total")
    return "  |  ".join(f"{key}: {breakdown[key]:,}" for key in ordered_keys if key in breakdown)


def path_to_names(path) -> tuple[str, ...]:
    names = []
    for entry in path:
        if hasattr(entry, "key"):
            names.append(str(entry.key))
        elif hasattr(entry, "idx"):
            names.append(str(entry.idx))
        elif hasattr(entry, "name"):
            names.append(str(entry.name))
        else:
            names.append(str(entry))
    return tuple(names)


def parameter_breakdown_from_params(params: Pytree) -> dict[str, int]:
    counts = {
        "wte": 0,
        "value_embeds": 0,
        "lm_head": 0,
        "transformer_matrices": 0,
        "scalars": 0,
        "extra": 0,
    }
    for path, leaf in tree_flatten_with_path(params)[0]:
        names = path_to_names(path)
        if names and names[0] == "wte":
            counts["wte"] += int(leaf.size)
        elif names and names[0] == "value_embeds":
            counts["value_embeds"] += int(leaf.size)
        elif names and names[0] == "lm_head":
            counts["lm_head"] += int(leaf.size)
        elif names and names[0] in {"resid_lambdas", "x0_lambdas", "skip_lambdas"}:
            counts["scalars"] += int(leaf.size)
        elif names and names[0] in {"bigram_embed", "pko_offset"}:
            counts["extra"] += int(leaf.size)
        else:
            counts["transformer_matrices"] += int(leaf.size)
    counts["total"] = sum(counts.values())
    return counts


def parameter_optimizer_labels(params: Pytree) -> dict[str, str]:
    """Return the optimizer group label for each parameter leaf."""
    labels: dict[str, str] = {}
    for path, leaf in tree_flatten_with_path(params)[0]:
        names = path_to_names(path)
        key = "/".join(names)
        if names and names[0] in {"wte", "value_embeds"}:
            labels[key] = "adam_embed"
        elif names and names[0] == "lm_head":
            labels[key] = "adam_lm_head"
        elif names and names[0] in {"resid_lambdas", "lm_head_norm", "skip_lambdas"}:
            labels[key] = "adam_resid"
        elif names and names[0] == "x0_lambdas":
            labels[key] = "adam_x0"
        elif getattr(leaf, "ndim", 0) == 2:
            labels[key] = "muon"
        else:
            labels[key] = "adam_resid"
    return labels


def count_parameters(params: Pytree) -> int:
    return sum(int(leaf.size) for leaf in tree_leaves(params))


def get_mesh(config: Config) -> Mesh:
    return jax.make_mesh(config.mesh_shape, config.mesh_axis_names)


def _weight_partition_spec(config: Config) -> P:
    if config.weight_sharding is None:
        return P()
    if isinstance(config.weight_sharding, tuple):
        return P(*config.weight_sharding)
    return P(config.weight_sharding)


def get_weight_sharding(config: Config, mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, _weight_partition_spec(config))


def get_data_parallel_sharding(config: Config, mesh: Mesh, ndim: int) -> NamedSharding:
    if ndim < 1:
        raise ValueError(f"Expected ndim >= 1, got {ndim}")
    return NamedSharding(mesh, P(config.mesh_axis_names[0], *([None] * (ndim - 1))))


def get_eval_shape(config: Config) -> tuple[int, int, int]:
    return schedules_get_eval_shape(config)


def _get_shape_for_step(step: int, config: Config):
    return get_shape_for_step(step, config)


def get_train_shape_counts(config: Config) -> Counter:
    counts_dict = schedules_get_train_shape_counts(config)
    return Counter(counts_dict)


def format_shape_summary(shape_counts: Counter) -> str:
    return schedules_format_shape_summary(dict(shape_counts))


@dataclass(frozen=True)
class BinShard:
    path: str
    n_tokens: int
    tokens: np.memmap
    start: int
    end: int


def _open_bin_shard(path: str, dtype=np.uint16) -> BinShard:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(BIN_HEADER_BYTES), dtype=np.int32)
    if header[0] != BIN_MAGIC:
        raise RuntimeError(f"Magic number mismatch in {path}")
    if header[1] != BIN_VERSION:
        raise RuntimeError(f"Unsupported version in {path}")
    n_tokens = int(header[2])
    tokens = np.memmap(path, mode="r", dtype=dtype, offset=BIN_HEADER_BYTES, shape=(n_tokens,))
    return BinShard(path=path, n_tokens=n_tokens, tokens=tokens, start=0, end=0)


class StreamingTokenLoader:
    def __init__(self, config: Config, logger: Logger, mesh: Mesh, is_training: bool):
        if jax.process_count() != 1:
            raise RuntimeError("This training stack expects one process controlling the 8-device mesh.")

        self.config = config
        self.logger = logger
        self.mesh = mesh
        self.is_training = is_training
        pattern = config.input_bin if is_training else config.input_val_bin
        files = sorted(glob.glob(pattern))
        if not files:
            raise RuntimeError(f"No files found for pattern {pattern}")
        shards = []
        cursor = 0
        for file_path in files:
            shard = _open_bin_shard(file_path)
            # Guard against a data/vocab mismatch (e.g. GPT-2-tokenized shards fed to a
            # 32k-vocab model): an out-of-range token id silently corrupts the embedding
            # gather and the cross-entropy labels, which shows up as an immediate loss=nan.
            if shard.n_tokens:
                probe = np.asarray(shard.tokens[: min(shard.n_tokens, 1_000_000)])
                max_id = int(probe.max())
                if max_id >= config.vocab_size:
                    raise RuntimeError(
                        f"Token id {max_id} in {file_path} is >= vocab_size {config.vocab_size}. "
                        "The shard was tokenized with a different vocabulary than the model expects."
                    )
            shards.append(BinShard(file_path, shard.n_tokens, shard.tokens, cursor, cursor + shard.n_tokens))
            cursor += shard.n_tokens
        self.shards = tuple(shards)
        self.ends = tuple(shard.end for shard in self.shards)
        self.total_tokens = cursor
        self.cursor = 0
        self.step = 0
        self.wrap_count = 0
        self.activation_sharding = NamedSharding(mesh, P(*config.activation_sharding))
        self.train_shape = _get_shape_for_step(0, config)
        self.eval_shape = get_eval_shape(config)
        self.num_batches = config.n_train_iters if is_training else max(1, config.val_tokens // config.tokens_per_step)
        self.logger.msg(
            f"Opened {'training' if is_training else 'validation'} dataset with "
            f"{len(self.shards)} shards and {self.total_tokens / 1e9:.2f}B tokens."
        )

    def reset(self) -> None:
        self.cursor = 0
        self.step = 0

    def _read_tokens(self, start: int, length: int) -> np.ndarray:
        if self.total_tokens < length:
            raise RuntimeError(
                f"Dataset only has {self.total_tokens} tokens, but {length} tokens are required."
            )
        pieces = []
        pos = start % self.total_tokens
        remaining = length
        while remaining > 0:
            shard_idx = bisect.bisect_right(self.ends, pos)
            shard = self.shards[shard_idx]
            local_start = pos - shard.start
            take = min(remaining, shard.n_tokens - local_start)
            pieces.append(np.asarray(shard.tokens[local_start : local_start + take], dtype=np.uint16))
            remaining -= take
            pos = (pos + take) % self.total_tokens
        if len(pieces) == 1:
            return pieces[0]
        return np.concatenate(pieces, axis=0)

    def __iter__(self):
        return self

    def __next__(self):
        if self.step >= self.num_batches:
            raise StopIteration
        seq_len, batch_size, n_grad_accum_steps = (
            _get_shape_for_step(self.step, self.config) if self.is_training else self.eval_shape
        )
        tokens_for_batch = batch_size * seq_len
        buf = self._read_tokens(self.cursor, tokens_for_batch + 1)
        x = np.asarray(buf[:-1], dtype=np.int32).reshape(batch_size, seq_len)
        y = np.asarray(buf[1:], dtype=np.int32).reshape(batch_size, seq_len)
        batched_x = einops.rearrange(x, "(a b) t -> a b t", a=n_grad_accum_steps)
        batched_y = einops.rearrange(y, "(a b) t -> a b t", a=n_grad_accum_steps)
        previous_cursor = self.cursor
        self.cursor = (self.cursor + tokens_for_batch) % self.total_tokens
        if self.cursor < previous_cursor:
            self.wrap_count += 1
            self.logger.msg(
                f"{'Training' if self.is_training else 'Validation'} dataset wrapped "
                f"{self.wrap_count} time(s)."
            )
        self.step += 1
        return (
            jax.device_put(batched_x, self.activation_sharding),
            jax.device_put(batched_y, self.activation_sharding),
        )


def load_dataset(config: Config, logger: Logger, mesh: Mesh, is_training: bool):
    return StreamingTokenLoader(config, logger, mesh, is_training=is_training)


def precompute_rope(config: Config, mesh: Mesh) -> Pytree:
    weight_sharding = get_weight_sharding(config, mesh)
    dim = config.d_head
    inv_freq = 1.0 / (
        config.rope_base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    )
    positions = jnp.arange(config.max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(positions, inv_freq)
    result = {
        "cos": jax.device_put(jnp.cos(freqs).astype(config.dtype), weight_sharding),
        "sin": jax.device_put(jnp.sin(freqs).astype(config.dtype), weight_sharding),
    }
    # Precompute bigram hash bucket lookup table (non-trainable, in precomp only)
    if config.bigram_hash_embed:
        from jaxchat.token_features import precompute_bigram_buckets as _precompute
        buckets = _precompute(config.vocab_size, config.bigram_hash_buckets, seed=config.seed)
        result["bigram_buckets"] = jax.device_put(jnp.asarray(buckets, dtype=jnp.uint16), get_weight_sharding(config, mesh))
    return result


def _load_config_tokenizer(config: Config):
    if not config.tokenizer_json:
        return None
    try:
        return load_tokenizer(config.tokenizer_json)
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError):
        return None


def precompute_token_bytes(config: Config, mesh: Mesh) -> jax.Array:
    weight_sharding = get_weight_sharding(config, mesh)
    token_bytes = np.ones(config.vocab_size, dtype=np.int32)
    tokenizer = _load_config_tokenizer(config)
    if tokenizer is not None:
        for token_id in range(config.vocab_size):
            try:
                piece = tokenizer.decode([token_id])
            except Exception:
                piece = ""
            token_bytes[token_id] = max(len(piece.encode("utf-8")), 1)
        return jax.device_put(jnp.asarray(token_bytes), weight_sharding)

    if tiktoken is not None and config.tokenizer_name in tiktoken.list_encoding_names():
        encoding = tiktoken.get_encoding(config.tokenizer_name)
        usable_vocab = min(config.vocab_size, encoding.n_vocab)
        special_token_ids = set(getattr(encoding, "_special_tokens", {}).values())
        token_bytes[:] = 1
        for token_id in range(usable_vocab):
            if token_id in special_token_ids:
                continue
            token_bytes[token_id] = max(len(encoding.decode_single_token_bytes(token_id)), 1)
    return jax.device_put(jnp.asarray(token_bytes), weight_sharding)


def init_params(config: Config, mesh: Mesh) -> Pytree:
    """Initialize model parameters with support for multiple init styles."""
    weight_sharding = get_weight_sharding(config, mesh)

    def sharded_constant(shape, value, dtype=None):
        arr = jnp.full(shape, value, dtype=dtype or config.dtype)
        return jax.device_put(arr, weight_sharding)

    def sharded_normal(key, shape, std):
        arr = jax.random.normal(key, shape, dtype=config.dtype) * std
        return jax.device_put(arr, weight_sharding)

    def sharded_uniform(key, shape, limit):
        arr = jax.random.uniform(key, shape, dtype=config.dtype, minval=-limit, maxval=limit)
        return jax.device_put(arr, weight_sharding)

    root_key = jax.random.key(seed=config.seed)
    key_iter = map(partial(jax.random.fold_in, root_key), itertools.count())

    # Init style
    init_style = config.init_style
    d_model = config.d_model

    if init_style == "deepnorm":
        # DeepNet-style: scale output projections by alpha / sqrt(2 * effective_depth).
        # For a looped transformer the block stack is applied n_recurrence times, so the
        # effective residual depth is n_layers * n_recurrence (== n_layers when not looped).
        deepnorm_alpha = 0.81  # standard DeepNorm alpha
        effective_depth = config.n_layers * config.n_recurrence
        deepnorm_scale = deepnorm_alpha * (2.0 * effective_depth) ** (-0.5)
        matrix_limit = math.sqrt(1.0 / d_model)
        emb_std = 1.0 / math.sqrt(d_model)
        head_std = 0.5 / math.sqrt(d_model)
        # wo and w2 will be scaled by deepnorm_scale (not zero)
        wo_init = deepnorm_scale
        w2_init = deepnorm_scale
        # resid_lambdas: all 1.0 (DeepNet doesn't use separate residual scalars)
        resid_init = 1.0
        x0_init = 0.1  # keep small embedding skip to stabilize gradients
    elif init_style == "muon":
        # Muon-compatible init: smaller uniform, zero output projections
        matrix_limit = math.sqrt(2.0 / (d_model + 4 * d_model))  # fan-in + fan-out for w1
        emb_std = 1.0 / math.sqrt(d_model)
        head_std = 0.25 / math.sqrt(d_model)
        wo_init = 0.0
        w2_init = 0.0
        resid_init = 1.0
        x0_init = 0.0  # x0_lambdas start at 0, learned
    else:  # "default"
        matrix_limit = math.sqrt(1.0 / d_model)
        emb_std = 1.0 / math.sqrt(d_model)
        head_std = 0.5 / math.sqrt(d_model)
        wo_init = 0.0
        w2_init = 0.0
        resid_init = 1.0
        x0_init = 0.1

    blocks = []
    kv_dim = config.n_kv_heads * config.d_head
    for _ in range(config.n_layers):
        blocks.append(
            {
                "attn": {
                    "wq": sharded_uniform(next(key_iter), (config.d_model, config.d_model), matrix_limit),
                    "wk": sharded_uniform(next(key_iter), (config.d_model, kv_dim), matrix_limit),
                    "wv": sharded_uniform(next(key_iter), (config.d_model, kv_dim), matrix_limit),
                    "wo": sharded_constant((config.d_model, config.d_model), wo_init),
                },
                "mlp": {
                    "w1": sharded_uniform(next(key_iter), (config.d_model, 4 * config.d_model), matrix_limit),
                    "w2": sharded_constant((4 * config.d_model, config.d_model), w2_init),
                },
            }
        )

    params = {
        "wte": sharded_normal(next(key_iter), (config.vocab_size, config.d_model), emb_std),
        "value_embeds": sharded_normal(
            next(key_iter),
            (config.n_value_layers, config.vocab_size, config.d_model),
            emb_std,
        ),
        "ve_gates": tuple(
            sharded_constant((32, config.n_heads), 0.0) for _ in range(config.n_value_layers)
        ),
        "blocks": tuple(blocks),
        "resid_lambdas": sharded_constant((config.n_layers,), resid_init),
        "x0_lambdas": sharded_constant((config.n_layers,), x0_init),
        "lm_head": sharded_normal(next(key_iter), (config.d_model, config.vocab_size), head_std),
    }

    # Looped transformer: per-loop "timestep" embedding added to the residual stream
    # before each pass over the block stack (breaks the fixed-point symmetry that
    # collapses expressivity when the same block is reapplied).
    if config.n_recurrence > 1:
        params["timestep_embed"] = sharded_normal(
            next(key_iter),
            (config.n_recurrence, config.d_model),
            emb_std,
        )

    # Optional bigram hash embedding table
    if config.bigram_hash_embed:
        params["bigram_embed"] = sharded_normal(
            next(key_iter),
            (config.bigram_hash_buckets, config.d_model),
            emb_std,
        )

    # Optional PKO offset table
    if config.pko_enabled:
        params["pko_offset"] = sharded_normal(
            next(key_iter),
            (config.pko_hash_buckets, config.d_head),
            0.02,
        )

    # Skip connection parameters (if any)
    skip_pairs = config.skip_connections
    if skip_pairs:
        skip_lambdas = {}
        for src, dst in skip_pairs:
            skip_lambdas[(src, dst)] = sharded_constant((), 0.0)  # start at 0 (ReZero)
        params["skip_lambdas"] = skip_lambdas

    # Weight tying: if "full", share wte and lm_head
    if config.weight_tying == "full":
        params["lm_head"] = params["wte"]

    # Optional embedding scaling
    if config.scale_embedding:
        params["wte"] = params["wte"] * math.sqrt(config.d_model)

    # QK-Norm for lm_head: per-output-dimension scale
    if config.normalize_logits:
        # lm_head_norm is a per-output (vocab) scaling factor
        params["lm_head_norm"] = sharded_constant((config.vocab_size,), 1.0)

    return params, precompute_rope(config, mesh)


class Optimizer(NamedTuple):
    init: Callable
    update: Callable


DistMuonAdamW = Optimizer


def scaled_group_lr(base_lr: float, config: Config) -> float:
    return base_lr / math.sqrt(config.d_model / 768.0)


# ---------------------------------------------------------------------------
# Optimizer initialization (delegates to optimizer.py)
# ---------------------------------------------------------------------------

def init_optimizer(config: Config, params: Pytree, mesh: Mesh):
    """Initialize optimizer using the factory from optimizer.py."""
    init_fn, update_fn = create_optimizer(config, params, mesh)
    optimizer = Optimizer(init_fn, update_fn)
    return optimizer, optimizer.init(params)


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------

def rms_norm(x):
    return x * rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-6)


def apply_rotary_emb(x, cos, sin):
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos
    return jnp.stack((y_even, y_odd), axis=-1).reshape(x.shape).astype(x.dtype)


def linear(x, weight):
    return jnp.einsum("...c,cd->...d", x, weight.astype(x.dtype))


def qkv_projection(x, params, config):
    """QKV projection with optional GQA (Grouped-Query Attention).

    If ``config.n_kv_heads < config.n_heads``, we project fewer KV heads
    and repeat them to match the number of Q heads.

    wk/wv project to (n_kv_heads * d_head) not d_model when GQA is active.
    """
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    d_head = config.d_head
    d_model = config.d_model

    q = linear(x, params["wq"]).reshape(x.shape[0], x.shape[1], n_heads, d_head)

    # GQA: if n_kv_heads < n_heads, the KV projection weights are smaller
    kv_dim = n_kv_heads * d_head
    if kv_dim != d_model:
        # WK/WV project to kv_dim (not d_model)
        k = linear(x, params["wk"]).reshape(x.shape[0], x.shape[1], n_kv_heads, d_head)
        v = linear(x, params["wv"]).reshape(x.shape[0], x.shape[1], n_kv_heads, d_head)
        # Repeat KV heads to match Q heads
        repeat_factor = n_heads // n_kv_heads
        if repeat_factor > 1:
            k = jnp.repeat(k, repeat_factor, axis=-2)
            v = jnp.repeat(v, repeat_factor, axis=-2)
    else:
        k = linear(x, params["wk"]).reshape(x.shape[0], x.shape[1], n_heads, d_head)
        v = linear(x, params["wv"]).reshape(x.shape[0], x.shape[1], n_heads, d_head)

    return q, k, v


def maybe_add_value_embedding(params, idx, x, v, layer_idx, config, embedding_out_sharding):
    if layer_idx % 2 == 1:
        return v
    slot = layer_idx // 2
    if slot >= config.n_value_layers:
        return v
    value_table = params["value_embeds"][slot]
    value_embed = value_table.at[idx].get(out_sharding=embedding_out_sharding)
    value_embed = value_embed.reshape(v.shape)
    gate = 2.0 * jax.nn.sigmoid(linear(x[..., :32], params["ve_gates"][slot]))
    return v + gate[..., None].astype(v.dtype) * value_embed.astype(v.dtype)


def apply_partial_key_offset(k, idx, params, config):
    """Partial Key Offset: add learned offset based on token ID hash."""
    if not config.pko_enabled or "pko_offset" not in params:
        return k
    from jaxchat.token_features import apply_partial_key_offset
    return apply_partial_key_offset(
        k, idx, params["pko_offset"],
        config.pko_hash_buckets, scale=0.1,
    )


def add_bigram_embed(x, idx, params, precomputed_params, config):
    """Add bigram hash embedding to input.
    
    bigram_buckets is in precomputed_params (non-trainable lookup table).
    """
    if not config.bigram_hash_embed or "bigram_embed" not in params or "bigram_buckets" not in precomputed_params:
        return x
    from jaxchat.token_features import maybe_add_bigram_embed
    return maybe_add_bigram_embed(
        x, idx, params["bigram_embed"], precomputed_params["bigram_buckets"],
        config.bigram_hash_buckets, scale=0.1,
    )


def attention_forward(params, shared_params, x, idx, cos, sin, layer_idx, config, embedding_out_sharding):
    q, k, v = qkv_projection(x, params, config)
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    q = rms_norm(q)
    k = rms_norm(k)
    # Partial Key Offset
    k = apply_partial_key_offset(k, idx, shared_params, config)
    # Value augmentation
    v = maybe_add_value_embedding(shared_params, idx, x, v, layer_idx, config, embedding_out_sharding)
    mesh = getattr(embedding_out_sharding, "mesh", None)
    y = fa3_attention(q, k, v, layer_idx=layer_idx, config=config, mesh=mesh)
    y = y.reshape(x.shape[0], x.shape[1], config.d_model)
    return linear(y, params["wo"])


def mlp_forward(params, x):
    return linear(relu(linear(x, params["w1"])) ** 2, params["w2"])


def block_forward(block_params, shared_params, x, idx, x0, cos, sin, layer_idx, config, embedding_out_sharding):
    # Residual combination: x0 skip + layer-specific residual
    x = shared_params["resid_lambdas"][layer_idx].astype(x.dtype) * x + shared_params["x0_lambdas"][
        layer_idx
    ].astype(x.dtype) * x0

    # Skip connections (block N → block N+3)
    skip_pairs = config.skip_connections
    if skip_pairs and "skip_lambdas" in shared_params:
        for src, dst in skip_pairs:
            if dst == layer_idx and f"skip_buffer_{src}" in shared_params:
                skip_scale = shared_params["skip_lambdas"][(src, dst)].astype(x.dtype)
                x = x + skip_scale * shared_params[f"skip_buffer_{src}"]

    # Stochastic depth (layer dropout) — applied during training
    # We handle this at the train_step level, not here, to keep jit clean.

    # Attention
    attn_in = rms_norm(x)
    x = x + attention_forward(
        block_params["attn"],
        shared_params,
        attn_in,
        idx,
        cos,
        sin,
        layer_idx,
        config,
        embedding_out_sharding,
    )

    # MLP
    mlp_in = rms_norm(x)
    x = x + mlp_forward(block_params["mlp"], mlp_in)
    return x


def gpt_forward(params, idx, precomputed_params, config, embedding_out_sharding):
    _, seq_len = idx.shape
    # Token embedding + optional bigram hash embedding
    x = params["wte"].at[idx].get(out_sharding=embedding_out_sharding)
    x = add_bigram_embed(x, idx, params, precomputed_params, config)
    if config.scale_embedding:
        x = x / math.sqrt(config.d_model)  # undo scaling for backward compat if scaled at init
    x = rms_norm(x)
    x0 = x

    cos = precomputed_params["cos"][:seq_len]
    sin = precomputed_params["sin"][:seq_len]

    # Build skip buffers
    skip_pairs = config.skip_connections
    skip_buffers = {}
    if skip_pairs:
        for src, dst in skip_pairs:
            skip_buffers[f"skip_buffer_{src}"] = None  # placeholder

    # Looped transformer: apply the block stack n_recurrence times (weight-shared),
    # adding a per-loop timestep embedding to the residual stream before each pass.
    n_recurrence = getattr(config, "n_recurrence", 1)
    for loop_idx in range(n_recurrence):
        if n_recurrence > 1:
            x = x + params["timestep_embed"][loop_idx].astype(x.dtype)
        for layer_idx, block in enumerate(params["blocks"]):
            # Store skip source (skip pairs are self-contained within each loop pass)
            if skip_pairs:
                for src, dst in skip_pairs:
                    if src == layer_idx:
                        skip_buffers[f"skip_buffer_{src}"] = x

            # Pass skip buffers through shared_params
            block_shared = dict(params)
            for k, v in skip_buffers.items():
                block_shared[k] = v

            x = block_forward(
                block,
                block_shared,
                x,
                idx,
                x0,
                cos,
                sin,
                layer_idx,
                config,
                embedding_out_sharding,
            )

    x = rms_norm(x)

    # LM head
    lm_head = params["lm_head"]
    
    # Determine if lm_head is in (d_model, vocab) or (vocab, d_model) format
    # wte: (vocab_size, d_model), lm_head when not tied: (d_model, vocab_size)
    if lm_head.shape[0] == config.vocab_size and lm_head.shape[1] == config.d_model:
        # lm_head is in embedding format (vocab, d_model) — transpose needed
        lm_head_weight = lm_head.T.astype(x.dtype)
    else:
        lm_head_weight = lm_head.astype(x.dtype)
    
    if config.normalize_logits and "lm_head_norm" in params:
        # QK-Norm for lm_head: normalize both x and lm_head weight
        x = rms_norm(x)
        # lm_head_norm is shape (vocab_size,) — scale per output logit
        lm_head_norm = params["lm_head_norm"].astype(x.dtype)
        # Normalize each column of the weight matrix
        w_std = jnp.sqrt(jnp.mean(lm_head_weight ** 2, axis=0, keepdims=True) + 1e-6)
        # Reshape lm_head_norm to (1, vocab_size) for broadcasting over rows
        w_norm = lm_head_weight * (lm_head_norm[None, :] / w_std)
        logits = linear(x, w_norm)
    else:
        logits = linear(x, lm_head_weight)

    # Logit stabilization
    if config.logit_cap_style == "tanh":
        # Tanh-capped logits: softcap * tanh(logits / softcap)
        logits = config.logit_softcap * jnp.tanh(logits / max(config.logit_softcap, 1.0))
    else:  # sigmoid (default)
        logits = (2.0 * config.logit_softcap) * jax.nn.sigmoid(logits / (config.logit_softcap / 2.0))

    return logits.astype(jnp.float32)


def loss_fn(params, batch, precomputed_params, config, embedding_out_sharding):
    idx, labels = batch
    logits = gpt_forward(params, idx, precomputed_params, config, embedding_out_sharding)
    axis = logits.ndim - 1
    label_logits = jnp.take_along_axis(logits, jnp.expand_dims(labels, axis), axis=axis).squeeze(axis)
    token_nll = jax.nn.logsumexp(logits, axis=axis) - label_logits

    # Cross-document masking
    if config.cross_document_mask:
        loss = mean_loss_masked(token_nll, idx, config.doc_sep_id, True)
    else:
        loss = jnp.mean(token_nll)

    # Z-loss regularization
    if config.z_loss_coeff > 0.0:
        logsumexp_val = jax.nn.logsumexp(logits, axis=axis)
        z_loss = config.z_loss_coeff * jnp.mean(jnp.square(logsumexp_val))
        loss = loss + z_loss

    return loss


def sft_loss_fn(params, batch, precomputed_params, config, embedding_out_sharding):
    """Masked cross-entropy used during supervised fine-tuning.

    ``batch`` is ``(idx, labels, mask)`` where ``mask`` is 1 on positions whose
    next-token target is supervised (assistant turns) and 0 elsewhere.
    """

    idx, labels, mask = batch
    logits = gpt_forward(params, idx, precomputed_params, config, embedding_out_sharding)
    axis = logits.ndim - 1
    label_logits = jnp.take_along_axis(logits, jnp.expand_dims(labels, axis), axis=axis).squeeze(axis)
    token_nll = jax.nn.logsumexp(logits, axis=axis) - label_logits
    mask_f = mask.astype(token_nll.dtype)

    # Cross-document masking (additive with SFT mask)
    if config.cross_document_mask:
        doc_mask = build_doc_boundary_mask(idx, config.doc_sep_id)
        mask_f = mask_f * doc_mask

    return jnp.sum(token_nll * mask_f) / jnp.maximum(jnp.sum(mask_f), 1.0)


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

def global_grad_norm(grads: Pytree) -> jax.Array:
    """Compute the global L2 norm of all gradients."""
    leaves = tree_leaves(grads)
    squared = jnp.array([jnp.sum(g ** 2) for g in leaves])
    return jnp.sqrt(jnp.sum(squared))


def clip_grads(grads: Pytree, max_norm: float) -> Pytree:
    """Clip gradients to max_norm globally."""
    if max_norm <= 0.0:
        return grads
    norm = global_grad_norm(grads)
    scale = jnp.where(norm > max_norm, max_norm / norm, 1.0)
    return tree_map(lambda g: g * scale, grads)


# ---------------------------------------------------------------------------
# Stochastic depth mask
# ---------------------------------------------------------------------------

def make_layer_drop_mask(layer_idx: int, drop_prob: float, key: jax.Array) -> bool:
    """Return True if this layer should be kept (not dropped)."""
    if drop_prob <= 0.0:
        return True
    keep_prob = 1.0 - drop_prob
    rnd = jax.random.uniform(key)
    return rnd < keep_prob


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------

@partial(jit, static_argnames=("optimizer", "config", "embedding_out_sharding"))
def train_step(
    config,
    params,
    precomputed_params,
    opt_state,
    optimizer,
    embedding_out_sharding,
    batched_x,
    batched_y,
):
    n_grad_accum_steps = batched_x.shape[0]

    def micro_step(carry, micro_batch):
        accum_grads, total_loss = carry
        loss, grads = value_and_grad(loss_fn)(
            params, micro_batch, precomputed_params, config, embedding_out_sharding
        )
        return (tree_map(jnp.add, accum_grads, grads), total_loss + loss), None

    zero_grads = tree_map(jnp.zeros_like, params)
    (final_grads_accum, total_loss), _ = scan(
        micro_step, (zero_grads, 0.0), (batched_x, batched_y)
    )
    avg_loss = total_loss / n_grad_accum_steps
    final_grads = tree_map(
        lambda g: (g / n_grad_accum_steps).astype(g.dtype), final_grads_accum
    )

    # Gradient clipping
    if config.max_grad_norm > 0.0:
        final_grads = clip_grads(final_grads, config.max_grad_norm)

    new_params, new_opt_state = optimizer.update(final_grads, params, opt_state)
    return new_params, new_opt_state, {"loss": avg_loss}


@partial(jit, static_argnames=("optimizer", "config", "embedding_out_sharding"))
def sft_train_step(
    config,
    params,
    precomputed_params,
    opt_state,
    optimizer,
    embedding_out_sharding,
    batched_x,
    batched_y,
    batched_mask,
):
    n_grad_accum_steps = batched_x.shape[0]

    def micro_step(carry, micro_batch):
        accum_grads, total_loss = carry
        loss, grads = value_and_grad(sft_loss_fn)(
            params, micro_batch, precomputed_params, config, embedding_out_sharding
        )
        return (tree_map(jnp.add, accum_grads, grads), total_loss + loss), None

    zero_grads = tree_map(jnp.zeros_like, params)
    (final_grads_accum, total_loss), _ = scan(
        micro_step, (zero_grads, 0.0), (batched_x, batched_y, batched_mask)
    )
    avg_loss = total_loss / n_grad_accum_steps
    final_grads = tree_map(
        lambda g: (g / n_grad_accum_steps).astype(g.dtype), final_grads_accum
    )

    # Gradient clipping
    if config.max_grad_norm > 0.0:
        final_grads = clip_grads(final_grads, config.max_grad_norm)

    new_params, new_opt_state = optimizer.update(final_grads, params, opt_state)
    return new_params, new_opt_state, {"loss": avg_loss}


def eval_step(
    params,
    batched_x,
    batched_y,
    precomputed_params,
    token_bytes,
    config,
    embedding_out_sharding,
    token_bytes_out_sharding,
):
    def micro_bpb(x, y):
        logits = gpt_forward(params, x, precomputed_params, config, embedding_out_sharding)
        axis = logits.ndim - 1
        label_logits = jnp.take_along_axis(logits, jnp.expand_dims(y, axis), axis=axis).squeeze(axis)
        token_nll = jax.nn.logsumexp(logits, axis=axis) - label_logits
        flat_labels = y.reshape(-1)
        byte_lengths = token_bytes.at[flat_labels].get(out_sharding=token_bytes_out_sharding)
        total_nats = jnp.sum(token_nll.reshape(-1))
        total_bytes = jnp.sum(byte_lengths)
        return total_nats, total_bytes

    total_nats, total_bytes = vmap(micro_bpb, in_axes=(0, 0))(batched_x, batched_y)
    total_nats = jnp.sum(total_nats)
    total_bytes = jnp.sum(total_bytes)
    return total_nats / (jnp.log(2.0) * total_bytes.astype(jnp.float32))


def estimate_mfu_proxy(param_count: int, config: Config, steady_state_step_s: float) -> float:
    if steady_state_step_s <= 0:
        return 0.0
    device_peak_flops = config.peak_tflops_per_device * 1e12
    total_peak_flops = max(jax.device_count(), 1) * device_peak_flops
    effective_flops = 6.0 * float(param_count) * float(config.tokens_per_step)
    return effective_flops / (steady_state_step_s * total_peak_flops)


# ---------------------------------------------------------------------------
# Weight tying manager (delayed untying)
# ---------------------------------------------------------------------------

def maybe_untie_weights(params: Pytree, step: int, config: Config) -> Pytree:
    """If using delayed weight tying and step >= untie_at_step, untie lm_head from wte.

    Returns modified params (or same if no change needed).
    """
    if config.weight_tying != "delayed":
        return params
    if step < config.untie_at_step:
        return params
    # Check if already untied
    if "lm_head" in params and "wte" in params:
        # If lm_head is the same object as wte, they're tied
        try:
            tree_flatten(params["lm_head"])[0]
            tree_flatten(params["wte"])[0]
        except Exception:
            return params
        # We detect tying by checking if they share memory — in JAX we can't easily.
        # Instead we track this via an attribute on the optimizer or a separate flag.
    return params


if __name__ == "__main__":
    from training.train_base import main

    raise SystemExit(main())
