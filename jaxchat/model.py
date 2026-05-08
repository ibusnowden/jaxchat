"""Shared JAX runtime, model, optimizer, and dataset helpers."""

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
from jaxchat.tokenizer import load_hf_tokenizer

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
    mesh_axis_names: tuple[str, ...] = ("dp",)
    mesh_shape: tuple[int, ...] = ()
    input_bin: str = ""
    input_val_bin: str = ""
    depth: int = 24
    n_layers: int = 0
    d_model: int = 0
    n_heads: int = 0
    d_head: int = 128
    n_value_layers: int = 0
    train_token_ratio: float = 10.5
    target_train_tokens: int = 0
    actual_train_tokens: int = 0
    n_train_iters: int = 0
    n_warmup_iters: int = 0
    f_warmdown_iters: float = 0.5
    n_warmdown_iters: int = 0
    val_loss_every: int = 125
    val_tokens: int = 524288
    save_every: int = 0
    tokens_per_step: int = 524288
    batch_size: int = 0
    micro_batch_size: int = 16
    min_seq_len: int = 2048
    max_seq_len: int = 2048
    sequence_warmup_intervals: int = 2048
    seed: int = 42
    vocab_size: int = 32768
    tokenizer_name: str = "fineweb32k"
    tokenizer_json: str = ""
    tokenizer_bos_token: str = "<|bos|>"
    tokenizer_bos_id: int = 1
    logit_softcap: float = 15.0
    rope_base: float = 10000.0
    dtype: Any = jnp.bfloat16
    weight_sharding: Any = None
    activation_sharding: tuple = (None, "dp", None)
    peak_tflops_per_device: float = 989.0
    use_pallas_attention: bool = True
    use_ring_attention: bool = True
    attention_block_q: int = 128
    attention_block_k: int = 128
    sliding_window_pattern: tuple[int, int, int, int] = (1024, 1024, 1024, 2048)
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
    weight_decay_base: float = 0.2
    weight_decay: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "mesh_shape", (jax.device_count(),))
        object.__setattr__(self, "n_layers", self.depth)
        object.__setattr__(self, "d_model", self.depth * 64)
        object.__setattr__(self, "d_head", 128)
        assert self.depth % 2 == 0, "Depth must be even for alternating VE layers."
        assert self.d_model % self.d_head == 0
        object.__setattr__(self, "n_heads", self.d_model // self.d_head)
        object.__setattr__(self, "n_value_layers", self.n_layers // 2)
        assert self.tokens_per_step % self.max_seq_len == 0
        derived_batch_size = self.tokens_per_step // self.max_seq_len
        object.__setattr__(self, "batch_size", derived_batch_size)
        assert self.batch_size % self.micro_batch_size == 0
        derived_weight_decay = self.weight_decay_base * (12.0 / self.depth) ** 2
        object.__setattr__(self, "weight_decay", derived_weight_decay)
        if self.target_train_tokens <= 0:
            total_params = expected_parameter_breakdown(self)["total"]
            object.__setattr__(
                self,
                "target_train_tokens",
                int(total_params * self.train_token_ratio),
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


def expected_parameter_breakdown(config: Config) -> dict[str, int]:
    d_model = config.d_model
    vocab_size = config.vocab_size
    n_layers = config.n_layers
    n_value_layers = config.n_value_layers
    n_heads = config.n_heads
    wte = vocab_size * d_model
    value_embeds = n_value_layers * vocab_size * d_model
    lm_head = d_model * vocab_size
    transformer_matrices = (
        n_layers * (4 * d_model * d_model + 8 * d_model * d_model)
        + n_value_layers * 32 * n_heads
    )
    scalars = 2 * n_layers
    total = wte + value_embeds + lm_head + transformer_matrices + scalars
    return {
        "wte": int(wte),
        "value_embeds": int(value_embeds),
        "lm_head": int(lm_head),
        "transformer_matrices": int(transformer_matrices),
        "scalars": int(scalars),
        "total": int(total),
    }


def format_parameter_breakdown(breakdown: dict[str, int]) -> str:
    ordered_keys = ("wte", "value_embeds", "lm_head", "transformer_matrices", "scalars", "total")
    return "  |  ".join(f"{key}: {breakdown[key]:,}" for key in ordered_keys)


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
    }
    for path, leaf in tree_flatten_with_path(params)[0]:
        names = path_to_names(path)
        if names and names[0] == "wte":
            counts["wte"] += int(leaf.size)
        elif names and names[0] == "value_embeds":
            counts["value_embeds"] += int(leaf.size)
        elif names and names[0] == "lm_head":
            counts["lm_head"] += int(leaf.size)
        elif names and names[0] in {"resid_lambdas", "x0_lambdas"}:
            counts["scalars"] += int(leaf.size)
        else:
            counts["transformer_matrices"] += int(leaf.size)
    counts["total"] = sum(counts.values())
    return counts


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
    seq_len = config.max_seq_len
    batch_size = config.tokens_per_step // seq_len
    n_grad_accum = batch_size // config.micro_batch_size
    return int(seq_len), int(batch_size), int(n_grad_accum)


def _get_shape_for_step(step: int, config: Config):
    del step
    seq_len = config.max_seq_len
    batch_size = config.tokens_per_step // seq_len
    batch_size = max(batch_size, config.micro_batch_size)
    return int(seq_len), int(batch_size), int(batch_size // config.micro_batch_size)


def get_train_shape_counts(config: Config) -> Counter:
    return Counter(_get_shape_for_step(step, config) for step in range(config.n_train_iters))


def format_shape_summary(shape_counts: Counter) -> str:
    parts = []
    for (seq_len, batch_size, grad_accum), count in sorted(shape_counts.items()):
        parts.append(
            f"(seq_len={seq_len}, batch={batch_size}, grad_accum={grad_accum}) x {count}"
        )
    return "; ".join(parts)


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
    return {
        "cos": jax.device_put(jnp.cos(freqs).astype(config.dtype), weight_sharding),
        "sin": jax.device_put(jnp.sin(freqs).astype(config.dtype), weight_sharding),
    }


def _load_hf_tokenizer(config: Config):
    if not config.tokenizer_json:
        return None
    try:
        return load_hf_tokenizer(config.tokenizer_json)
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError):
        return None


def precompute_token_bytes(config: Config, mesh: Mesh) -> jax.Array:
    weight_sharding = get_weight_sharding(config, mesh)
    token_bytes = np.ones(config.vocab_size, dtype=np.int32)
    tokenizer = _load_hf_tokenizer(config)
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
    matrix_limit = math.sqrt(1.0 / config.d_model)
    emb_std = 1.0 / math.sqrt(config.d_model)
    head_std = 0.5 / math.sqrt(config.d_model)

    blocks = []
    for _ in range(config.n_layers):
        blocks.append(
            {
                "attn": {
                    "wq": sharded_uniform(next(key_iter), (config.d_model, config.d_model), matrix_limit),
                    "wk": sharded_uniform(next(key_iter), (config.d_model, config.d_model), matrix_limit),
                    "wv": sharded_uniform(next(key_iter), (config.d_model, config.d_model), matrix_limit),
                    "wo": sharded_constant((config.d_model, config.d_model), 0.0),
                },
                "mlp": {
                    "w1": sharded_uniform(next(key_iter), (config.d_model, 4 * config.d_model), matrix_limit),
                    "w2": sharded_constant((4 * config.d_model, config.d_model), 0.0),
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
        "resid_lambdas": sharded_constant((config.n_layers,), 1.0),
        "x0_lambdas": sharded_constant((config.n_layers,), 0.1),
        "lm_head": sharded_normal(next(key_iter), (config.d_model, config.vocab_size), head_std),
    }
    return params, precompute_rope(config, mesh)


class Optimizer(NamedTuple):
    init: Callable
    update: Callable


DistMuonAdamW = Optimizer


def get_lr_scale(step, n_warmup_iters, n_warmdown_iters, n_train_iters):
    step = step.astype(jnp.float32)
    if n_warmup_iters > 0:
        warmup_lr = (step + 1.0) / float(n_warmup_iters)
    else:
        warmup_lr = 1.0
    if n_warmdown_iters > 0:
        warmdown_start = n_train_iters - n_warmdown_iters
        warmdown_lr = jnp.maximum((n_train_iters - step) / float(n_warmdown_iters), 0.0)
    else:
        warmdown_start = n_train_iters
        warmdown_lr = 1.0
    return jnp.where(step < n_warmup_iters, warmup_lr, jnp.where(step < warmdown_start, 1.0, warmdown_lr))


def get_weight_decay_scale(step, n_train_iters):
    step = step.astype(jnp.float32)
    return jnp.maximum(1.0 - step / float(max(n_train_iters, 1)), 0.0)


def scaled_group_lr(base_lr: float, config: Config) -> float:
    return base_lr / math.sqrt(config.d_model / 768.0)


def get_muon_momentum(step, config: Config):
    frac = jnp.minimum(step / float(max(config.muon_momentum_warmup_steps, 1)), 1.0)
    return config.muon_warmup_momentum_init + frac * (
        config.muon_warmup_momentum_final - config.muon_warmup_momentum_init
    )


def polar_express_orthogonalize(g, steps: int, eps: float):
    transpose = g.shape[-2] > g.shape[-1]
    x = jnp.swapaxes(g, -1, -2) if transpose else g
    x = x.astype(jnp.float32)
    x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps)
    for _ in range(steps):
        gram = jnp.matmul(jnp.swapaxes(x, -1, -2), x)
        x = 1.5 * x - 0.5 * jnp.matmul(x, gram)
    x = jnp.swapaxes(x, -1, -2) if transpose else x
    return x.astype(g.dtype)


def _label_param(names: tuple[str, ...], leaf: jax.Array) -> str:
    if names and names[0] == "wte":
        return "adam_embed"
    if names and names[0] == "value_embeds":
        return "adam_embed"
    if names and names[0] == "lm_head":
        return "adam_lm_head"
    if names and names[0] == "resid_lambdas":
        return "adam_resid"
    if names and names[0] == "x0_lambdas":
        return "adam_x0"
    if leaf.ndim == 2:
        return "muon"
    raise ValueError(f"Unexpected parameter leaf at {names} with shape {leaf.shape}")


def parameter_optimizer_labels(params: Pytree) -> dict[str, str]:
    labels = {}
    for path, leaf in tree_flatten_with_path(params)[0]:
        names = path_to_names(path)
        labels["/".join(names)] = _label_param(names, leaf)
    return labels


def init_optimizer(config: Config, params: Pytree, mesh: Mesh):
    del mesh
    path_leaves, treedef = tree_flatten_with_path(params)
    flat_paths = tuple(path_to_names(path) for path, _ in path_leaves)
    flat_params = tuple(leaf for _, leaf in path_leaves)
    labels = tuple(_label_param(path, leaf) for path, leaf in zip(flat_paths, flat_params, strict=True))

    adam_groups = {
        "adam_embed": tuple(i for i, label in enumerate(labels) if label == "adam_embed"),
        "adam_lm_head": tuple(i for i, label in enumerate(labels) if label == "adam_lm_head"),
        "adam_resid": tuple(i for i, label in enumerate(labels) if label == "adam_resid"),
        "adam_x0": tuple(i for i, label in enumerate(labels) if label == "adam_x0"),
    }
    muon_groups = {}
    for idx, label in enumerate(labels):
        if label != "muon":
            continue
        muon_groups.setdefault(flat_params[idx].shape, []).append(idx)
    muon_group_specs = tuple((shape, tuple(indices)) for shape, indices in sorted(muon_groups.items()))

    adam_hparams = {
        "adam_embed": {
            "lr": scaled_group_lr(config.embed_lr_base, config),
            "beta1": config.embed_beta1,
            "beta2": config.adam_beta2,
            "weight_decay": config.weight_decay,
        },
        "adam_lm_head": {
            "lr": scaled_group_lr(config.lm_head_lr_base, config),
            "beta1": config.lm_head_beta1,
            "beta2": config.adam_beta2,
            "weight_decay": config.weight_decay,
        },
        "adam_resid": {
            "lr": config.scalar_resid_lr,
            "beta1": config.scalar_beta1,
            "beta2": config.adam_beta2,
            "weight_decay": 0.0,
        },
        "adam_x0": {
            "lr": config.scalar_x0_lr,
            "beta1": config.x0_beta1,
            "beta2": config.adam_beta2,
            "weight_decay": 0.0,
        },
    }

    def init(params):
        del params
        state = {"step": jnp.array(0, dtype=jnp.int32)}
        for name, indices in adam_groups.items():
            state[name] = {
                "m": tuple(jnp.zeros_like(flat_params[i], dtype=jnp.float32) for i in indices),
                "v": tuple(jnp.zeros_like(flat_params[i], dtype=jnp.float32) for i in indices),
            }
        muon_state = []
        for shape, indices in muon_group_specs:
            muon_state.append(
                {
                    "m": jnp.zeros((len(indices),) + shape, dtype=jnp.float32),
                    "row_v": jnp.zeros((len(indices), shape[0]), dtype=jnp.float32),
                    "col_v": jnp.zeros((len(indices), shape[1]), dtype=jnp.float32),
                }
            )
        state["muon"] = tuple(muon_state)
        return state

    def update(grads, params, state):
        grad_leaves = tree_flatten(grads)[0]
        param_leaves = tree_flatten(params)[0]
        new_leaves = list(param_leaves)
        new_state = {"step": state["step"] + 1}
        step = state["step"].astype(jnp.float32)
        lr_scale = get_lr_scale(state["step"], config.n_warmup_iters, config.n_warmdown_iters, config.n_train_iters)
        wd_scale = get_weight_decay_scale(state["step"], config.n_train_iters)

        for name, indices in adam_groups.items():
            hparams = adam_hparams[name]
            lr = jnp.asarray(hparams["lr"], dtype=jnp.float32) * lr_scale
            beta1 = jnp.asarray(hparams["beta1"], dtype=jnp.float32)
            beta2 = jnp.asarray(hparams["beta2"], dtype=jnp.float32)
            group_m = []
            group_v = []
            old_group_state = state[name]
            for local_idx, leaf_idx in enumerate(indices):
                grad = grad_leaves[leaf_idx].astype(jnp.float32)
                param = param_leaves[leaf_idx].astype(jnp.float32)
                m_prev = old_group_state["m"][local_idx]
                v_prev = old_group_state["v"][local_idx]
                m = beta1 * m_prev + (1.0 - beta1) * grad
                v = beta2 * v_prev + (1.0 - beta2) * jnp.square(grad)
                m_hat = m / (1.0 - beta1 ** (step + 1.0))
                v_hat = v / (1.0 - beta2 ** (step + 1.0))
                update_value = m_hat / (jnp.sqrt(v_hat) + config.adam_eps)
                if hparams["weight_decay"] > 0.0:
                    update_value = update_value + hparams["weight_decay"] * wd_scale * param
                new_param = param - lr * update_value
                new_leaves[leaf_idx] = new_param.astype(param_leaves[leaf_idx].dtype)
                group_m.append(m)
                group_v.append(v)
            new_state[name] = {"m": tuple(group_m), "v": tuple(group_v)}

        muon_state = []
        momentum = get_muon_momentum(step, config).astype(jnp.float32)
        muon_lr = jnp.asarray(config.muon_base_lr, dtype=jnp.float32) * lr_scale
        for group_idx, ((shape, indices), group_state) in enumerate(zip(muon_group_specs, state["muon"], strict=True)):
            del group_idx
            grad_stack = jnp.stack([grad_leaves[i].astype(jnp.float32) for i in indices], axis=0)
            param_stack = jnp.stack([param_leaves[i].astype(jnp.float32) for i in indices], axis=0)
            m_prev = group_state["m"]
            row_prev = group_state["row_v"]
            col_prev = group_state["col_v"]

            row_stat = jnp.mean(jnp.square(grad_stack), axis=-1)
            col_stat = jnp.mean(jnp.square(grad_stack), axis=-2)
            row_v = config.muon_beta2 * row_prev + (1.0 - config.muon_beta2) * row_stat
            col_v = config.muon_beta2 * col_prev + (1.0 - config.muon_beta2) * col_stat
            row_scale = row_v / (jnp.mean(row_v, axis=-1, keepdims=True) + config.muon_eps)
            factored_var = row_scale[..., :, None] * col_v[..., None, :]
            precond_grad = grad_stack / jnp.sqrt(factored_var + config.muon_eps)

            m = momentum * m_prev + (1.0 - momentum) * precond_grad
            nesterov = precond_grad + momentum * m
            ortho = polar_express_orthogonalize(nesterov, config.muon_polar_iters, config.muon_eps)
            scale = math.sqrt(max(1.0, shape[0] / shape[1]))
            decay_mask = (grad_stack * param_stack) >= 0
            decay = config.weight_decay * wd_scale * decay_mask.astype(jnp.float32) * param_stack
            new_param_stack = param_stack - muon_lr * (scale * ortho + decay)
            for offset, leaf_idx in enumerate(indices):
                new_leaves[leaf_idx] = new_param_stack[offset].astype(param_leaves[leaf_idx].dtype)
            muon_state.append({"m": m, "row_v": row_v, "col_v": col_v})
        new_state["muon"] = tuple(muon_state)
        return tree_unflatten(treedef, new_leaves), new_state

    optimizer = DistMuonAdamW(init, update)
    return optimizer, optimizer.init(params)


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
    q = linear(x, params["wq"]).reshape(x.shape[0], x.shape[1], config.n_heads, config.d_head)
    k = linear(x, params["wk"]).reshape(x.shape[0], x.shape[1], config.n_heads, config.d_head)
    v = linear(x, params["wv"]).reshape(x.shape[0], x.shape[1], config.n_heads, config.d_head)
    return q, k, v


def maybe_add_value_embedding(params, idx, x, v, layer_idx, config, embedding_out_sharding):
    if layer_idx % 2 == 1:
        return v
    slot = layer_idx // 2
    value_table = params["value_embeds"][slot]
    value_embed = value_table.at[idx].get(out_sharding=embedding_out_sharding)
    value_embed = value_embed.reshape(v.shape)
    gate = 2.0 * jax.nn.sigmoid(linear(x[..., :32], params["ve_gates"][slot]))
    return v + gate[..., None].astype(v.dtype) * value_embed.astype(v.dtype)


def attention_forward(params, shared_params, x, idx, cos, sin, layer_idx, config, embedding_out_sharding):
    q, k, v = qkv_projection(x, params, config)
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    q = rms_norm(q)
    k = rms_norm(k)
    v = maybe_add_value_embedding(shared_params, idx, x, v, layer_idx, config, embedding_out_sharding)
    mesh = getattr(embedding_out_sharding, "mesh", None)
    y = fa3_attention(q, k, v, layer_idx=layer_idx, config=config, mesh=mesh)
    y = y.reshape(x.shape[0], x.shape[1], config.d_model)
    return linear(y, params["wo"])


def mlp_forward(params, x):
    return linear(relu(linear(x, params["w1"])) ** 2, params["w2"])


def block_forward(block_params, shared_params, x, idx, x0, cos, sin, layer_idx, config, embedding_out_sharding):
    x = shared_params["resid_lambdas"][layer_idx].astype(x.dtype) * x + shared_params["x0_lambdas"][
        layer_idx
    ].astype(x.dtype) * x0
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
    mlp_in = rms_norm(x)
    x = x + mlp_forward(block_params["mlp"], mlp_in)
    return x


def gpt_forward(params, idx, precomputed_params, config, embedding_out_sharding):
    _, seq_len = idx.shape
    x = params["wte"].at[idx].get(out_sharding=embedding_out_sharding)
    x = rms_norm(x)
    x0 = x
    cos = precomputed_params["cos"][:seq_len]
    sin = precomputed_params["sin"][:seq_len]
    for layer_idx, block in enumerate(params["blocks"]):
        x = block_forward(
            block,
            params,
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
    logits = linear(x, params["lm_head"])
    logits = (2.0 * config.logit_softcap) * jax.nn.sigmoid(logits / (config.logit_softcap / 2.0))
    return logits.astype(jnp.float32)


def loss_fn(params, batch, precomputed_params, config, embedding_out_sharding):
    idx, labels = batch
    logits = gpt_forward(params, idx, precomputed_params, config, embedding_out_sharding)
    axis = logits.ndim - 1
    label_logits = jnp.take_along_axis(logits, jnp.expand_dims(labels, axis), axis=axis).squeeze(axis)
    return jnp.mean(jax.nn.logsumexp(logits, axis=axis) - label_logits)


def sft_loss_fn(params, batch, precomputed_params, config, embedding_out_sharding):
    """Masked cross-entropy used during supervised fine-tuning.

    ``batch`` is ``(idx, labels, mask)`` where ``mask`` is 1 on positions whose
    next-token target is supervised (assistant turns) and 0 elsewhere.  The
    denominator is ``mask.sum()`` so the loss does not double-count padding.
    """

    idx, labels, mask = batch
    logits = gpt_forward(params, idx, precomputed_params, config, embedding_out_sharding)
    axis = logits.ndim - 1
    label_logits = jnp.take_along_axis(logits, jnp.expand_dims(labels, axis), axis=axis).squeeze(axis)
    token_nll = jax.nn.logsumexp(logits, axis=axis) - label_logits
    mask_f = mask.astype(token_nll.dtype)
    return jnp.sum(token_nll * mask_f) / jnp.maximum(jnp.sum(mask_f), 1.0)


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
    """SFT train step with masked CE; same gradient-accumulation shape as ``train_step``."""

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


if __name__ == "__main__":
    from training.train_base import main

    raise SystemExit(main())
