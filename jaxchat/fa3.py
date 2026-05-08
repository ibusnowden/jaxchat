"""Attention backend selection for the JAX training stack."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.lax import ppermute
from jax.nn import dot_product_attention
from jax.sharding import PartitionSpec as P

try:
    from jax.experimental.shard_map import shard_map
except ImportError:  # pragma: no cover - unavailable in some JAX builds
    shard_map = None

try:
    from jax.experimental.pallas.ops.gpu import attention as gpu_attention
except ImportError:  # pragma: no cover - CPU-only envs
    gpu_attention = None

try:
    from jax.experimental.pallas.ops.gpu import attention_mgpu as gpu_attention_mgpu
except ImportError:  # pragma: no cover - CPU-only envs
    gpu_attention_mgpu = None


@dataclass(frozen=True)
class AttentionRuntimeInfo:
    jax_version: str
    default_backend: str
    device_platforms: tuple[str, ...]
    gpu_runtime_available: bool
    shard_map_available: bool
    gpu_attention_module: bool
    gpu_attention_mgpu_module: bool
    gpu_attention_mgpu_training_safe: bool


@dataclass(frozen=True)
class AttentionBackendDecision:
    backend: str
    reason: str
    window_size: int


def layer_window_size(layer_idx: int, config) -> int:
    if layer_idx == config.n_layers - 1:
        return config.max_seq_len
    return int(config.sliding_window_pattern[layer_idx % len(config.sliding_window_pattern)])


def runtime_info() -> AttentionRuntimeInfo:
    devices = tuple(jax.devices())
    platforms = tuple(device.platform for device in devices)
    gpu_runtime = any(platform == "gpu" for platform in platforms)
    return AttentionRuntimeInfo(
        jax_version=jax.__version__,
        default_backend=jax.default_backend(),
        device_platforms=platforms,
        gpu_runtime_available=gpu_runtime,
        shard_map_available=shard_map is not None,
        gpu_attention_module=gpu_attention is not None,
        gpu_attention_mgpu_module=gpu_attention_mgpu is not None,
        gpu_attention_mgpu_training_safe=False,
    )


def format_runtime_info() -> str:
    info = runtime_info()
    return (
        f"jax={info.jax_version} backend={info.default_backend} "
        f"devices={info.device_platforms or ('none',)} "
        f"gpu_runtime={info.gpu_runtime_available} "
        f"gpu_attention={info.gpu_attention_module} "
        f"gpu_attention_mgpu={info.gpu_attention_mgpu_module} "
        f"mgpu_causal_training_safe={info.gpu_attention_mgpu_training_safe} "
        f"shard_map={info.shard_map_available}"
    )


def _is_gpu_runtime() -> bool:
    try:
        return any(device.platform == "gpu" for device in jax.devices())
    except RuntimeError:  # pragma: no cover - backend initialization failures
        return False


def _can_use_pallas_mha(q, k, v, window_size: int, config) -> bool:
    if gpu_attention is None or not config.use_pallas_attention or not _is_gpu_runtime():
        return False
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        return False
    if q.shape[1] != k.shape[1]:
        return False
    if window_size != q.shape[1]:
        return False
    if window_size < q.shape[1]:
        return False
    if q.shape[1] % config.attention_block_q != 0:
        return False
    if k.shape[1] % config.attention_block_k != 0:
        return False
    if q.shape[-1] % 64 != 0:
        return False
    return True


def _sdpa_attention(q, k, v, *, scale: float, window_size: int):
    local_window = None
    if window_size < k.shape[1]:
        local_window = (window_size - 1, 0)
    return dot_product_attention(
        q,
        k,
        v,
        scale=scale,
        is_causal=True,
        local_window_size=local_window,
    )


def _pallas_attention(q, k, v, *, scale: float, config):
    block_sizes = gpu_attention.BlockSizes(
        block_q=config.attention_block_q,
        block_k=config.attention_block_k,
        block_q_dkv=32,
        block_kv_dkv=32,
        block_q_dq=32,
        block_kv_dq=32,
    )
    return gpu_attention.mha(
        q,
        k,
        v,
        segment_ids=None,
        sm_scale=scale,
        causal=True,
        block_sizes=block_sizes,
    )


def backend_decision(q, k, v, *, layer_idx: int, config, mesh=None) -> AttentionBackendDecision:
    window_size = layer_window_size(layer_idx, config)
    if _can_use_ring(q, k, v, window_size, mesh, config):
        return AttentionBackendDecision(
            backend="ring",
            reason="multi-device GPU mesh is available and the sequence is evenly shardable",
            window_size=window_size,
        )
    if _can_use_pallas_mha(q, k, v, window_size, config):
        return AttentionBackendDecision(
            backend="pallas_gpu_mha",
            reason="installed gpu.attention kernel supports this full-context causal shape",
            window_size=window_size,
        )
    if window_size < q.shape[1]:
        return AttentionBackendDecision(
            backend="sdpa",
            reason="installed Pallas kernels in JAX 0.9.1 do not provide training-safe sliding-window causal attention here",
            window_size=window_size,
        )
    if not _is_gpu_runtime():
        return AttentionBackendDecision(
            backend="sdpa",
            reason="no GPU JAX runtime is visible on this host",
            window_size=window_size,
        )
    return AttentionBackendDecision(
        backend="sdpa",
        reason="shape or runtime constraints rejected the installed Pallas GPU kernels",
        window_size=window_size,
    )


def backend_summary_for_config(config, mesh=None) -> str:
    info = runtime_info()
    sample_seq_len = config.max_seq_len
    sample_shape = (1, sample_seq_len, config.n_heads, config.d_head)
    q = jnp.zeros(sample_shape, dtype=config.dtype)
    k = jnp.zeros(sample_shape, dtype=config.dtype)
    v = jnp.zeros(sample_shape, dtype=config.dtype)
    first_layer = backend_decision(q, k, v, layer_idx=0, config=config, mesh=mesh)
    final_layer = backend_decision(q, k, v, layer_idx=config.n_layers - 1, config=config, mesh=mesh)
    return (
        f"{format_runtime_info()} | "
        f"layer0={first_layer.backend} ({first_layer.reason}) | "
        f"last_layer={final_layer.backend} ({final_layer.reason})"
    )


def _can_use_ring(q, k, v, window_size: int, mesh, config) -> bool:
    if shard_map is None or mesh is None or not config.use_ring_attention or not _is_gpu_runtime():
        return False
    axis_name = mesh.axis_names[0]
    num_devices = mesh.shape[axis_name]
    if num_devices <= 1:
        return False
    if q.shape[1] % num_devices != 0 or k.shape[1] % num_devices != 0:
        return False
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        return False
    return True


def distributed_ring_attention(q, k, v, *, scale: float, window_size: int, mesh):
    axis_name = mesh.axis_names[0]
    num_devices = mesh.shape[axis_name]
    q_seq_len = q.shape[1]
    kv_seq_len = k.shape[1]
    local_q = q_seq_len // num_devices
    local_kv = kv_seq_len // num_devices
    in_spec = P(None, axis_name, None, None)

    @partial(shard_map, mesh=mesh, in_specs=(in_spec, in_spec, in_spec), out_specs=in_spec)
    def ring_attention_spmd(local_q_block, local_k_block, local_v_block):
        device_idx = jax.lax.axis_index(axis_name)
        q_positions = device_idx * local_q + jnp.arange(local_q, dtype=jnp.int32)
        running_output = jnp.zeros_like(local_q_block, dtype=jnp.float32)
        running_m = jnp.full(local_q_block.shape[:-1], -jnp.inf, dtype=jnp.float32)
        running_l = jnp.zeros(local_q_block.shape[:-1], dtype=jnp.float32)
        current_k = local_k_block
        current_v = local_v_block

        for step in range(num_devices):
            source_idx = (device_idx - step) % num_devices
            k_positions = source_idx * local_kv + jnp.arange(local_kv, dtype=jnp.int32)
            scores = jnp.einsum("bthd,bshd->bths", local_q_block, current_k).astype(jnp.float32)
            scores = scores * scale
            causal_mask = q_positions[:, None] >= k_positions[None, :]
            if window_size < kv_seq_len:
                window_mask = (q_positions[:, None] - k_positions[None, :]) < window_size
                mask = causal_mask & window_mask
            else:
                mask = causal_mask
            masked_scores = jnp.where(mask[None, :, None, :], scores, -jnp.inf)
            curr_m = jnp.max(masked_scores, axis=-1)
            safe_shifted = jnp.where(
                jnp.isfinite(curr_m[..., None]),
                masked_scores - curr_m[..., None],
                -jnp.inf,
            )
            probs = jnp.where(mask[None, :, None, :], jnp.exp(safe_shifted), 0.0)
            curr_l = jnp.sum(probs, axis=-1)
            curr_o = jnp.einsum("bths,bshd->bthd", probs.astype(jnp.float32), current_v.astype(jnp.float32))

            next_m = jnp.maximum(running_m, curr_m)
            prev_scale = jnp.where(
                jnp.isfinite(running_m),
                jnp.exp(running_m - next_m),
                0.0,
            )
            curr_scale = jnp.where(
                jnp.isfinite(curr_m),
                jnp.exp(curr_m - next_m),
                0.0,
            )
            running_l = prev_scale * running_l + curr_scale * curr_l
            running_output = prev_scale[..., None] * running_output + curr_scale[..., None] * curr_o
            running_m = next_m

            if step + 1 < num_devices:
                perm = [(i, (i + 1) % num_devices) for i in range(num_devices)]
                current_k = ppermute(current_k, axis_name=axis_name, perm=perm)
                current_v = ppermute(current_v, axis_name=axis_name, perm=perm)

        return (running_output / jnp.maximum(running_l[..., None], 1e-9)).astype(local_q_block.dtype)

    return ring_attention_spmd(q, k, v)


def attention(q, k, v, *, layer_idx: int, config, mesh=None):
    decision = backend_decision(q, k, v, layer_idx=layer_idx, config=config, mesh=mesh)
    window_size = decision.window_size
    scale = 1.0 / math.sqrt(q.shape[-1])
    if decision.backend == "ring":
        return distributed_ring_attention(q, k, v, scale=scale, window_size=window_size, mesh=mesh)
    if decision.backend == "pallas_gpu_mha":
        return _pallas_attention(q, k, v, scale=scale, config=config)
    return _sdpa_attention(q, k, v, scale=scale, window_size=window_size)
