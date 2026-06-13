"""Token-feature enrichment: Partial Key Offset (PKO) and Bigram Hash Embeddings.

These augment the standard token embeddings with additional signals that
improve convergence, especially in the early stages of training.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def token_bigram_hash(
    idx: jax.Array,
    bigram_bucket: jax.Array,
    n_buckets: int,
) -> jax.Array:
    """Compute bigram hash embedding indices.

    For each position i (except the first), hash(token_{i-1}, token_i) maps to
    a bucket.  The first token gets bucket 0 (no bigram info).

    CAUSALITY NOTE: position i must only see tokens <= i.  The pre-2026-06-12
    version paired (token_i, token_{i+1}) and padded at the END, which leaked
    the NEXT token into position i's input embedding — a label leak that
    inflated every teacher-forced metric (val_bpb, CORE) of bigram-enabled
    runs and made autoregressive generation collapse (at the generating
    position the "next token" is padding).  Checkpoints trained before the
    fix expect the leaky feature and are invalid either way.

    Args:
        idx: token IDs, shape (..., seq_len)
        bigram_bucket: precomputed (vocab_size, vocab_size) lookup table of
            bucket assignments (uint16)
        n_buckets: number of bigram hash buckets

    Returns:
        bucket indices, shape (..., seq_len), dtype int32
    """
    batch_shape = idx.shape[:-1]
    seq_len = idx.shape[-1]
    flat_idx = idx.reshape(-1, seq_len)

    # Compute pairs: [token_{i-1}, token_i]
    pairs = jnp.stack([flat_idx[:, :-1], flat_idx[:, 1:]], axis=-1)  # (B, S-1, 2)

    # Look up bucket from precomputed table
    buckets = bigram_bucket[pairs[..., 0], pairs[..., 1]]  # (B, S-1)

    # Pad with 0 for the FIRST position (no preceding token).
    padded = jnp.pad(buckets, ((0, 0), (1, 0)), mode="constant", constant_values=0)
    return padded.reshape(*batch_shape, seq_len).astype(jnp.int32)


def precompute_bigram_buckets(
    vocab_size: int,
    n_buckets: int,
    seed: int = 42,
) -> np.ndarray:
    """Precompute a random hash table mapping (token_i, token_{i+1}) → bucket.

    Returns a (vocab_size, vocab_size) uint16 array.
    """
    rng = np.random.default_rng(seed)
    buckets = rng.integers(0, n_buckets, size=(vocab_size, vocab_size), dtype=np.uint16)
    return buckets


def maybe_add_bigram_embed(
    x: jax.Array,
    idx: jax.Array,
    bigram_table: jax.Array,
    bigram_bucket: jax.Array,
    n_buckets: int,
    scale: float = 0.1,
) -> jax.Array:
    """Add bigram hash embedding to the input embedding.

    Args:
        x: input embedding, shape (batch, seq_len, d_model)
        idx: token IDs, shape (batch, seq_len)
        bigram_table: embedding table, shape (n_buckets, d_model)
        bigram_bucket: precomputed hash bucket lookup, (vocab_size, vocab_size)
        n_buckets: number of hash buckets
        scale: scaling factor for the bigram embedding

    Returns:
        x + bigram_embedding * scale
    """
    bucket_ids = token_bigram_hash(idx, bigram_bucket, n_buckets)
    bigram_emb = bigram_table[bucket_ids]  # (batch, seq_len, d_model)
    return x + scale * bigram_emb


def apply_partial_key_offset(
    k: jax.Array,
    idx: jax.Array,
    pko_table: jax.Array,
    hash_buckets: int,
    scale: float = 0.1,
) -> jax.Array:
    """Partial Key Offset: add a learned offset to keys based on token-ID hash.

    Args:
        k: key tensor, shape (batch, seq_len, n_heads, d_head)
        idx: token IDs, shape (batch, seq_len)
        pko_table: offset table, shape (hash_buckets, d_head) (broadcasted over heads)
        hash_buckets: number of hash buckets
        scale: scaling factor

    Returns:
        k + k_offset * scale
    """
    # Hash token IDs into buckets
    buckets = idx % hash_buckets  # (batch, seq_len)
    offset = pko_table[buckets]  # (batch, seq_len, d_head)
    offset = offset[:, :, None, :]  # broadcast over heads
    return k + scale * offset
