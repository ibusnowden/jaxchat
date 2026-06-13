"""Regression test for the bigram-hash label leak (fixed 2026-06-12).

The pre-fix token_bigram_hash gave position i the bucket of
hash(token_i, token_{i+1}), leaking the prediction target into the input.
These tests pin the causal contract: features at position i may depend only
on tokens at positions <= i.

Run: .venv/bin/python -m pytest tests/test_token_features_causality.py -q
"""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from jaxchat.token_features import precompute_bigram_buckets, token_bigram_hash  # noqa: E402

VOCAB = 257
N_BUCKETS = 64
SEQ = 33


def _buckets(idx: np.ndarray, table: np.ndarray) -> np.ndarray:
    import jax.numpy as jnp

    return np.asarray(token_bigram_hash(jnp.asarray(idx), jnp.asarray(table), N_BUCKETS))


def test_bigram_is_causal():
    """Changing token j must not change bucket assignments at positions < j."""
    rng = np.random.default_rng(0)
    table = precompute_bigram_buckets(VOCAB, N_BUCKETS, seed=1)
    idx = rng.integers(0, VOCAB, size=(2, SEQ), dtype=np.int32)
    base = _buckets(idx, table)
    for j in [1, SEQ // 2, SEQ - 1]:
        mutated = idx.copy()
        mutated[:, j] = (mutated[:, j] + 1) % VOCAB
        out = _buckets(mutated, table)
        assert np.array_equal(out[:, :j], base[:, :j]), (
            f"bucket at a position < {j} changed when token {j} changed — "
            "future tokens are leaking into past features"
        )


def test_bigram_uses_previous_and_current_token():
    """Position i must encode hash(token_{i-1}, token_i); position 0 gets 0."""
    rng = np.random.default_rng(2)
    table = precompute_bigram_buckets(VOCAB, N_BUCKETS, seed=1)
    idx = rng.integers(0, VOCAB, size=(1, SEQ), dtype=np.int32)
    out = _buckets(idx, table)
    assert out[0, 0] == 0, "first position has no preceding token; bucket must be 0"
    for i in range(1, SEQ):
        expected = int(table[idx[0, i - 1], idx[0, i]])
        assert int(out[0, i]) == expected, f"position {i} bucket != hash(prev, cur)"
