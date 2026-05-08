"""Verify ``sft_loss_fn`` reduces to mean CE only over masked positions."""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jaxchat.model import gpt_forward, get_data_parallel_sharding, get_mesh, init_params, precompute_rope, sft_loss_fn  # noqa: E402
from jaxchat.presets import SMOKE  # noqa: E402


def test_sft_loss_matches_manual_masked_ce():
    config = SMOKE
    mesh = get_mesh(config)
    with mesh:
        params, _ = init_params(config, mesh)
        precomputed = precompute_rope(config, mesh)
    emb_sharding = get_data_parallel_sharding(config, mesh, ndim=3)

    rng = np.random.default_rng(0)
    B, T = 2, 64
    idx = rng.integers(0, config.vocab_size, size=(B, T)).astype(np.int32)
    labels = rng.integers(0, config.vocab_size, size=(B, T)).astype(np.int32)
    mask = np.zeros((B, T), dtype=np.int32)
    mask[:, 32:] = 1  # supervise only the second half.

    with mesh:
        logits = np.asarray(gpt_forward(params, jnp.asarray(idx), precomputed, config, emb_sharding))
        loss = float(
            sft_loss_fn(
                params,
                (jnp.asarray(idx), jnp.asarray(labels), jnp.asarray(mask)),
                precomputed,
                config,
                emb_sharding,
            )
        )

    # Manual masked CE in numpy.
    logz = np.log(np.sum(np.exp(logits - logits.max(axis=-1, keepdims=True)), axis=-1)) + logits.max(axis=-1)
    label_logits = np.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    nll = logz - label_logits
    manual = float((nll * mask).sum() / max(mask.sum(), 1))

    assert abs(loss - manual) < 1e-3, (loss, manual)


def test_sft_loss_zero_mask_is_zero():
    config = SMOKE
    mesh = get_mesh(config)
    with mesh:
        params, _ = init_params(config, mesh)
        precomputed = precompute_rope(config, mesh)
    emb_sharding = get_data_parallel_sharding(config, mesh, ndim=3)

    B, T = 2, 32
    idx = jnp.zeros((B, T), dtype=jnp.int32)
    labels = jnp.zeros((B, T), dtype=jnp.int32)
    mask = jnp.zeros((B, T), dtype=jnp.int32)

    with mesh:
        loss = float(sft_loss_fn(params, (idx, labels, mask), precomputed, config, emb_sharding))
    # Masked sum / max(0, 1) == 0.
    assert loss == 0.0
