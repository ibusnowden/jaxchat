"""Regression tests for the GRPO batch builder and reference-logprob pass.

These guard the two memory fixes that let depth-20 0.5B GRPO fit on the RTX
node: (1) ``_build_padded_batch`` crops to the batch's actual length (rounded to
a mesh-divisible bucket) instead of always padding to ``max_seq_len``, and
(2) ``rl_loss_fn`` consumes a precomputed frozen ``ref_logp_token`` so only one
``(B, T, vocab)`` logits tensor is ever live in the differentiated step.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import dataclasses  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

from jaxchat.model import (  # noqa: E402
    get_mesh,
    init_optimizer,
    init_params,
    precompute_rope,
)
from jaxchat.presets import PRESETS  # noqa: E402
from scripts.chat_rl import (  # noqa: E402
    _build_padded_batch,
    _token_logprobs,
    ref_logprobs,
    rl_loss_fn,
    rl_train_step,
)


def test_padded_batch_crops_below_max_seq_len():
    # Short sequences must NOT be padded out to max_seq_len (the OOM driver).
    prompts = [[1, 2, 3], [1, 2, 3, 4, 5]]
    gens = [[7, 8], [7, 8, 9]]
    idx, labels, mask, adv = _build_padded_batch(
        prompt_ids_list=prompts, gen_ids_list=gens, advantages=[0.5, -0.5],
        pad_id=0, max_seq_len=2048, pad_multiple=8,
    )
    # longest seq = 5 + 3 = 8 tokens -> 7 next-token targets -> rounded up to 8.
    assert idx.shape == (2, 8)
    # row 0: seq = [1,2,3,7,8]; idx = seq[:-1], labels = seq[1:].
    assert list(idx[0, :4]) == [1, 2, 3, 7]
    assert list(labels[0, :4]) == [2, 3, 7, 8]
    # only generated tokens are supervised: gen_start = len(prompt)-1 = 2.
    assert list(mask[0]) == [0, 0, 1, 1, 0, 0, 0, 0]
    assert list(adv) == [0.5, -0.5]


def test_padded_batch_respects_multiple_and_cap():
    # Every length is rounded up to ``pad_multiple`` (shard divisibility) ...
    for length in (1, 9, 17, 63):
        idx, *_ = _build_padded_batch(
            prompt_ids_list=[list(range(length))], gen_ids_list=[[1]],
            advantages=[0.0], pad_id=0, max_seq_len=4096, pad_multiple=8,
        )
        assert idx.shape[1] % 8 == 0
    # ... but never exceeds max_seq_len.
    idx, *_ = _build_padded_batch(
        prompt_ids_list=[list(range(50))], gen_ids_list=[list(range(100))],
        advantages=[0.0], pad_id=0, max_seq_len=16, pad_multiple=8,
    )
    assert idx.shape == (1, 16)


def _tiny_config():
    # Mirror the 0.5B preset's RL-relevant flags (no value embeds / bigram /
    # skips) but tiny, and replicate activations so the XLA sdpa attention path
    # works on a single CPU device.
    base = PRESETS["0p56b-rust65k"]
    return dataclasses.replace(
        base, depth=4, vocab_size=512, n_kv_heads=1, min_seq_len=64,
        max_seq_len=64, n_value_layers=0, bigram_hash_embed=False,
        skip_connections=(), activation_sharding=(None, None, None),
    )


def test_rl_loss_matches_closed_form_objective():
    config = _tiny_config()
    mesh = get_mesh(config)
    with mesh:
        params, _ = init_params(config, mesh)
        pre = precompute_rope(config, mesh)
        emb = NamedSharding(mesh, P(*config.activation_sharding))

        B, T = 4, 16
        rng = np.random.default_rng(0)
        idx = jnp.asarray(rng.integers(0, config.vocab_size, (B, T), np.int32))
        lbl = jnp.asarray(rng.integers(0, config.vocab_size, (B, T), np.int32))
        # Mask out a few positions so masking is actually exercised.
        mask_np = np.ones((B, T), np.int32)
        mask_np[:, :2] = 0
        mask = jnp.asarray(mask_np)
        adv = jnp.asarray(np.array([0.5, -0.5, 1.0, -1.0], np.float32))
        kl_beta, clip_eps = 0.05, 0.2

        ref_logp = ref_logprobs(config, params, pre, emb, idx, lbl)
        assert ref_logp.shape == (B, T)

        (loss, aux), grads = jax.value_and_grad(rl_loss_fn, has_aux=True)(
            params, ref_logp, (idx, lbl, mask, adv), pre, config, emb, kl_beta, clip_eps,
        )

        # Recompute the objective in numpy from the SAME policy log-probs the
        # loss uses (identical _token_logprobs path => no bf16 mismatch), so this
        # checks the pg/clip/KL/mask algebra exactly rather than forward rounding.
        logp = np.asarray(_token_logprobs(params, idx, lbl, pre, config, emb), np.float64)
        ref = np.asarray(ref_logp, np.float64)
        mf = mask_np.astype(np.float64)
        den = max(mf.sum(), 1.0)
        a = np.asarray(adv, np.float64)[:, None]
        r = np.exp(logp - ref)
        cr = np.clip(r, 1.0 - clip_eps, 1.0 + clip_eps)
        pg = -(np.minimum(a * r, a * cr) * mf).sum() / den
        kl = ((logp - ref) * mf).sum() / den
        expected = pg + kl_beta * kl

        assert abs(float(loss) - expected) < 1e-4, (float(loss), expected)
        # On-policy first step: ratio ~ 1 within bf16 noise of the two passes.
        assert abs(float(aux["ratio_mean"]) - 1.0) < 5e-2
        # Policy is trainable through the single-logits path.
        gnorm = float(jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(grads))))
        assert np.isfinite(gnorm) and gnorm > 0.0


def test_rl_train_step_runs_and_updates():
    config = _tiny_config()
    mesh = get_mesh(config)
    with mesh:
        params, _ = init_params(config, mesh)
        pre = precompute_rope(config, mesh)
        emb = NamedSharding(mesh, P(*config.activation_sharding))
        optimizer, opt_state = init_optimizer(config, params, mesh)

        B, T = 4, 16
        rng = np.random.default_rng(1)
        idx = jnp.asarray(rng.integers(0, config.vocab_size, (B, T), np.int32))
        lbl = jnp.asarray(rng.integers(0, config.vocab_size, (B, T), np.int32))
        mask = jnp.ones((B, T), jnp.int32)
        adv = jnp.asarray(np.array([0.5, -0.5, 1.0, -1.0], np.float32))

        ref_logp = ref_logprobs(config, params, pre, emb, idx, lbl)
        new_params, _, metrics = rl_train_step(
            config, params, ref_logp, pre, opt_state, optimizer, emb,
            idx, lbl, mask, adv, 0.01, 0.2,
        )
        assert np.isfinite(float(metrics["loss"]))
        assert set(metrics) == {"loss", "pg", "kl", "ratio_mean"}
