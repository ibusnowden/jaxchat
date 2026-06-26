"""Test whether Engine's bug is the in-jit dynamic index logits[0, pos].

score_continuation (healthy) runs _jit_logprobs_full and slices on host.
_next_token_logits (broken) runs _jit_logits_at which indexes logits[0, pos]
INSIDE jit, under the explicit-sharding mesh.  If that dynamic gather reads
the wrong shard, "position n-1" actually returns a pad-region row — exactly
matching the observed behavior (predictions track the PAD CONTENT: ' of' pads
-> predicts ' of'; story-text pads -> predicts story continuations).

Probe: same padded buffer, three reads of position n-1:
  A. _jit_logits_at (in-jit dynamic index)          — suspected broken
  B. _jit_logprobs_full then host-side [0, n-1]     — suspected correct
  C. host-side read at OTHER positions of B to locate where A's row really is
Then greedy-decode 15 tokens via B (expect ' Paris...').
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = "/project/inniang/jaxchat"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jaxchat.model as model_lib  # noqa: E402

model_lib.configure_jax_runtime()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jaxchat.engine import Engine  # noqa: E402

TOKENIZER = os.path.join(PROJECT_ROOT, "data/124m_rtx_run/tokenizer/tokenizer.json")
BASE_RUN = os.path.join(PROJECT_ROOT, "data/0p5b_e2e/runs/base_chinchilla")
PROMPT = "The capital of France is"


def top5(engine, logits):
    idx = np.argsort(logits)[::-1][:5]
    return " | ".join(f"{engine.tokenizer.decode([int(i)])!r}:{logits[int(i)]:.2f}" for i in idx)


def main() -> int:
    engine = Engine.from_run_dir(BASE_RUN, stage="base", tokenizer_path=TOKENIZER)
    tok = engine.tokenizer
    T = engine.config.max_seq_len
    bos = int(tok.get_bos_token_id())
    ids = [bos] + list(tok.encode(PROMPT))
    n = len(ids)
    buf = np.full((1, T), bos, dtype=np.int32)
    buf[0, :n] = np.asarray(ids, dtype=np.int32)
    jbuf = jnp.asarray(buf)

    with engine.mesh:
        l_a = engine._jit_logits_at(
            engine.params, engine.precomputed_params, engine.config,
            engine.embedding_out_sharding, jbuf, n - 1,
        )
        full = engine._jit_logprobs_full(
            engine.params, engine.precomputed_params, engine.config,
            engine.embedding_out_sharding, jbuf,
        )
    l_a = np.asarray(jax.device_get(l_a))
    full = np.asarray(jax.device_get(full))[0]  # (T, vocab)
    l_b = full[n - 1]

    print(f"A in-jit  logits[0,{n-1}] top5: {top5(engine, l_a)}")
    print(f"B host    full[{n-1}]     top5: {top5(engine, l_b)}")
    print(f"max |A - B| = {np.max(np.abs(l_a - l_b)):.4f}")

    # Where does A's row actually live in the full logits?
    dists = np.max(np.abs(full - l_a[None, :]), axis=-1)
    j = int(np.argmin(dists))
    print(f"A's row best matches full[{j}] (max|diff|={dists[j]:.4f}); n-1={n-1}, T={T}")

    print("\n=== GREEDY x15 via full-logits host indexing (path B) ===")
    g = list(ids)
    for _ in range(15):
        b2 = np.full((1, T), bos, dtype=np.int32)
        b2[0, :len(g)] = np.asarray(g, dtype=np.int32)
        with engine.mesh:
            f2 = engine._jit_logprobs_full(
                engine.params, engine.precomputed_params, engine.config,
                engine.embedding_out_sharding, jnp.asarray(b2),
            )
        g.append(int(np.argmax(np.asarray(jax.device_get(f2))[0, len(g) - 1])))
    print(repr(tok.decode(g[n:])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
