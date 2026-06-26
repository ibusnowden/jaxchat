"""Prove the non-causal leak is the bigram hash embedding off-by-one, and
quantify how much the trained model relies on it.

token_features.token_bigram_hash assigns position i the bucket of
hash(token_i, token_{i+1}) (pad at END) — position i's input embeds the very
token it must predict.  Predictions:
  P1: changing ONLY buf[n] (the token right after the last real one) changes
      logits at n-1 a lot (the leaked "next token" changes).
  P2: changing ONLY buf[n+1:] (two+ ahead) changes logits at n-1 ~not at all
      (the rest of the network IS causal).
  P3: with the bigram table zeroed, teacher-forced bpb on real text gets much
      WORSE than the leaky 0.52 (the model leaned on the leak), and greedy
      decode becomes locally coherent (no false <|bos|>-next signal).
"""

from __future__ import annotations

import math
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
PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog. Paris is the capital of "
    "France and one of the most visited cities in the world. Water boils at "
    "one hundred degrees Celsius at sea level. In mathematics, two plus two "
    "equals four, and the square root of nine is three."
)


def top5(engine, logits):
    idx = np.argsort(logits)[::-1][:5]
    return " | ".join(f"{engine.tokenizer.decode([int(i)])!r}:{logits[int(i)]:.2f}" for i in idx)


def logits_at(engine, buf, pos):
    with engine.mesh:
        out = engine._jit_logits_at(
            engine.params, engine.precomputed_params, engine.config,
            engine.embedding_out_sharding, jnp.asarray(buf), pos,
        )
    return np.asarray(jax.device_get(out))


def paragraph_bpb(engine) -> float:
    tok = engine.tokenizer
    ids = tok.encode(PARAGRAPH)
    bos = int(tok.get_bos_token_id())
    nll = -engine.score_continuation([bos], ids)
    return nll / (math.log(2) * len(PARAGRAPH.encode("utf-8")))


def main() -> int:
    engine = Engine.from_run_dir(BASE_RUN, stage="base", tokenizer_path=TOKENIZER)
    tok = engine.tokenizer
    T = engine.config.max_seq_len
    bos = int(tok.get_bos_token_id())
    ids = [bos] + list(tok.encode(PROMPT))
    n = len(ids)

    base = np.full((1, T), bos, dtype=np.int32)
    base[0, :n] = np.asarray(ids, dtype=np.int32)

    only_next = base.copy(); only_next[0, n] = 287          # change ONLY position n
    only_far = base.copy(); only_far[0, n + 1:] = 287       # change ONLY n+1 onward

    l0 = logits_at(engine, base, n - 1)
    l1 = logits_at(engine, only_next, n - 1)
    l2 = logits_at(engine, only_far, n - 1)
    print(f"P1 change only buf[n]:    max|dlogits| = {np.max(np.abs(l0 - l1)):.4f}  (expect LARGE)")
    print(f"P2 change only buf[n+1:]: max|dlogits| = {np.max(np.abs(l0 - l2)):.4f}  (expect ~0)")
    print(f"   top5 base:      {top5(engine, l0)}")
    print(f"   top5 only-next: {top5(engine, l1)}", flush=True)

    print(f"\nbpb with leaky bigram as-is:  {paragraph_bpb(engine):.4f}")

    # Zero the bigram embedding table -> no next-token side channel.
    engine.params = dict(engine.params)
    engine.params["bigram_embed"] = jnp.zeros_like(engine.params["bigram_embed"])
    print(f"bpb with bigram table ZEROED: {paragraph_bpb(engine):.4f}  (leak reliance = the gap)", flush=True)

    print("\n=== GREEDY x25 with bigram zeroed ===")
    g = list(ids)
    for _ in range(25):
        b2 = np.full((1, T), bos, dtype=np.int32)
        b2[0, :len(g)] = np.asarray(g, dtype=np.int32)
        g.append(int(np.argmax(logits_at(engine, b2, len(g) - 1))))
    print(repr(tok.decode(g[n:])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
