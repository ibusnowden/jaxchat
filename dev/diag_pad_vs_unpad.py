"""Diff Engine's full-pad (T=1024) next-token logits against a small-bucket pad.

Engine._next_token_logits pads to max_seq_len(1024) with pad_id(=bos=0) and
reads logits[0, n-1]; generation from it is word salad with <|bos|> injection,
while teacher-forced score_continuation through the SAME padded forward is
healthy.  Truly unpadded forwards fail to shard on the 8-way mesh (tiny T), so
this compares full-pad against pad-to-128 — if they disagree, right-padding
leaks into position n-1 (i.e. attention is not causal-safe at inference) and
the leak grows with pad length.
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
from jaxchat.model import gpt_forward  # noqa: E402

TOKENIZER = os.path.join(PROJECT_ROOT, "data/124m_rtx_run/tokenizer/tokenizer.json")
BASE_RUN = os.path.join(PROJECT_ROOT, "data/0p5b_e2e/runs/base_chinchilla")
PROMPT = "The capital of France is"
BUCKET = 128


def bucket_logits(engine: Engine, ids: list[int]) -> np.ndarray:
    # Same jitted path as Engine._next_token_logits, just a smaller pad buffer
    # (each new T compiles once; eager gpt_forward fails under explicit sharding).
    n = len(ids)
    T = ((n + BUCKET - 1) // BUCKET) * BUCKET
    buf = np.full((1, T), engine._pad_id, dtype=np.int32)
    buf[0, :n] = np.asarray(ids, dtype=np.int32)
    with engine.mesh:
        logits = engine._jit_logits_at(
            engine.params, engine.precomputed_params, engine.config,
            engine.embedding_out_sharding, jnp.asarray(buf), n - 1,
        )
    return np.asarray(jax.device_get(logits))


def top10(engine: Engine, logits: np.ndarray) -> str:
    idx = np.argsort(logits)[::-1][:10]
    return " | ".join(f"{engine.tokenizer.decode([int(i)])!r}:{logits[int(i)]:.2f}" for i in idx)


def main() -> int:
    engine = Engine.from_run_dir(BASE_RUN, stage="base", tokenizer_path=TOKENIZER)
    tok = engine.tokenizer
    bos = int(tok.get_bos_token_id())
    ids = [bos] + list(tok.encode(PROMPT))
    print(f"prompt ids (n={len(ids)}): {ids}", flush=True)

    lp = engine._next_token_logits(ids)          # full pad to 1024
    lb = bucket_logits(engine, ids)              # pad to 128
    print(f"\nfullpad(1024) top10: {top10(engine, lp)}")
    print(f"bucket(128)   top10: {top10(engine, lb)}")
    print(f"max |fullpad - bucket| = {np.max(np.abs(lp - lb)):.4f}")
    print(f"argmax fullpad={int(np.argmax(lp))} bucket={int(np.argmax(lb))}", flush=True)

    print("\n=== GREEDY DECODE x25, full-pad (Engine path) ===", flush=True)
    g = list(ids)
    for _ in range(25):
        g.append(int(np.argmax(engine._next_token_logits(g))))
    print(repr(tok.decode(g[len(ids):])), flush=True)

    print("\n=== GREEDY DECODE x25, bucket-128 pad ===", flush=True)
    g = list(ids)
    for _ in range(25):
        g.append(int(np.argmax(bucket_logits(engine, g))))
    print(repr(tok.decode(g[len(ids):])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
