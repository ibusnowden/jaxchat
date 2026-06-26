"""Bisect which attention backend leaks future tokens.

diag_causality.py proved a non-causal leak (pad-content changes logits at the
last real position by ~14).  Backend selection order in jaxchat/fa3.py is
ring -> pallas_gpu_mha -> sdpa; the 0.5B config has use_ring_attention=True
and use_pallas_attention=True, T=1024 divisible by the 8-GPU mesh, so ring is
chosen.  This forces each backend in turn and reruns the leak probe + a short
greedy decode.  The backend whose probe goes clean (and whose greedy text says
something sane after "The capital of France is") is the causal-safe one.
"""

from __future__ import annotations

import dataclasses
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

VARIANTS = [
    ("default (ring)", {}),
    ("no-ring (pallas)", {"use_ring_attention": False}),
    ("sdpa only", {"use_ring_attention": False, "use_pallas_attention": False}),
]


def logits_at(engine: Engine, buf: np.ndarray, pos: int) -> np.ndarray:
    with engine.mesh:
        out = engine._jit_logits_at(
            engine.params, engine.precomputed_params, engine.config,
            engine.embedding_out_sharding, jnp.asarray(buf), pos,
        )
    return np.asarray(jax.device_get(out))


def top5(engine: Engine, logits: np.ndarray) -> str:
    idx = np.argsort(logits)[::-1][:5]
    return " | ".join(f"{engine.tokenizer.decode([int(i)])!r}:{logits[int(i)]:.2f}" for i in idx)


def main() -> int:
    engine = Engine.from_run_dir(BASE_RUN, stage="base", tokenizer_path=TOKENIZER)
    tok = engine.tokenizer
    T = engine.config.max_seq_len
    bos = int(tok.get_bos_token_id())
    ids = [bos] + list(tok.encode(PROMPT))
    n = len(ids)
    base_config = engine.config

    def make_buf(fill_id: int) -> np.ndarray:
        buf = np.full((1, T), fill_id, dtype=np.int32)
        buf[0, :n] = np.asarray(ids, dtype=np.int32)
        return buf

    for name, overrides in VARIANTS:
        engine.config = dataclasses.replace(base_config, **overrides) if overrides else base_config
        print(f"\n{'=' * 70}\nBACKEND VARIANT: {name}  {overrides}\n{'=' * 70}", flush=True)
        try:
            l_bos = logits_at(engine, make_buf(bos), n - 1)
            l_287 = logits_at(engine, make_buf(287), n - 1)
            d = float(np.max(np.abs(l_bos - l_287)))
            print(f"top5 (bos pad): {top5(engine, l_bos)}")
            print(f"top5 (287 pad): {top5(engine, l_287)}")
            print(f"max pad-content sensitivity = {d:.4f}  ->  {'LEAKS' if d > 0.01 else 'causal-clean'}")

            g = list(ids)
            for _ in range(15):
                buf = np.full((1, T), bos, dtype=np.int32)
                buf[0, :len(g)] = np.asarray(g, dtype=np.int32)
                g.append(int(np.argmax(logits_at(engine, buf, len(g) - 1))))
            print(f"greedy x15: {tok.decode(g[n:])!r}", flush=True)
        except Exception as e:  # surface per-variant failures, keep bisecting
            print(f"FAILED: {type(e).__name__}: {e}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
