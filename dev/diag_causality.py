"""Prove or refute a causality leak in the Engine's padded attention path.

Under causal attention, logits at position n-1 depend ONLY on ids[0..n-1]:
  T1: changing the PAD TOKEN VALUE after n-1 must not change logits[n-1].
  T2: replacing the pad region with real text must not change logits[n-1].
If either changes materially, future positions leak into the past — the
attention kernel is not causal in this configuration, which would explain
<|bos|>-flooded word-salad generation (queries drown in 1000+ <|bos|> pad keys)
while long-context teacher-forced scoring stays plausible.
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
FILLER = (
    "Water boils at one hundred degrees Celsius at sea level and freezes at "
    "zero. The quick brown fox jumps over the lazy dog near the river bank. "
) * 40


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

    def make_buf(fill_ids: list[int]) -> np.ndarray:
        buf = np.full((1, T), bos, dtype=np.int32)
        buf[0, :n] = np.asarray(ids, dtype=np.int32)
        m = min(T - n, len(fill_ids))
        buf[0, n:n + m] = np.asarray(fill_ids[:m], dtype=np.int32)
        return buf

    l_bos_pad = logits_at(engine, make_buf([]), n - 1)                      # pad = <|bos|>
    l_tok_pad = logits_at(engine, make_buf([287] * T), n - 1)               # pad = constant real token
    real_fill = list(tok.encode(FILLER))
    l_txt_pad = logits_at(engine, make_buf(real_fill), n - 1)               # pad = real text

    print(f"prompt n={n}, T={T}")
    print(f"\npad=<|bos|>      top5: {top5(engine, l_bos_pad)}")
    print(f"pad=tok287       top5: {top5(engine, l_tok_pad)}")
    print(f"pad=real text    top5: {top5(engine, l_txt_pad)}")
    d1 = float(np.max(np.abs(l_bos_pad - l_tok_pad)))
    d2 = float(np.max(np.abs(l_bos_pad - l_txt_pad)))
    print(f"\nmax|bos_pad - tok_pad| = {d1:.4f}")
    print(f"max|bos_pad - txt_pad| = {d2:.4f}")
    causal = d1 < 0.01 and d2 < 0.01
    print(f"\nVERDICT: {'CAUSAL (pads innocent — bug is elsewhere)' if causal else 'NON-CAUSAL LEAK CONFIRMED (future pad tokens change past logits)'}")

    # Bonus: with real-text fill, what does the model think follows the prompt
    # if we score the SAME position but give it a fully real context buffer?
    # (If the leak verdict is NON-CAUSAL, l_txt_pad's top5 being sane while
    # l_bos_pad's is junk directly shows the <|bos|> sea is what corrupts it.)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
