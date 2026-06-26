"""Tokenizer-identity diagnostic for the 0.5B pipeline.

The chat smoke test (dev/smoke_chat_0p5b.py, job 115169) produced fluent-but-
nonsensical text from both sft_math2 and rl_math2 using the surviving
data/124m_rtx_run/tokenizer/tokenizer.json.  Two hypotheses:

  A. That tokenizer differs from the deleted fineweb32k_real_29/tokenizer.json
     the 0.5B was trained with -> decode permutation -> word salad everywhere.
  B. Tokenizer is fine; the chat stages / template are broken.

Discriminator: raw completions from the 0.5B BASE checkpoint (known-healthy:
val_bpb 0.4786 at train time).  Coherent -> hypothesis B.  Salad -> A.

Also prints teacher-forced bpb of a fixed English paragraph under base — a
mismatched tokenizer inflates this far above the ~0.5-0.9 a healthy model
gives (independent of sampling).
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

from jaxchat.engine import Engine  # noqa: E402

TOKENIZER = os.path.join(PROJECT_ROOT, "data/124m_rtx_run/tokenizer/tokenizer.json")
BASE_RUN = os.path.join(PROJECT_ROOT, "data/0p5b_e2e/runs/base_chinchilla")

PROMPTS = ["Once upon a time", "The capital of France is", "Today the weather"]

PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog. Paris is the capital of "
    "France and one of the most visited cities in the world. Water boils at "
    "one hundred degrees Celsius at sea level. In mathematics, two plus two "
    "equals four, and the square root of nine is three."
)


def main() -> int:
    engine = Engine.from_run_dir(BASE_RUN, stage="base", tokenizer_path=TOKENIZER)
    print(f"loaded stage={engine.stage} step={engine.step}", flush=True)

    print("\n=== RAW COMPLETIONS (base, max_new_tokens=120, temp=0.7) ===", flush=True)
    for p in PROMPTS:
        out = engine.generate(p, max_new_tokens=120, temperature=0.7, top_k=50, top_p=0.95, seed=0)
        print(f"\n[prompt] {p}\n[completion] {out}", flush=True)

    print("\n=== TEACHER-FORCED bpb OF FIXED PARAGRAPH ===", flush=True)
    tok = engine.tokenizer
    ids = tok.encode(PARAGRAPH)
    bos = int(tok.get_bos_token_id())
    nll = -engine.score_continuation([bos], ids)  # total nats over the continuation
    n_bytes = len(PARAGRAPH.encode("utf-8"))
    bpb = nll / (math.log(2) * n_bytes)
    print(f"tokens={len(ids)} bytes={n_bytes} total_nll={nll:.2f} bpb={bpb:.4f}")
    print("(healthy 0.5B base: ~0.5-1.0 bpb; tokenizer mismatch: >>2)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
