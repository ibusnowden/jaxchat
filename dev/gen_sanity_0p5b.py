"""Post-retrain generation sanity check for the 0.5B stack.

The bigram-hash label leak (fixed 2026-06-12) left every teacher-forced metric
(val_bpb, CORE) healthy while AUTOREGRESSIVE generation collapsed to word salad
with literal ``<|bos|>`` tokens injected mid-stream.  Teacher-forced evals
therefore CANNOT certify a working chatbot — only real sampling can.  This script
loads a checkpoint through the same ``Engine`` path that serves chat, samples a
few generations, prints them, and emits a PASS/FAIL verdict.

It exits non-zero ONLY on the unambiguous regression signature (special-token
literals injected into output, or degenerate single-token repetition), so a
strict heuristic can't false-fail a genuinely-fine run; milder incoherence is
flagged WARN and exits 0 for a human to eyeball.

Run inside a GPU allocation, e.g.::

    python -m dev.gen_sanity_0p5b --run-dir data/0p5b_e2e/runs/base_chinchilla \
        --stage base --tokenizer-json data/124m_rtx_run/tokenizer/tokenizer.json
"""

from __future__ import annotations

import argparse
import os
import re
import sys

PROJECT_ROOT = "/project/inniang/jaxchat"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jaxchat.model as model_lib  # noqa: E402

model_lib.configure_jax_runtime()

from jaxchat.engine import Engine  # noqa: E402

# Neutral completion probes for a base LM (coherent model continues sensibly).
BASE_PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a little",
    "Water is made of hydrogen and",
    "The most important thing about learning is",
    "In the morning, many people like to drink",
]
# Chat probes for an instruction-tuned model.
CHAT_PROMPTS = [
    "What is the capital of France?",
    "Write one sentence about the ocean.",
    "Hello! Who are you?",
    "List three colors.",
]

SPECIAL_LITERAL = re.compile(r"<\|[a-z_]+\|>")
WORD = re.compile(r"[A-Za-z]+")


def _coherence(text: str) -> dict:
    """Cheap, tokenizer-free heuristics that separate fluent text from salad."""
    specials = SPECIAL_LITERAL.findall(text)
    words = WORD.findall(text)
    n = len(words)
    uniq_ratio = (len(set(w.lower() for w in words)) / n) if n else 0.0
    wordlike = [w for w in words if len(w) >= 2 and re.search(r"[aeiouAEIOU]", w)]
    wordlike_ratio = (len(wordlike) / n) if n else 0.0
    # max share of any single repeated word (degenerate-loop detector)
    top_share = 0.0
    if n:
        from collections import Counter
        top_share = Counter(w.lower() for w in words).most_common(1)[0][1] / n
    return {
        "n_words": n,
        "n_special_literals": len(specials),
        "uniq_ratio": uniq_ratio,
        "wordlike_ratio": wordlike_ratio,
        "top_word_share": top_share,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--stage", required=True, choices=["base", "sft", "rl", "mid"])
    p.add_argument("--tokenizer-json",
                   default=os.path.join(PROJECT_ROOT, "data/124m_rtx_run/tokenizer/tokenizer.json"))
    p.add_argument("--chat", action="store_true",
                   help="Use the chat template (default for sft/rl). Base uses raw completion.")
    p.add_argument("--max-new-tokens", type=int, default=80)
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))

    use_chat = args.chat or args.stage in ("sft", "rl")
    print(f"\n{'='*70}\nGEN SANITY  stage={args.stage}  chat={use_chat}\n  run={args.run_dir}\n{'='*70}", flush=True)

    engine = Engine.from_run_dir(args.run_dir, stage=args.stage, tokenizer_path=args.tokenizer_json)
    print(f"loaded stage={engine.stage} step={engine.step}", flush=True)

    prompts = CHAT_PROMPTS if use_chat else BASE_PROMPTS
    stats = []
    injected_specials = 0
    for q in prompts:
        if use_chat:
            out = engine.chat([{"role": "user", "content": q}],
                              max_new_tokens=args.max_new_tokens, temperature=0.7,
                              top_k=50, top_p=0.95, seed=0)
        else:
            out = engine.generate(q, max_new_tokens=args.max_new_tokens,
                                  temperature=0.7, top_k=50, top_p=0.95, seed=0)
        c = _coherence(out)
        stats.append(c)
        injected_specials += c["n_special_literals"]
        print(f"\n[prompt] {q}\n[gen]    {out!r}\n         "
              f"words={c['n_words']} specials={c['n_special_literals']} "
              f"uniq={c['uniq_ratio']:.2f} wordlike={c['wordlike_ratio']:.2f} "
              f"top_word_share={c['top_word_share']:.2f}", flush=True)

    nonempty = [s for s in stats if s["n_words"] > 0]
    mean = lambda k: (sum(s[k] for s in nonempty) / len(nonempty)) if nonempty else 0.0
    m_uniq, m_wordlike, m_top = mean("uniq_ratio"), mean("wordlike_ratio"), mean("top_word_share")

    # Hard regression signature (exactly what the bigram leak produced).
    regressed = (injected_specials > 0) or (m_top > 0.5) or (len(nonempty) < len(prompts) // 2 + 1)
    # Softer fluency bar.
    fluent = (m_wordlike >= 0.6) and (m_uniq >= 0.5)

    print(f"\n{'-'*70}\nSUMMARY  mean_uniq={m_uniq:.2f}  mean_wordlike={m_wordlike:.2f}  "
          f"mean_top_word_share={m_top:.2f}  injected_specials={injected_specials}", flush=True)
    if regressed:
        print(f"VERDICT: ❌ FAIL — generation shows the broken-Engine signature "
              f"(specials_injected={injected_specials}, top_word_share={m_top:.2f}). "
              f"The bigram fix did NOT take, or the checkpoint is the old leaky one.", flush=True)
        return 1
    if not fluent:
        print(f"VERDICT: ⚠️  WARN — no hard regression, but fluency is low "
              f"(wordlike={m_wordlike:.2f}, uniq={m_uniq:.2f}). Eyeball the samples above.", flush=True)
        return 0
    print(f"VERDICT: ✅ PASS — coherent generation, no injected special tokens, no degenerate loop.", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
