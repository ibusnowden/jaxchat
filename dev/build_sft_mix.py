"""Mix a general-chat SFT pool with a task-specific (GSM8K) SFT pool.

The math-SFT warmup needs the policy to learn the ``\\boxed{}`` GSM8K format
*without* forgetting general chat ability. This builds one shuffled JSONL that
holds ``--n-smoltalk`` general rows plus the GSM8K rows up-sampled (repeated) to
``--n-gsm8k`` so math is a meaningful, tunable fraction of the mix.

Both inputs are ``{"messages": [...]}`` JSONL (smoltalk from
``dev/synth_smoltalk.py``; gsm8k from ``dev/synth_gsm8k.py --mode sft``). Output
order is deterministic for a given ``--seed``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys


def _load(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _take(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample exactly ``n`` rows, up-sampling with repeats when ``n`` exceeds the
    pool (so a small GSM8K set can still be weighted up) and sub-sampling without
    replacement otherwise."""
    if not rows or n <= 0:
        return []
    if n <= len(rows):
        return rng.sample(rows, n)
    out = list(rows)
    while len(out) < n:
        out.append(rng.choice(rows))
    return out[:n]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a mixed SFT pool (general + GSM8K).")
    parser.add_argument("--smoltalk", required=True, help="General-chat {messages} JSONL.")
    parser.add_argument("--gsm8k-sft", required=True, help="GSM8K {messages} JSONL (boxed CoT).")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-smoltalk", type=int, default=16000)
    parser.add_argument("--n-gsm8k", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    smol = _load(args.smoltalk)
    math = _load(args.gsm8k_sft)
    if not math:
        raise RuntimeError(f"No GSM8K-SFT rows in {args.gsm8k_sft}")

    pool = _take(smol, args.n_smoltalk, rng) + _take(math, args.n_gsm8k, rng)
    rng.shuffle(pool)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in pool:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_math = min(args.n_gsm8k, len(pool))
    frac = (n_math / len(pool)) if pool else 0.0
    print(f"[build_sft_mix] wrote {len(pool)} rows to {args.out} "
          f"(smoltalk~{min(args.n_smoltalk, len(smol) if args.n_smoltalk <= len(smol) else args.n_smoltalk)}, "
          f"gsm8k={args.n_gsm8k} → math frac {frac:.2f}; smoltalk pool={len(smol)}, gsm8k pool={len(math)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
