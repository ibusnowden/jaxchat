"""Snapshot a small GSM8K split to JSONL for local RL/eval.

If the dataset is unreachable, a tiny synthetic fallback is written so the
pipeline can still demonstrate end-to-end wiring.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


SYNTHETIC = [
    {"question": "What is 2 + 3?", "answer": "Adding gives 5.\n#### 5"},
    {"question": "What is 6 * 7?", "answer": "Multiplying 6 and 7 gives 42.\n#### 42"},
    {"question": "If a pen costs 4 dollars and you buy 3 pens, how much do you spend?", "answer": "3 * 4 = 12.\n#### 12"},
    {"question": "What is half of 18?", "answer": "18 / 2 = 9.\n#### 9"},
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize GSM8K splits as JSONL.")
    parser.add_argument("--out", required=True, help="Output JSONL.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    written = 0
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split=args.split, streaming=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            for row in ds:
                handle.write(json.dumps({"question": row["question"], "answer": row["answer"]}, ensure_ascii=False) + "\n")
                written += 1
                if written >= args.n:
                    break
    except Exception as exc:  # pragma: no cover - environment-specific
        print(f"[synth_gsm8k] dataset unavailable ({exc!r}); writing synthetic fallback.")
        with open(args.out, "w", encoding="utf-8") as handle:
            for _ in range((args.n + len(SYNTHETIC) - 1) // len(SYNTHETIC)):
                for row in SYNTHETIC:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= args.n:
                        break
                if written >= args.n:
                    break
    print(f"[synth_gsm8k] wrote {written} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
