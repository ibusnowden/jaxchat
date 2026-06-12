"""Materialize GSM8K as JSONL for local RL/eval/SFT.

Two output modes:

* ``--mode rl``  (default): one ``{"question", "answer"}`` row per example — the
  format ``scripts.chat_rl`` / ``tasks.gsm8k.evaluate`` expect.
* ``--mode sft``: one ``{"messages": [system, user, assistant]}`` row with a
  ``\\boxed{}``-terminated chain-of-thought target (via
  :func:`tasks.gsm8k.build_sft_messages`). Feed these into ``scripts.chat_sft``
  so the policy learns to *emit* the boxed answer format — without that, a
  smoltalk-only SFT model never boxes and GRPO's reward stays pinned at 0.

Source is either the HuggingFace ``openai/gsm8k`` dataset (default) or, for
offline compute nodes, a local ``--from-jsonl`` of ``{question, answer}`` rows
(e.g. the already-downloaded ``gsm8k_train_4k.jsonl``). If neither is reachable
a tiny synthetic fallback is written so the pipeline still wires end-to-end.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tasks import gsm8k  # noqa: E402


SYNTHETIC = [
    {"question": "What is 2 + 3?", "answer": "Adding gives 5.\n#### 5"},
    {"question": "What is 6 * 7?", "answer": "Multiplying 6 and 7 gives 42.\n#### 42"},
    {"question": "If a pen costs 4 dollars and you buy 3 pens, how much do you spend?", "answer": "3 * 4 = 12.\n#### 12"},
    {"question": "What is half of 18?", "answer": "18 / 2 = 9.\n#### 9"},
]


def _iter_local(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "question" in row and "answer" in row:
                yield {"question": row["question"], "answer": row["answer"]}


def _iter_hf(split: str):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split, streaming=True)
    for row in ds:
        yield {"question": row["question"], "answer": row["answer"]}


def _format_row(row: dict, mode: str) -> dict:
    if mode == "sft":
        return gsm8k.build_sft_messages(row["question"], row["answer"])
    return {"question": row["question"], "answer": row["answer"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize GSM8K splits as JSONL.")
    parser.add_argument("--out", required=True, help="Output JSONL.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--mode", choices=("rl", "sft"), default="rl",
                        help="rl = {question,answer}; sft = {messages} with boxed CoT.")
    parser.add_argument("--from-jsonl", default=None,
                        help="Convert a local {question,answer} JSONL instead of hitting HF (offline-safe).")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    if args.from_jsonl:
        source = _iter_local(args.from_jsonl)
        src_desc = f"local {args.from_jsonl}"
    else:
        try:
            source = _iter_hf(args.split)
            src_desc = f"HF openai/gsm8k[{args.split}]"
        except Exception as exc:  # pragma: no cover - environment-specific
            print(f"[synth_gsm8k] dataset unavailable ({exc!r}); writing synthetic fallback.")
            source = (SYNTHETIC[i % len(SYNTHETIC)] for i in range(args.n))
            src_desc = "synthetic fallback"

    written = skipped = 0
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in source:
            try:
                out_row = _format_row(row, args.mode)
            except Exception:  # malformed answer — skip rather than abort
                skipped += 1
                continue
            handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written += 1
            if written >= args.n:
                break

    print(f"[synth_gsm8k] wrote {written} rows ({args.mode}) from {src_desc} to {args.out}"
          + (f" (skipped {skipped} malformed)" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
