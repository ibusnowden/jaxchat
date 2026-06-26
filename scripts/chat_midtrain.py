"""Midtraining entry point for chat, MCQ, and tool-use traces.

This intentionally reuses the SFT trainer and checkpoint schema. A midtrain run
saves an ``sft`` stage checkpoint; final SFT can then use
``scripts.chat_sft --parent-stage sft --base-run-dir <midtrain-run>``.
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts import chat_sft  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--midtrain-data", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-iters", type=int, default=800)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lr-scale", type=float, default=0.1)
    parser.add_argument("--tokenizer-json", default=None)
    args = parser.parse_args(argv)

    forwarded = [
        "--base-run-dir", args.base_run_dir,
        "--parent-stage", "base",
        "--sft-data", args.midtrain_data,
        "--run-dir", args.run_dir,
        "--n-iters", str(args.n_iters),
        "--max-seq-len", str(args.max_seq_len),
        "--lr-scale", str(args.lr_scale),
    ]
    if args.tokenizer_json:
        forwarded.extend(["--tokenizer-json", args.tokenizer_json])
    return chat_sft.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
