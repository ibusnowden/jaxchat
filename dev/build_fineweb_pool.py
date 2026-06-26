"""Rebuild the 32k-tokenized FineWeb-Edu pretraining pool (deleted 2026-06-12).

Streams ``HuggingFaceFW/fineweb-edu`` (sample-10BT) — the same source the
original ``fineweb10B``/``fineweb32k_real_29`` pool came from (modded-nanogpt's
``fineweb10B`` dir IS the FineWeb-Edu 10B sample; the 32k tokenizer was trained
on FineWeb-Edu per the README) — and packs it into ``seq_len`` sequences with
the surviving 32k tokenizer.

Train/val are kept DISJOINT: we emit one contiguous stream of ``--shards`` x
``--shard-tokens`` token shards, then rename the LAST shard (the stream tail) to
``fineweb_val_000000.bin``.  The stock ``data.cached_fineweb`` CLI instead reads
both train and val from the start of ``split=train``, which overlaps them — an
optimistic val_bpb we explicitly do not want for the post-bigram-fix rerun.

Requires internet -> run on the LOGIN node (compute nodes are offline).

Usage::

    .venv/bin/python -m dev.build_fineweb_pool \
        --output-dir data/fineweb32k_real_29 \
        --tokenizer-json data/124m_rtx_run/tokenizer/tokenizer.json \
        --shards 30 --shard-tokens 100000000 --seq-len 1024
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.cached_fineweb import preprocess_split  # noqa: E402
from jaxchat.tokenizer import find_existing_tokenizer_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--text-field", default="text")
    p.add_argument("--tokenizer-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--shard-tokens", type=int, default=100_000_000)
    p.add_argument("--shards", type=int, default=30,
                   help="Total shards to emit; the last becomes the val shard.")
    p.add_argument("--bos-token", default="<|bos|>")
    p.add_argument("--buffer-docs", type=int, default=1024)
    args = p.parse_args(list(sys.argv[1:] if argv is None else argv))

    tok = find_existing_tokenizer_path(args.tokenizer_json) or args.tokenizer_json
    if not os.path.exists(tok):
        print(f"ERROR: tokenizer not found at {tok}", file=sys.stderr)
        return 2
    os.makedirs(args.output_dir, exist_ok=True)

    # Each shard fills to (shard_tokens // seq_len) * seq_len tokens; target the
    # exact multiple so we get `--shards` full shards and no tiny tail shard.
    per_shard = max((args.shard_tokens // args.seq_len) * args.seq_len, args.seq_len)
    target = per_shard * args.shards

    print(f"[build_fineweb_pool] dataset={args.dataset_name}:{args.dataset_config} "
          f"seq_len={args.seq_len} per_shard={per_shard:,} shards={args.shards} "
          f"target_tokens={target:,} -> {args.output_dir}", flush=True)

    t0 = time.time()
    preprocess_split(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split="train",
        text_field=args.text_field,
        tokenizer_json=tok,
        bos_token=args.bos_token,
        bos_id=None,
        seq_len=args.seq_len,
        shard_tokens=args.shard_tokens,
        output_dir=args.output_dir,
        prefix="fineweb_train",
        target_tokens=target,
        buffer_docs=args.buffer_docs,
        tokenizer_train_if_missing=False,
        tokenizer_vocab_size=32768,
        tokenizer_max_documents=None,
    )
    dt = time.time() - t0

    shards = sorted(glob.glob(os.path.join(args.output_dir, "fineweb_train_*.bin")))
    if len(shards) < 2:
        print(f"ERROR: expected >=2 shards, got {len(shards)}", file=sys.stderr)
        return 3
    val_src = shards[-1]
    val_dst = os.path.join(args.output_dir, "fineweb_val_000000.bin")
    os.replace(val_src, val_dst)  # last (tail) shard -> disjoint val

    # Stage the tokenizer alongside the bins (base/eval read ${DATA_DIR}/tokenizer.json).
    staged_tok = os.path.join(args.output_dir, "tokenizer.json")
    if os.path.abspath(tok) != os.path.abspath(staged_tok):
        shutil.copyfile(tok, staged_tok)

    train_shards = sorted(glob.glob(os.path.join(args.output_dir, "fineweb_train_*.bin")))
    print(f"[build_fineweb_pool] DONE in {dt/60:.1f} min | "
          f"train_shards={len(train_shards)} (~{len(train_shards)*per_shard:,} tok) | "
          f"val=fineweb_val_000000.bin (~{per_shard:,} tok) | tokenizer staged", flush=True)
    return 0


if __name__ == "__main__":
    _rc = main()
    # All shards are written, headers flushed, and the val rename/tokenizer copy
    # are done synchronously inside main() before it returns.  HF datasets keeps a
    # background parquet-download thread alive whose teardown races with the GIL
    # during interpreter finalization and aborts with SIGABRT (cosmetic, but it
    # would poison the exit code a dependency chain checks).  Skip finalization.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)
