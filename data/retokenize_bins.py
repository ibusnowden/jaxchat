"""Offline re-tokenizer: GPT-2 ``*.bin`` shards -> 32k-BPE packed ``*.bin`` shards.

The legacy ``data/fineweb10B`` shards (the modded-nanogpt FineWeb format, header
magic ``20240520``) are tokenized with GPT-2 BPE (vocab 50257), but the jaxchat
model and tokenizer use a 32k custom BPE.  Feeding GPT-2 token ids into a 32768-row
embedding table is the cause of the immediate ``loss: nan`` on every 124M run.

This script reconstructs the source text losslessly (GPT-2 BPE is byte-level and
reversible), re-encodes it with the 32k tokenizer, and repacks it into fixed
``seq_len`` sequences using the same best-fit packer as ``data/cached_fineweb.py``.
No internet required -- it only reads bytes that are already on disk.

Usage (from the repo root)::

    python -m data.retokenize_bins \
        --source-dir data/fineweb10B \
        --output-dir data/fineweb32k_real \
        --tokenizer-json data/fineweb32k/tokenizer.json \
        --seq-len 1024 \
        --train-target-tokens 0   # 0 = use all available source tokens

Then point the preset / launchers at ``--output-dir``.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Iterator

import numpy as np

if __package__ in {None, ""}:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

import tiktoken

from data.cached_fineweb import BinWriter, best_fit_crop_sequences  # noqa: E402
from jaxchat.tokenizer import find_existing_tokenizer_path, load_tokenizer  # noqa: E402

BIN_HEADER_INTS = 256
BIN_HEADER_BYTES = BIN_HEADER_INTS * 4
BIN_MAGIC = 20240520
BIN_VERSION = 1

# GPT-2 <|endoftext|> -- delimits documents in the source shards.
GPT2_EOT = 50256


def _open_source_shard(path: str) -> np.memmap:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(BIN_HEADER_BYTES), dtype=np.int32)
    if int(header[0]) != BIN_MAGIC:
        raise RuntimeError(f"Magic number mismatch in {path}: {int(header[0])}")
    if int(header[1]) != BIN_VERSION:
        raise RuntimeError(f"Unsupported version in {path}: {int(header[1])}")
    n_tokens = int(header[2])
    return np.memmap(path, mode="r", dtype=np.uint16, offset=BIN_HEADER_BYTES, shape=(n_tokens,))


def _iter_source_docs(paths: list[str]) -> Iterator[np.ndarray]:
    """Yield each document (a run of GPT-2 token ids between EOT markers)."""
    for path in paths:
        tokens = _open_source_shard(path)
        # Indices where a new document starts (EOT token positions).
        bounds = np.flatnonzero(tokens == GPT2_EOT)
        starts = np.concatenate([[0], bounds + 1])
        ends = np.concatenate([bounds, [len(tokens)]])
        for s, e in zip(starts, ends):
            if e <= s:
                continue
            doc = np.asarray(tokens[s:e])
            # Strip any stray EOT (shouldn't happen after the split, but be safe).
            if doc.size and doc[0] == GPT2_EOT:
                doc = doc[1:]
            if doc.size:
                yield doc


def _retokenized_docs(
    source_paths: list[str],
    *,
    gpt2_enc: "tiktoken.Encoding",
    hf_tok,
    vocab_size: int,
    batch_docs: int,
) -> Iterator[np.ndarray]:
    """Decode GPT-2 docs -> text -> re-encode with the 32k tokenizer, in batches."""
    batch: list[np.ndarray] = []
    n_docs = 0
    n_src_tok = 0
    n_dst_tok = 0
    t0 = time.time()

    def flush() -> Iterator[np.ndarray]:
        nonlocal n_docs, n_dst_tok
        if not batch:
            return
        texts = gpt2_enc.decode_batch([d.tolist() for d in batch])
        if hasattr(hf_tok, "tokenizer"):
            encs = [enc.ids for enc in hf_tok.tokenizer.encode_batch(texts, add_special_tokens=False)]
        else:
            encs = hf_tok.encode(texts)
        for ids in encs:
            if not ids:
                continue
            arr = np.asarray(ids, dtype=np.uint16)
            if arr.size and int(arr.max()) >= vocab_size:
                raise RuntimeError(
                    f"Re-encoded token id {int(arr.max())} >= vocab_size {vocab_size}"
                )
            n_docs += 1
            n_dst_tok += arr.size
            yield arr
        batch.clear()

    for doc in _iter_source_docs(source_paths):
        n_src_tok += doc.size
        batch.append(doc)
        if len(batch) >= batch_docs:
            yield from flush()
            if n_docs and n_docs % (batch_docs * 50) < batch_docs:
                rate = n_src_tok / max(time.time() - t0, 1e-6)
                print(
                    f"  ... {n_docs:,} docs | src_tok={n_src_tok:,} dst_tok={n_dst_tok:,} "
                    f"| {rate/1e6:.2f}M src tok/s",
                    flush=True,
                )
    yield from flush()
    print(
        f"  done re-encoding: {n_docs:,} docs | src_tok={n_src_tok:,} dst_tok={n_dst_tok:,} "
        f"| ratio dst/src={n_dst_tok/max(n_src_tok,1):.3f}",
        flush=True,
    )


def _concat_sequences(doc_iter: Iterator[np.ndarray], *, bos_id: int, chunk_tokens: int) -> Iterator[np.ndarray]:
    """Plain nanoGPT-style packing: [bos, *doc, bos, *doc, ...] with no cropping.

    Yields contiguous ``chunk_tokens``-sized arrays (the final chunk may be shorter).
    The training loader windows this stream arbitrarily; ``cross_document_mask`` (when
    enabled) keys on ``bos_id`` to stop attention/loss bleeding across documents.
    """
    buf: list[np.ndarray] = []
    have = 0
    bos = np.asarray([bos_id], dtype=np.uint16)
    for doc in doc_iter:
        buf.append(bos)
        buf.append(np.asarray(doc, dtype=np.uint16))
        have += 1 + doc.size
        while have >= chunk_tokens:
            stream = np.concatenate(buf)
            yield np.asarray(stream[:chunk_tokens], dtype=np.uint16)
            rest = stream[chunk_tokens:]
            buf = [rest] if rest.size else []
            have = rest.size
    if have:
        yield np.concatenate(buf).astype(np.uint16)


def _pack_split(
    *,
    source_paths: list[str],
    output_dir: str,
    prefix: str,
    gpt2_enc,
    hf_tok,
    vocab_size: int,
    bos_id: int,
    seq_len: int,
    shard_tokens: int,
    target_tokens: int,
    buffer_docs: int,
    batch_docs: int,
    pack_mode: str,
) -> int:
    shard_capacity = max((shard_tokens // seq_len) * seq_len, seq_len)
    writer = BinWriter(output_dir=output_dir, prefix=prefix, shard_token_capacity=shard_capacity)
    doc_iter = _retokenized_docs(
        source_paths,
        gpt2_enc=gpt2_enc,
        hf_tok=hf_tok,
        vocab_size=vocab_size,
        batch_docs=batch_docs,
    )
    if pack_mode == "concat":
        seq_iter = _concat_sequences(doc_iter, bos_id=bos_id, chunk_tokens=shard_capacity)
    else:
        seq_iter = best_fit_crop_sequences(doc_iter, seq_len=seq_len, bos_id=bos_id, buffer_docs=buffer_docs)
    emitted = 0
    try:
        for seq in seq_iter:
            if target_tokens > 0 and emitted + int(seq.size) > target_tokens:
                seq = seq[: max(target_tokens - emitted, 0)]
                if seq.size:
                    writer.write_sequence(np.asarray(seq, dtype=np.uint16))
                    emitted += int(seq.size)
                break
            writer.write_sequence(np.asarray(seq, dtype=np.uint16))
            emitted += int(seq.size)
        writer.close()
    finally:
        writer.close()
    print(
        f"[{prefix}] emitted_tokens={writer.total_tokens:,} shards={writer.shard_index + 1} "
        f"pack_mode={pack_mode} dir={output_dir}",
        flush=True,
    )
    return writer.total_tokens


def _verify(output_dir: str, prefix: str, vocab_size: int) -> None:
    for path in sorted(glob.glob(os.path.join(output_dir, f"{prefix}_*.bin"))):
        toks = _open_source_shard(path)
        sample = np.asarray(toks[: min(len(toks), 5_000_000)])
        lo, hi = int(sample.min()), int(sample.max())
        assert hi < vocab_size, f"{path}: max token {hi} >= vocab {vocab_size}"
        print(f"  verify {os.path.basename(path)}: n={len(toks):,} min={lo} max={hi}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="data/fineweb10B")
    parser.add_argument(
        "--train-glob", default="fineweb_train_*.bin", help="glob (within source-dir) for train shards"
    )
    parser.add_argument(
        "--val-glob", default="fineweb_val_*.bin", help="glob (within source-dir) for val shards"
    )
    parser.add_argument("--output-dir", default="data/fineweb32k_real")
    parser.add_argument("--tokenizer-json", default="data/fineweb32k/tokenizer.json",
                        help="Tokenizer artifact path/dir; accepts tokenizer.json or tokenizer.pkl.")
    parser.add_argument("--gpt2-encoding", default="gpt2")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--bos-id", type=int, default=-1, help="-1 = use tokenizer's <|bos|> id")
    parser.add_argument("--train-target-tokens", type=int, default=0, help="0 = all available")
    parser.add_argument("--val-target-tokens", type=int, default=0, help="0 = all available")
    parser.add_argument("--buffer-docs", type=int, default=1024)
    parser.add_argument("--batch-docs", type=int, default=2048, help="docs per decode/encode batch")
    parser.add_argument(
        "--pack-mode",
        choices=("concat", "crop"),
        default="concat",
        help="'concat': nanoGPT-style [bos,*doc,...] stream, no token loss (default). "
        "'crop': fixed-length best-fit packing that drops doc tails.",
    )
    parser.add_argument("--copy-tokenizer", action="store_true", help="also copy tokenizer artifact into output-dir")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    src = args.source_dir
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    train_paths = sorted(glob.glob(os.path.join(src, args.train_glob)))
    val_paths = sorted(glob.glob(os.path.join(src, args.val_glob)))
    if not train_paths:
        raise SystemExit(f"No train shards matched {os.path.join(src, args.train_glob)}")
    if not val_paths:
        raise SystemExit(f"No val shards matched {os.path.join(src, args.val_glob)}")

    tok_path = find_existing_tokenizer_path(args.tokenizer_json) or args.tokenizer_json
    hf_tok = load_tokenizer(tok_path)
    vocab_size = hf_tok.get_vocab_size()
    bos_id = args.bos_id if args.bos_id >= 0 else int(hf_tok.get_bos_token_id())
    gpt2_enc = tiktoken.get_encoding(args.gpt2_encoding)

    print(
        f"source={src} | train_shards={len(train_paths)} val_shards={len(val_paths)} | "
        f"tokenizer={tok_path} vocab={vocab_size} bos_id={bos_id} seq_len={args.seq_len}",
        flush=True,
    )

    t0 = time.time()
    _pack_split(
        source_paths=train_paths,
        output_dir=out,
        prefix="fineweb_train",
        gpt2_enc=gpt2_enc,
        hf_tok=hf_tok,
        vocab_size=vocab_size,
        bos_id=bos_id,
        seq_len=args.seq_len,
        shard_tokens=args.shard_tokens,
        target_tokens=args.train_target_tokens,
        buffer_docs=args.buffer_docs,
        batch_docs=args.batch_docs,
        pack_mode=args.pack_mode,
    )
    _pack_split(
        source_paths=val_paths,
        output_dir=out,
        prefix="fineweb_val",
        gpt2_enc=gpt2_enc,
        hf_tok=hf_tok,
        vocab_size=vocab_size,
        bos_id=bos_id,
        seq_len=args.seq_len,
        shard_tokens=args.shard_tokens,
        target_tokens=args.val_target_tokens,
        buffer_docs=args.buffer_docs,
        batch_docs=args.batch_docs,
        pack_mode=args.pack_mode,
    )

    if args.copy_tokenizer:
        import shutil

        shutil.copyfile(tok_path, os.path.join(out, os.path.basename(tok_path)))

    print("verifying output shards ...", flush=True)
    _verify(out, "fineweb_train", vocab_size)
    _verify(out, "fineweb_val", vocab_size)
    print(f"done in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
