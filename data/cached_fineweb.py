"""FineWeb preprocessing helpers for the 32k-token JAX training stack."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np

if __package__ in {None, ""}:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from jaxchat.tokenizer import (
    DEFAULT_FINEWEB_TOKENIZER_DATASETS,
    DEFAULT_FINEWEB_TOKENIZER_DIR,
    DEFAULT_FINEWEB_VOCAB_SIZE,
    ensure_tokenizer,
    iter_hf_dataset_rows,
    find_existing_tokenizer_path,
)


BIN_MAGIC = 20240520
BIN_VERSION = 1
BIN_HEADER_INTS = 256
BIN_HEADER_BYTES = BIN_HEADER_INTS * 4

def _write_header(handle, n_tokens: int) -> None:
    header = np.zeros(BIN_HEADER_INTS, dtype=np.int32)
    header[0] = BIN_MAGIC
    header[1] = BIN_VERSION
    header[2] = n_tokens
    handle.seek(0)
    handle.write(header.tobytes())


@dataclass
class BinWriter:
    output_dir: str
    prefix: str
    shard_token_capacity: int

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.shard_index = 0
        self.total_tokens = 0
        self.current_tokens = 0
        self.handle = None
        self.current_path = None
        self._open_next()

    def _open_next(self) -> None:
        if self.handle is not None:
            self.close()
        self.current_path = os.path.join(
            self.output_dir, f"{self.prefix}_{self.shard_index:06d}.bin"
        )
        self.handle = open(self.current_path, "wb")
        self.handle.write(b"\0" * BIN_HEADER_BYTES)
        self.current_tokens = 0

    def write_sequence(self, tokens: np.ndarray) -> None:
        if tokens.dtype != np.uint16:
            raise ValueError(f"Expected uint16 tokens, got {tokens.dtype}")
        if self.current_tokens + int(tokens.size) > self.shard_token_capacity and self.current_tokens > 0:
            self.shard_index += 1
            self._open_next()
        self.handle.write(tokens.tobytes())
        self.current_tokens += int(tokens.size)
        self.total_tokens += int(tokens.size)

    def close(self) -> None:
        if self.handle is None:
            return
        _write_header(self.handle, self.current_tokens)
        self.handle.close()
        self.handle = None


def load_tokenizer(
    *,
    tokenizer_path_or_dir: str,
    train_if_missing: bool,
    dataset_names: str,
    dataset_configs: str | None,
    split: str,
    text_field: str,
    vocab_size: int,
    max_documents: int | None,
):
    return ensure_tokenizer(
        tokenizer_path_or_dir,
        impl="auto",
        train_if_missing=train_if_missing,
        dataset_names=dataset_names,
        dataset_configs=dataset_configs,
        split=split,
        text_field=text_field,
        vocab_size=vocab_size,
        max_documents=max_documents,
    )


def resolve_bos_id(tokenizer, bos_token: str, bos_id: int | None) -> int:
    if bos_id is not None:
        return int(bos_id)
    if bos_token == "<|bos|>":
        return int(tokenizer.get_bos_token_id())
    resolved = tokenizer.encode_special(bos_token)
    if resolved is None:
        raise RuntimeError(
            f"Could not resolve BOS token {bos_token!r} from the provided tokenizer."
        )
    return int(resolved)


def iter_tokenized_documents(rows, tokenizer, text_field: str):
    for row in rows:
        text = row.get(text_field)
        if not isinstance(text, str) or not text:
            continue
        encoded = tokenizer.encode(text)
        if not encoded:
            continue
        yield np.asarray(encoded, dtype=np.uint16)


def choose_best_fit(buffer: list[np.ndarray], remaining: int) -> int:
    fits = [idx for idx, doc in enumerate(buffer) if doc.size >= remaining]
    if fits:
        return min(fits, key=lambda idx: int(buffer[idx].size - remaining))
    return max(range(len(buffer)), key=lambda idx: int(buffer[idx].size))


def best_fit_crop_sequences(doc_iter, *, seq_len: int, bos_id: int, buffer_docs: int):
    payload_len = seq_len - 1
    buffer: list[np.ndarray] = []
    stats = {"documents": 0, "cropped_tokens": 0, "emitted_tokens": 0, "sequences": 0}

    def refill() -> None:
        while len(buffer) < buffer_docs:
            try:
                buffer.append(next(doc_iter))
                stats["documents"] += 1
            except StopIteration:
                break

    refill()
    while buffer:
        remaining = payload_len
        pieces = [np.asarray([bos_id], dtype=np.uint16)]
        while remaining > 0:
            refill()
            if not buffer:
                return stats
            best_idx = choose_best_fit(buffer, remaining)
            doc = buffer.pop(best_idx)
            take = min(int(doc.size), remaining)
            pieces.append(doc[:take])
            remaining -= take
            stats["cropped_tokens"] += max(int(doc.size) - take, 0)
        seq = np.concatenate(pieces, axis=0)
        if seq.size != seq_len:
            raise RuntimeError(f"Expected sequence length {seq_len}, got {seq.size}")
        stats["emitted_tokens"] += int(seq.size)
        stats["sequences"] += 1
        yield seq


def preprocess_split(
    *,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_field: str,
    tokenizer_json: str,
    bos_token: str,
    bos_id: int | None,
    seq_len: int,
    shard_tokens: int,
    output_dir: str,
    prefix: str,
    target_tokens: int,
    buffer_docs: int,
    tokenizer_train_if_missing: bool,
    tokenizer_vocab_size: int,
    tokenizer_max_documents: int | None,
) -> None:
    tokenizer = load_tokenizer(
        tokenizer_path_or_dir=tokenizer_json,
        train_if_missing=tokenizer_train_if_missing,
        dataset_names=dataset_name,
        dataset_configs=dataset_config,
        split=split,
        text_field=text_field,
        vocab_size=tokenizer_vocab_size,
        max_documents=tokenizer_max_documents,
    )
    bos_token_id = resolve_bos_id(tokenizer, bos_token, bos_id)
    rows = iter_hf_dataset_rows(dataset_name, dataset_configs=dataset_config, split=split)
    doc_iter = iter_tokenized_documents(rows, tokenizer, text_field)
    shard_capacity = max((shard_tokens // seq_len) * seq_len, seq_len)
    writer = BinWriter(output_dir=output_dir, prefix=prefix, shard_token_capacity=shard_capacity)
    emitted = 0
    stats = {"documents": 0, "cropped_tokens": 0, "emitted_tokens": 0, "sequences": 0}

    try:
        for sequence in best_fit_crop_sequences(
            doc_iter, seq_len=seq_len, bos_id=bos_token_id, buffer_docs=buffer_docs
        ):
            if target_tokens > 0 and emitted + int(sequence.size) > target_tokens:
                break
            writer.write_sequence(sequence)
            emitted += int(sequence.size)
        writer.close()
    finally:
        writer.close()

    print(
        f"[{prefix}] emitted_tokens={writer.total_tokens:,} "
        f"approx_shards={writer.shard_index + 1} output_dir={output_dir}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess FineWeb into 32k-token packed bins.")
    argv = list(sys.argv[1:] if argv is None else argv)
    parser.add_argument(
        "--dataset-name",
        default=",".join(DEFAULT_FINEWEB_TOKENIZER_DATASETS),
        help="Comma-separated dataset names to retokenize.",
    )
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokenizer-dir", default=DEFAULT_FINEWEB_TOKENIZER_DIR)
    parser.add_argument("--tokenizer-json", default=None, help="Tokenizer artifact path/dir; accepts tokenizer.json or tokenizer.pkl.")
    parser.add_argument("--train-tokenizer", action="store_true", help="Train the 32k tokenizer if it is missing.")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=DEFAULT_FINEWEB_VOCAB_SIZE)
    parser.add_argument("--tokenizer-max-documents", type=int, default=None)
    parser.add_argument("--bos-token", default="<|bos|>")
    parser.add_argument("--bos-id", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--train-target-tokens", type=int, default=14_533_312_248)
    parser.add_argument("--val-target-tokens", type=int, default=100_000_000)
    parser.add_argument("--buffer-docs", type=int, default=1024)
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "fineweb32k"))
    args = parser.parse_args(argv)
    tokenizer_path = args.tokenizer_json or args.tokenizer_dir
    tokenizer_artifact = find_existing_tokenizer_path(tokenizer_path) or tokenizer_path

    preprocess_split(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.train_split,
        text_field=args.text_field,
        tokenizer_json=tokenizer_artifact,
        bos_token=args.bos_token,
        bos_id=args.bos_id,
        seq_len=args.seq_len,
        shard_tokens=args.shard_tokens,
        output_dir=args.output_dir,
        prefix="fineweb_train",
        target_tokens=args.train_target_tokens,
        buffer_docs=args.buffer_docs,
        tokenizer_train_if_missing=args.train_tokenizer,
        tokenizer_vocab_size=args.tokenizer_vocab_size,
        tokenizer_max_documents=args.tokenizer_max_documents,
    )
    preprocess_split(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.val_split,
        text_field=args.text_field,
        tokenizer_json=tokenizer_artifact,
        bos_token=args.bos_token,
        bos_id=args.bos_id,
        seq_len=args.seq_len,
        shard_tokens=args.shard_tokens,
        output_dir=args.output_dir,
        prefix="fineweb_val",
        target_tokens=args.val_target_tokens,
        buffer_docs=args.buffer_docs,
        tokenizer_train_if_missing=False,
        tokenizer_vocab_size=args.tokenizer_vocab_size,
        tokenizer_max_documents=args.tokenizer_max_documents,
    )
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
