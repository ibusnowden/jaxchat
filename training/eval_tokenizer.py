"""Lightweight tokenizer evaluation for the staged JAX speedrun."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __package__ in {None, ""}:
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from jaxchat.tokenizer import (
    DEFAULT_FINEWEB_TOKENIZER_DATASETS,
    iter_hf_dataset_text,
    load_hf_tokenizer,
    resolve_tokenizer_json_path,
)


ROUNDTRIP_TEXT = "Hello tokenizer.\nThis is a roundtrip check."


def _resolve_output_path(tokenizer_path_or_dir: str) -> str:
    tokenizer_json = resolve_tokenizer_json_path(tokenizer_path_or_dir)
    tokenizer_dir = os.path.dirname(tokenizer_json) or "."
    return os.path.join(tokenizer_dir, "tokenizer_eval.json")


def _compute_sample_metrics(tokenizer, sample_texts: Iterable[str]) -> dict[str, float | int]:
    sampled_documents = 0
    total_tokens = 0
    total_chars = 0
    total_utf8_bytes = 0

    for text in sample_texts:
        if not text:
            continue
        encoded = tokenizer.encode(text)
        if not encoded:
            continue
        sampled_documents += 1
        total_tokens += len(encoded)
        total_chars += len(text)
        total_utf8_bytes += len(text.encode("utf-8"))

    if total_tokens == 0 or sampled_documents == 0:
        return {
            "sampled_documents": sampled_documents,
            "sampled_total_tokens": total_tokens,
            "sampled_total_chars": total_chars,
            "sampled_total_utf8_bytes": total_utf8_bytes,
            "avg_tokens_per_document": 0.0,
            "avg_chars_per_token": 0.0,
            "avg_bytes_per_token": 0.0,
            "utf8_bytes_per_document": 0.0,
        }

    return {
        "sampled_documents": sampled_documents,
        "sampled_total_tokens": total_tokens,
        "sampled_total_chars": total_chars,
        "sampled_total_utf8_bytes": total_utf8_bytes,
        "avg_tokens_per_document": total_tokens / sampled_documents,
        "avg_chars_per_token": total_chars / total_tokens,
        "avg_bytes_per_token": total_utf8_bytes / total_tokens,
        "utf8_bytes_per_document": total_utf8_bytes / sampled_documents,
    }


def evaluate_tokenizer(
    tokenizer_path_or_dir: str,
    *,
    dataset_names: str = ",".join(DEFAULT_FINEWEB_TOKENIZER_DATASETS),
    dataset_configs: str | None = None,
    split: str = "train",
    text_field: str = "text",
    max_documents: int = 128,
    output_path: str | None = None,
    sample_texts: Sequence[str] | None = None,
) -> dict[str, object]:
    tokenizer = load_hf_tokenizer(tokenizer_path_or_dir)
    bos_id = int(tokenizer.get_bos_token_id())
    bos_token = tokenizer.id_to_token(bos_id)

    encoded = tokenizer.encode(ROUNDTRIP_TEXT)
    decoded = tokenizer.decode(encoded)

    if sample_texts is None:
        sample_texts = tuple(
            iter_hf_dataset_text(
                dataset_names,
                dataset_configs=dataset_configs,
                split=split,
                text_field=text_field,
                max_documents=max_documents,
            )
        )

    sample_metrics = _compute_sample_metrics(tokenizer, sample_texts)
    report = {
        "tokenizer_present": True,
        "tokenizer_json": resolve_tokenizer_json_path(tokenizer_path_or_dir),
        "vocab_size": int(tokenizer.get_vocab_size()),
        "bos_token_id": bos_id,
        "bos_token": bos_token,
        "roundtrip_text": ROUNDTRIP_TEXT,
        "roundtrip_matches": decoded == ROUNDTRIP_TEXT,
        "roundtrip_token_count": len(encoded),
        "dataset_name": dataset_names,
        "dataset_config": dataset_configs,
        "split": split,
        "text_field": text_field,
        "max_documents": max_documents,
    } | sample_metrics

    final_output_path = output_path or _resolve_output_path(tokenizer_path_or_dir)
    os.makedirs(os.path.dirname(final_output_path) or ".", exist_ok=True)
    with open(final_output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the staged FineWeb tokenizer.")
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument(
        "--dataset-name",
        default=",".join(DEFAULT_FINEWEB_TOKENIZER_DATASETS),
        help="Comma-separated dataset names used for lightweight compression sampling.",
    )
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-documents", type=int, default=128)
    args = parser.parse_args(argv)

    report = evaluate_tokenizer(
        args.tokenizer_dir,
        dataset_names=args.dataset_name,
        dataset_configs=args.dataset_config,
        split=args.split,
        text_field=args.text_field,
        max_documents=args.max_documents,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
