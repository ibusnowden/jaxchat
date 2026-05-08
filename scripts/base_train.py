"""Pretraining entry point (nanochat-style: ``python -m scripts.base_train``).

Thin wrapper around :func:`training.train_base.train_loop` so the canonical
pipeline order — tok_train → base_train → base_eval → chat_sft → chat_rl →
chat_eval → chat_cli — is uniformly invoked from ``scripts/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jaxchat.model as model_lib  # noqa: E402

model_lib.configure_jax_runtime()

from jaxchat.presets import PRESETS  # noqa: E402
from training.train_base import train_loop  # noqa: E402


def _build_config(args):
    config = PRESETS[args.preset]
    preset_name = args.preset
    overrides = {}
    if args.depth is not None:
        overrides["depth"] = args.depth
        preset_name = f"depth{args.depth}"
    if args.vocab_size is not None:
        overrides["vocab_size"] = args.vocab_size
    if args.input_bin:
        overrides["input_bin"] = args.input_bin
    if args.input_val_bin:
        overrides["input_val_bin"] = args.input_val_bin
    if args.tokenizer_json is not None:
        overrides["tokenizer_json"] = args.tokenizer_json
    if overrides:
        config = dataclasses.replace(config, **overrides)
    return config, preset_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pretrain the jaxchat base model.")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="d4")
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--input-bin", default="")
    parser.add_argument("--input-val-bin", default="")
    parser.add_argument("--tokenizer-json", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-iters", type=int, default=None)
    args = parser.parse_args(argv)

    config, preset_name = _build_config(args)
    train_loop(
        config,
        preset_name=preset_name,
        run_dir=args.run_dir,
        resume=args.resume,
        smoke_iters=args.smoke_iters,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
