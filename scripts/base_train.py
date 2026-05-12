"""Pretraining entry point (nanochat-style: ``python -m scripts.base_train``).

Thin wrapper around :func:`training.train_base.train_loop` so the canonical
pipeline order — tok_train → base_train → base_eval → chat_sft → chat_rl →
chat_eval → chat_cli — is uniformly invoked from ``scripts/``.
"""

from __future__ import annotations

import argparse
import ast
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


_CONFIG_FIELD_TYPES = {f.name: f.type for f in dataclasses.fields(model_lib.Config)}


def _coerce_override(name: str, raw: str):
    """Coerce a ``key=value`` string to the type of ``Config.<key>``.

    bools accept true/false/1/0/yes/no; tuples/lists go through ``ast.literal_eval``;
    everything else is parsed by the field's annotated type, falling back to a
    best-effort literal_eval then the raw string.
    """
    if name not in _CONFIG_FIELD_TYPES:
        raise SystemExit(f"--config-override: unknown Config field {name!r}")
    ftype = _CONFIG_FIELD_TYPES[name]
    type_str = str(ftype)
    if "bool" in type_str:
        v = raw.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
        raise SystemExit(f"--config-override: {name}={raw!r} is not a bool")
    if "tuple" in type_str or "list" in type_str:
        return tuple(ast.literal_eval(raw))
    if ftype is int or type_str == "int":
        return int(raw)
    if ftype is float or type_str == "float":
        return float(raw)
    if ftype is str or type_str == "str":
        return raw
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


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
    if args.optimizer is not None:
        overrides["optimizer"] = args.optimizer
    if args.lr_schedule is not None:
        overrides["lr_schedule"] = args.lr_schedule
    if args.weight_tying is not None:
        overrides["weight_tying"] = args.weight_tying
    for item in args.config_override or []:
        if "=" not in item:
            raise SystemExit(f"--config-override expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip()
        overrides[key] = _coerce_override(key, raw)
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
    parser.add_argument("--optimizer", choices=("muon_adamw", "normuon", "soap"), default=None,
                        help="Optimizer (default: muon_adamw)")
    parser.add_argument("--lr-schedule", choices=("linear", "cosine", "wsd"), default=None,
                        help="LR schedule type")
    parser.add_argument("--weight-tying", choices=("none", "full", "delayed"), default=None,
                        help="Weight tying mode")
    parser.add_argument("--config-override", action="append", metavar="KEY=VALUE", default=[],
                        help="Override an arbitrary Config field, e.g. --config-override train_token_ratio=20 "
                             "(repeatable; types are coerced from the dataclass field).")
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
