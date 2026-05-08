"""Chat-stage eval: CORE-subset accuracy + GSM8K accuracy."""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jaxchat.model as model_lib  # noqa: E402

model_lib.configure_jax_runtime()

from jaxchat.engine import Engine  # noqa: E402
from tasks.core import run_subset  # noqa: E402
from tasks import gsm8k  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a chat (sft/rl) checkpoint.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage", default=None, choices=(None, "base", "sft", "rl"))
    parser.add_argument("--core-n", type=int, default=200)
    parser.add_argument("--gsm8k-n", type=int, default=100)
    parser.add_argument("--gsm8k-max-new-tokens", type=int, default=256)
    parser.add_argument("--skip-core", action="store_true")
    parser.add_argument("--skip-gsm8k", action="store_true")
    parser.add_argument("--tokenizer-json", default=None)
    args = parser.parse_args(argv)

    engine = Engine.from_run_dir(args.run_dir, stage=args.stage, tokenizer_path=args.tokenizer_json)
    out: dict = {"run_dir": os.path.abspath(args.run_dir), "stage": engine.stage, "step": engine.step}

    if not args.skip_core:
        out["core"] = run_subset(engine, n_per_task=args.core_n)
    if not args.skip_gsm8k:
        try:
            out["gsm8k"] = gsm8k.evaluate(engine, n_max=args.gsm8k_n, max_new_tokens=args.gsm8k_max_new_tokens)
        except Exception as exc:  # pragma: no cover - environment-specific
            out["gsm8k"] = {"error": f"{type(exc).__name__}: {exc}"}

    out_path = os.path.join(out["run_dir"], "chat_eval.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
