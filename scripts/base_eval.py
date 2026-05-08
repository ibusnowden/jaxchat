"""Base-stage eval: validation BPB plus a CORE-subset accuracy report."""

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
from training.eval_base import evaluate_run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validation BPB + CORE-subset for the base checkpoint.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--core-n", type=int, default=200, help="Examples per CORE task.")
    parser.add_argument("--skip-core", action="store_true", help="Only compute val_bpb and skip CORE.")
    parser.add_argument("--tokenizer-json", default=None)
    args = parser.parse_args(argv)

    bpb_report = evaluate_run(args.run_dir)
    output: dict = dict(bpb_report)

    if not args.skip_core:
        engine = Engine.from_run_dir(args.run_dir, stage="base", tokenizer_path=args.tokenizer_json)
        output["core"] = run_subset(engine, n_per_task=args.core_n)

    out_path = os.path.join(os.path.abspath(args.run_dir), "base_eval.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
