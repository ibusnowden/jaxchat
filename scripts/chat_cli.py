"""Interactive CLI: chat with a trained jaxchat checkpoint."""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jaxchat.model as model_lib  # noqa: E402

model_lib.configure_jax_runtime()

from jaxchat.engine import Engine  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat REPL backed by a jaxchat checkpoint.")
    parser.add_argument("--run-dir", required=True, help="Directory produced by base_train / chat_sft / chat_rl.")
    parser.add_argument("--stage", default=None, choices=(None, "base", "sft", "rl"))
    parser.add_argument("--tokenizer-json", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt",
        default=None,
        help="If given, run a single non-interactive turn and exit.",
    )
    args = parser.parse_args(argv)

    engine = Engine.from_run_dir(args.run_dir, stage=args.stage, tokenizer_path=args.tokenizer_json)
    print(f"Loaded {engine.stage} stage @ step {engine.step}")

    if args.prompt is not None:
        text = engine.chat(
            [{"role": "user", "content": args.prompt}],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )
        print(text)
        return 0

    messages: list[dict] = []
    print("(type 'exit' to quit, 'reset' to clear history)")
    seed = args.seed
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break
        if user.lower() == "reset":
            messages = []
            print("(history cleared)")
            continue
        messages.append({"role": "user", "content": user})
        seed += 1
        reply = engine.chat(
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=seed,
        )
        print(f"bot> {reply}")
        messages.append({"role": "assistant", "content": reply})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
