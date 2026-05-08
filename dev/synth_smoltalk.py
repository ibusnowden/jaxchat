"""Generate a small SmolTalk-style JSONL for SFT.

Default source: ``HuggingFaceTB/smoltalk`` (subset ``smol-magpie-ultra``).  Each
output line is ``{"messages": [{"role": ..., "content": ...}, ...]}`` matching
the schema expected by :meth:`jaxchat.tokenizer.HuggingFaceTokenizer.render_conversation`.

If the dataset cannot be downloaded (offline), a tiny synthetic fallback is
written so the rest of the pipeline can still run end-to-end.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


SYNTHETIC_FALLBACK = [
    {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Translate 'good morning' to Spanish."},
            {"role": "assistant", "content": "Good morning in Spanish is 'buenos días'."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Give me a haiku about autumn."},
            {"role": "assistant", "content": "Crisp wind, falling leaves\nAmber tones across the sky\nQuiet woods exhale"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is 7 times 8?"},
            {"role": "assistant", "content": "7 times 8 is 56."},
        ]
    },
]


def _normalize(row: dict) -> dict | None:
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None
    cleaned: list[dict] = []
    for m in msgs:
        role = m.get("role") or m.get("from")
        content = m.get("content") or m.get("value")
        if role in {"system", "user", "assistant"} and isinstance(content, str) and content.strip():
            if role == "system":
                role = "system"
            cleaned.append({"role": role, "content": content})
    if not cleaned or cleaned[-1]["role"] != "assistant":
        return None
    # Skip system-only or assistant-leading conversations.
    if cleaned[0]["role"] == "assistant":
        return None
    return {"messages": cleaned}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a SmolTalk-style SFT JSONL.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--dataset", default="HuggingFaceTB/smoltalk")
    parser.add_argument("--config", default="smol-magpie-ultra")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-chars", type=int, default=4000)
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    written = 0
    try:
        from datasets import load_dataset

        ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            for row in ds:
                norm = _normalize(row)
                if norm is None:
                    continue
                rendered_len = sum(len(m["content"]) for m in norm["messages"])
                if rendered_len > args.max_chars:
                    continue
                handle.write(json.dumps(norm, ensure_ascii=False) + "\n")
                written += 1
                if written >= args.n:
                    break
    except Exception as exc:  # pragma: no cover - environment-specific
        print(f"[synth_smoltalk] dataset unavailable ({exc!r}); writing synthetic fallback.")
        with open(args.out, "w", encoding="utf-8") as handle:
            for _ in range((args.n + len(SYNTHETIC_FALLBACK) - 1) // len(SYNTHETIC_FALLBACK)):
                for row in SYNTHETIC_FALLBACK:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= args.n:
                        break
                if written >= args.n:
                    break

    print(f"[synth_smoltalk] wrote {written} conversations to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
