"""Create a compact midtraining mix: chat, MCQ, and Python tool-use traces."""

from __future__ import annotations

import argparse
import json
import os
import random


MCQ_ROWS = [
    {
        "question": "Which planet is known as the Red Planet?",
        "choices": ["Venus", "Mars", "Jupiter", "Mercury"],
        "answer": "Mars",
    },
    {
        "question": "What is the derivative of x^2?",
        "choices": ["x", "2x", "x^3", "2"],
        "answer": "2x",
    },
    {
        "question": "Which protocol is commonly used to serve web pages?",
        "choices": ["SMTP", "HTTP", "SSH", "NTP"],
        "answer": "HTTP",
    },
]


TOOL_ROWS = [
    {
        "messages": [
            {"role": "user", "content": "Use Python to calculate 37 * 42."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will calculate it directly.\n"},
                    {"type": "python", "text": "37 * 42"},
                    {"type": "python_output", "text": "1554"},
                    {"type": "text", "text": "\n37 * 42 = 1554."},
                ],
            },
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Use Python to sort these numbers: 9, 2, 5, 1."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will sort them with Python.\n"},
                    {"type": "python", "text": "sorted([9, 2, 5, 1])"},
                    {"type": "python_output", "text": "[1, 2, 5, 9]"},
                    {"type": "text", "text": "\nThe sorted numbers are 1, 2, 5, 9."},
                ],
            },
        ]
    },
]


def _mcq_to_chat(row: dict) -> dict:
    choices = "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(row["choices"]))
    return {
        "messages": [
            {"role": "user", "content": f"{row['question']}\n{choices}\nAnswer with the correct option and a brief reason."},
            {"role": "assistant", "content": f"The correct answer is {row['answer']}."},
        ]
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoltalk", default="")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    rows = []
    if args.smoltalk and os.path.exists(args.smoltalk):
        with open(args.smoltalk, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    synthetic = [_mcq_to_chat(row) for row in MCQ_ROWS] + TOOL_ROWS
    while len(rows) < args.n:
        rows.append(rng.choice(synthetic))
    rows = rows[: args.n]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} midtrain rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
