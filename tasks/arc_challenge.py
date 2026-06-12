"""ARC-Challenge loader for CORE-style ranked evaluation."""

from __future__ import annotations

from tasks.core import RankedExample


def load_examples(n_max: int = 200) -> list[RankedExample]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
    examples: list[RankedExample] = []
    for row in ds:
        choices = row["choices"]["text"]
        labels = row["choices"]["label"]
        gold = row["answerKey"]
        if gold not in labels:
            continue
        examples.append(
            RankedExample(
                context=f"Question: {row['question']}\nAnswer:",
                continuations=[f" {choice}" for choice in choices],
                gold_idx=labels.index(gold),
            )
        )
        if len(examples) >= n_max:
            break
    return examples
