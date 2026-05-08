"""ARC-Easy loader for the CORE-subset eval."""

from __future__ import annotations

from tasks.core import RankedExample


def load_examples(n_max: int = 200) -> list[RankedExample]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
    examples: list[RankedExample] = []
    for row in ds:
        choices = row["choices"]["text"]
        labels = row["choices"]["label"]
        gold = row["answerKey"]
        if gold not in labels:
            continue
        gold_idx = labels.index(gold)
        ctx = f"Question: {row['question']}\nAnswer:"
        continuations = [f" {choice}" for choice in choices]
        examples.append(RankedExample(context=ctx, continuations=continuations, gold_idx=gold_idx))
        if len(examples) >= n_max:
            break
    return examples
