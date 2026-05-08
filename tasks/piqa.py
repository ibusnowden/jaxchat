"""PIQA loader for the CORE-subset eval."""

from __future__ import annotations

from tasks.core import RankedExample


def load_examples(n_max: int = 200) -> list[RankedExample]:
    from datasets import load_dataset

    ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
    examples: list[RankedExample] = []
    for row in ds:
        ctx = row["goal"]
        continuations = [" " + row["sol1"], " " + row["sol2"]]
        gold_idx = int(row["label"])
        examples.append(RankedExample(context=ctx, continuations=continuations, gold_idx=gold_idx))
        if len(examples) >= n_max:
            break
    return examples
