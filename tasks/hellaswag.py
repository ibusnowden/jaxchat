"""HellaSwag loader for the CORE-subset eval (validation split)."""

from __future__ import annotations

from tasks.core import RankedExample


def load_examples(n_max: int = 200) -> list[RankedExample]:
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation")
    examples: list[RankedExample] = []
    for row in ds:
        ctx = (row["activity_label"] + ": " + row["ctx_a"] + " " + row["ctx_b"].capitalize()).strip()
        endings = row["endings"]
        gold_idx = int(row["label"])
        examples.append(
            RankedExample(
                context=ctx,
                continuations=[" " + e for e in endings],
                gold_idx=gold_idx,
            )
        )
        if len(examples) >= n_max:
            break
    return examples
