"""PIQA loader for the CORE-subset eval."""

from __future__ import annotations

import json
import os

from tasks.core import RankedExample


def load_examples(n_max: int = 200) -> list[RankedExample]:
    examples: list[RankedExample] = []

    local_dir = os.environ.get("JAXCHAT_PIQA_DIR", "/project/inniang/selfloop/data/piqa")
    local_data = os.path.join(local_dir, "valid.jsonl")
    local_labels = os.path.join(local_dir, "valid-labels.lst")
    if os.path.isfile(local_data) and os.path.isfile(local_labels):
        with open(local_data, "r", encoding="utf-8") as data_f, open(local_labels, "r", encoding="utf-8") as labels_f:
            for data_line, label_line in zip(data_f, labels_f):
                if not data_line.strip() or not label_line.strip():
                    continue
                row = json.loads(data_line)
                row["label"] = int(label_line.strip())
                examples.append(_row_to_example(row))
                if len(examples) >= n_max:
                    break
        return examples

    from datasets import load_dataset

    ds = load_dataset("ybisk/piqa", split="validation")
    for row in ds:
        examples.append(_row_to_example(row))
        if len(examples) >= n_max:
            break
    return examples


def _row_to_example(row) -> RankedExample:
    ctx = row["goal"]
    continuations = [" " + row["sol1"], " " + row["sol2"]]
    gold_idx = int(row["label"])
    return RankedExample(context=ctx, continuations=continuations, gold_idx=gold_idx)
