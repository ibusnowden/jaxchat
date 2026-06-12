"""MMLU multiple-choice eval harness."""

from __future__ import annotations

from tasks.core import RankedExample, evaluate_task


DEFAULT_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "college_computer_science",
    "high_school_world_history",
    "professional_law",
)


def _format_subject(subject: str) -> str:
    return subject.replace("_", " ")


def load_examples(n_max: int = 200, *, subjects: tuple[str, ...] = DEFAULT_SUBJECTS) -> list[RankedExample]:
    from datasets import load_dataset

    examples: list[RankedExample] = []
    per_subject = max(n_max // max(len(subjects), 1), 1)
    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split="test")
        count = 0
        for row in ds:
            choices = row["choices"]
            answer = int(row["answer"])
            examples.append(
                RankedExample(
                    context=f"The following is a multiple choice question about {_format_subject(subject)}.\n"
                            f"Question: {row['question']}\nAnswer:",
                    continuations=[f" {choice}" for choice in choices],
                    gold_idx=answer,
                )
            )
            count += 1
            if count >= per_subject or len(examples) >= n_max:
                break
        if len(examples) >= n_max:
            break
    return examples


def evaluate(engine, *, n_max: int = 200) -> dict:
    return evaluate_task(engine, load_examples(n_max=n_max))
