"""GSM8K loader, prompt builder, answer parser, and reward function."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

GSM8K_SYSTEM = (
    "You are a helpful assistant. Solve the math problem step by step and "
    "return the final number inside \\boxed{...}."
)

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")


@dataclass
class GSM8KExample:
    question: str
    answer_text: str
    gold_value: float


def _gold_from_solution(answer: str) -> float | None:
    if "####" in answer:
        tail = answer.rsplit("####", 1)[1]
    else:
        tail = answer
    matches = _NUMBER_RE.findall(tail.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def load_examples(split: str = "test", n_max: int = 200) -> list[GSM8KExample]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    out: list[GSM8KExample] = []
    for row in ds:
        gold = _gold_from_solution(row["answer"])
        if gold is None:
            continue
        out.append(GSM8KExample(question=row["question"], answer_text=row["answer"], gold_value=gold))
        if len(out) >= n_max:
            break
    return out


def build_prompt(question: str) -> list[dict]:
    return [
        {"role": "system", "content": GSM8K_SYSTEM},
        {"role": "user", "content": question},
    ]


def parse_boxed(text: str) -> float | None:
    matches = _BOXED_RE.findall(text)
    if matches:
        candidate = matches[-1].replace(",", "").strip()
        try:
            return float(candidate)
        except ValueError:
            pass
    nums = _NUMBER_RE.findall(text.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def reward(answer_text: str, gold: float, *, tol: float = 1e-4) -> float:
    pred = parse_boxed(answer_text)
    if pred is None:
        return 0.0
    return 1.0 if abs(pred - gold) <= tol else 0.0


def evaluate(engine, *, n_max: int = 100, max_new_tokens: int = 256, temperature: float = 0.0, seed: int = 0) -> dict:
    examples = load_examples(split="test", n_max=n_max)
    correct = 0
    total = 0
    rewards: list[float] = []
    for ex in examples:
        text = engine.chat(build_prompt(ex.question), max_new_tokens=max_new_tokens, temperature=temperature, top_k=None, top_p=None, seed=seed)
        r = reward(text, ex.gold_value)
        rewards.append(r)
        total += 1
        correct += int(r > 0.5)
    return {
        "n": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
    }


__all__ = [
    "GSM8KExample",
    "load_examples",
    "build_prompt",
    "parse_boxed",
    "reward",
    "evaluate",
]
