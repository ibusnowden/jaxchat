"""GSM8K loader, prompt builder, answer parser, and reward functions.

Two reward flavors live here:

* :func:`reward` — the strict 0/1 exact-match metric used for *evaluation*
  (``chat_eval``). Its semantics must stay fixed so the headline GSM8K number
  remains comparable across runs.
* :func:`shaped_reward` / :func:`reward_components` — a *training* reward for
  GRPO that adds partial credit for emitting a ``\\boxed{}`` answer and
  (optionally) for landing numerically close to the gold value. A pure 0/1
  reward gives a fresh 0.5B policy zero within-group variance (every rollout
  scores 0 → advantage 0 → no gradient); the shaping breaks that deadlock while
  keeping the *correctness* term identical to the eval metric.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Iterable

GSM8K_SYSTEM = (
    "You are a helpful assistant. Solve the math problem step by step and "
    "return the final number inside \\boxed{...}."
)

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)*")
# GSM8K calculator annotations like ``<<48/2=24>>`` — stripped for SFT targets.
_CALC_RE = re.compile(r"<<[^>]*>>")


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


def parse_boxed_strict(text: str) -> float | None:
    """Parse the last ``\\boxed{...}`` value, with *no* fallback to a bare
    trailing number. Used for the ``has_format`` shaping signal so we can tell a
    model that actually wrote the boxed format from one that merely happened to
    end on a number."""
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    candidate = matches[-1].replace(",", "").strip()
    try:
        return float(candidate)
    except ValueError:
        return None


def reward(answer_text: str, gold: float, *, tol: float = 1e-4) -> float:
    """Strict 0/1 exact-match — the evaluation metric. Do not change."""
    pred = parse_boxed(answer_text)
    if pred is None:
        return 0.0
    return 1.0 if abs(pred - gold) <= tol else 0.0


def reward_components(
    answer_text: str,
    gold: float,
    *,
    tol: float = 1e-4,
    format_bonus: float = 0.1,
    proximity_coef: float = 0.0,
) -> dict:
    """Decompose the GRPO training reward so each piece can be logged.

    Returned keys:

    * ``correct``    — 1.0 iff the parsed answer matches ``gold`` (this uses the
      same lenient :func:`parse_boxed` as :func:`reward`, so the correctness
      term the policy optimizes is *exactly* the eval metric).
    * ``has_format`` — 1.0 iff a well-formed ``\\boxed{<number>}`` is present.
    * ``proximity``  — in ``[0, 1)``, ``exp(-|pred-gold|/(|gold|+1))`` for
      incorrect-but-parseable answers, else 0. Only contributes when
      ``proximity_coef > 0``.
    * ``reward``     — ``correct + format_bonus*has_format + proximity_coef*proximity``.

    The format bonus is pure shaping *on top of* correctness: it nudges the
    policy toward the boxed answer format the SFT data teaches (which also makes
    parsing robust), without altering what "correct" means.
    """
    pred = parse_boxed(answer_text)
    correct = pred is not None and abs(pred - gold) <= tol
    has_format = parse_boxed_strict(answer_text) is not None

    proximity = 0.0
    if proximity_coef and pred is not None and not correct:
        proximity = math.exp(-abs(pred - gold) / (abs(gold) + 1.0))

    total = (
        (1.0 if correct else 0.0)
        + (format_bonus if has_format else 0.0)
        + proximity_coef * proximity
    )
    return {
        "reward": float(total),
        "correct": 1.0 if correct else 0.0,
        "has_format": 1.0 if has_format else 0.0,
        "proximity": float(proximity),
    }


def shaped_reward(
    answer_text: str,
    gold: float,
    *,
    tol: float = 1e-4,
    format_bonus: float = 0.1,
    proximity_coef: float = 0.0,
) -> float:
    """Scalar GRPO training reward (see :func:`reward_components`)."""
    return reward_components(
        answer_text,
        gold,
        tol=tol,
        format_bonus=format_bonus,
        proximity_coef=proximity_coef,
    )["reward"]


def make_reward_fn(
    mode: str = "shaped",
    *,
    tol: float = 1e-4,
    format_bonus: float = 0.1,
    proximity_coef: float = 0.0,
) -> Callable[[str, float], dict]:
    """Return ``fn(text, gold) -> components`` for the chosen reward ``mode``.

    ``mode="strict"`` reproduces the old 0/1 GRPO reward (``format_bonus`` and
    ``proximity_coef`` forced to 0); ``mode="shaped"`` applies the shaping
    knobs. Both return the full :func:`reward_components` dict so callers can log
    correctness / format rates regardless of mode.
    """
    if mode not in ("strict", "shaped"):
        raise ValueError(f"Unknown reward mode {mode!r} (expected 'strict' or 'shaped').")
    fb = 0.0 if mode == "strict" else format_bonus
    pc = 0.0 if mode == "strict" else proximity_coef

    def _fn(text: str, gold: float) -> dict:
        return reward_components(text, gold, tol=tol, format_bonus=fb, proximity_coef=pc)

    return _fn


def format_solution(answer: str) -> str:
    """Turn a raw GSM8K ``answer`` (CoT with ``<<calc>>`` markers and a
    ``#### N`` tail) into a clean assistant target that ends in
    ``The final answer is \\boxed{N}.`` — the format the reward parses and the
    GSM8K system prompt asks for."""
    body, _, tail = answer.partition("####")
    body = _CALC_RE.sub("", body).strip()
    final = tail.strip()
    if not final:
        # No #### marker; fall back to the gold value parsed from the body.
        gold = _gold_from_solution(answer)
        final = ("" if gold is None else (str(int(gold)) if gold == int(gold) else str(gold)))
    boxed = f"The final answer is \\boxed{{{final}}}." if final else "\\boxed{}"
    return f"{body}\n{boxed}".strip() if body else boxed


def build_sft_messages(question: str, answer: str) -> dict:
    """Canonical GSM8K → SFT row: ``{"messages": [system, user, assistant]}``
    matching :func:`build_prompt` (so SFT, RL, and eval share one prompt
    distribution) with a ``\\boxed{}``-terminated assistant solution."""
    return {
        "messages": build_prompt(question) + [
            {"role": "assistant", "content": format_solution(answer)},
        ]
    }


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
    "GSM8K_SYSTEM",
    "load_examples",
    "build_prompt",
    "parse_boxed",
    "parse_boxed_strict",
    "reward",
    "reward_components",
    "shaped_reward",
    "make_reward_fn",
    "format_solution",
    "build_sft_messages",
    "evaluate",
]
