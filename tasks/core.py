"""CORE-style ranked-classification eval harness.

A task contributes a list of examples ``(context, [continuations...], gold_idx)``.
We score each continuation by its conditional log-probability under the model
(via :meth:`jaxchat.engine.Engine.score_continuation`), then take ``argmax`` and
report top-1 accuracy.

This is a deliberately compact subset of the DCLM CORE benchmark that is fast
enough to run on a 10M-param model in a few minutes on a single H100.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from jaxchat.engine import Engine


@dataclass
class RankedExample:
    context: str
    continuations: list[str]
    gold_idx: int


def evaluate_task(engine: Engine, examples: Iterable[RankedExample], *, normalize_by_length: bool = True) -> dict:
    correct = 0
    total = 0
    for ex in examples:
        ctx_ids = list(engine.tokenizer.encode(ex.context))
        scores: list[float] = []
        for cont in ex.continuations:
            cont_ids = list(engine.tokenizer.encode(cont))
            if not cont_ids:
                scores.append(float("-inf"))
                continue
            logp = engine.score_continuation(ctx_ids, cont_ids)
            scores.append(logp / max(len(cont_ids), 1) if normalize_by_length else logp)
        if not scores:
            continue
        pred = max(range(len(scores)), key=scores.__getitem__)
        total += 1
        if pred == ex.gold_idx:
            correct += 1
    accuracy = (correct / total) if total else 0.0
    stderr = (accuracy * (1.0 - accuracy) / total) ** 0.5 if total else 0.0
    return {
        "n": total,
        "correct": correct,
        "accuracy": accuracy,
        "stderr": stderr,
    }


def run_subset(engine: Engine, *, n_per_task: int = 200) -> dict:
    """Run a small subset of CORE-style tasks.

    Each task loader is gated so a missing dataset (e.g. offline) skips that
    task instead of failing the whole eval.
    """

    from tasks import arc_challenge, arc_easy, hellaswag, piqa

    out: dict[str, dict] = {}
    for name, loader in (
        ("arc_easy", arc_easy.load_examples),
        ("arc_challenge", arc_challenge.load_examples),
        ("hellaswag", hellaswag.load_examples),
        ("piqa", piqa.load_examples),
    ):
        try:
            examples = loader(n_max=n_per_task)
        except Exception as exc:  # pragma: no cover - dataset availability is env-specific
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        out[name] = evaluate_task(engine, examples)
    if out:
        scored = [v for v in out.values() if isinstance(v, dict) and "accuracy" in v]
        accuracies = [v["accuracy"] for v in scored]
        if accuracies:
            out["_mean_accuracy"] = sum(accuracies) / len(accuracies)
            # Macro uncertainty for the task-average score.  This is an
            # approximation, but it is far more honest than plotting a clean
            # trend line through 100-example task subsets.
            out["_mean_stderr"] = (sum(v.get("stderr", 0.0) ** 2 for v in scored) ** 0.5) / len(scored)
            out["_task_count"] = len(scored)
            out["_total_n"] = sum(int(v.get("n", 0)) for v in scored)
            out["_total_correct"] = sum(int(v.get("correct", 0)) for v in scored)
            out["_micro_accuracy"] = (
                out["_total_correct"] / out["_total_n"] if out["_total_n"] else 0.0
            )
    return out


__all__ = ["RankedExample", "evaluate_task", "run_subset"]
