"""Small HumanEval harness for chat checkpoints."""

from __future__ import annotations

import contextlib
import io
import json
import multiprocessing as mp
import os
import textwrap
import time


def _load_rows(n_max: int):
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/openai_humaneval", split="test")
        rows = []
        for row in ds:
            rows.append(row)
            if len(rows) >= n_max:
                break
        return rows
    except Exception:
        local_jsonl = "/project/inniang/hf-cache/humaneval/HumanEval.jsonl"
        if not os.path.exists(local_jsonl):
            raise
        rows = []
        with open(local_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                if len(rows) >= n_max:
                    break
        return rows


def _run_candidate(code: str, check_code: str, entry_point: str, timeout_s: float, queue) -> None:
    ns = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(code, ns, ns)
            exec(check_code, ns, ns)
            ns["check"](ns[entry_point])
        queue.put(True)
    except Exception:
        queue.put(False)


def _passes(code: str, check_code: str, entry_point: str, timeout_s: float = 3.0) -> bool:
    queue = mp.Queue()
    proc = mp.Process(target=_run_candidate, args=(code, check_code, entry_point, timeout_s, queue))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(1.0)
        return False
    return bool(queue.get()) if not queue.empty() else False


def _extract_code(text: str) -> str:
    if "```" not in text:
        return text
    parts = text.split("```")
    for part in parts:
        body = part
        if body.lstrip().startswith("python"):
            body = body.lstrip()[len("python"):]
        if "def " in body:
            return body.strip()
    return text


def evaluate(engine, *, n_max: int = 20, max_new_tokens: int = 384) -> dict:
    rows = _load_rows(n_max)
    correct = 0
    started = time.perf_counter()
    for i, row in enumerate(rows):
        prompt = (
            "Complete the following Python function. Return only code.\n\n"
            + row["prompt"]
        )
        completion = engine.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.2, top_k=50, seed=1234 + i)
        code = row["prompt"] + "\n" + textwrap.dedent(_extract_code(completion))
        if _passes(code, row["test"], row["entry_point"]):
            correct += 1
    total = len(rows)
    acc = correct / total if total else 0.0
    return {"n": total, "correct": correct, "accuracy": acc, "elapsed_s": time.perf_counter() - started}
