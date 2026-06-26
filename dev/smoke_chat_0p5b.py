"""End-to-end smoke test for the 0.5B chat pipeline.

Loads the math-SFT and final GRPO checkpoints, runs a few chat turns, and
prints RAW GSM8K generations — to (a) prove the checkpoint -> engine -> chat
path works and (b) diagnose why frac_format stayed 0.0 for all RL steps
(does the model ever emit \\boxed{}? does it truncate mid-CoT?).

Run inside a GPU allocation:
    python dev/smoke_chat_0p5b.py
"""

from __future__ import annotations

import json
import os
import sys

PROJECT_ROOT = "/project/inniang/jaxchat"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jaxchat.model as model_lib  # noqa: E402

model_lib.configure_jax_runtime()

from jaxchat.engine import Engine  # noqa: E402
from tasks import gsm8k  # noqa: E402

TOKENIZER = os.path.join(PROJECT_ROOT, "data/124m_rtx_run/tokenizer/tokenizer.json")
E2E = os.path.join(PROJECT_ROOT, "data/0p5b_e2e")
GSM8K_DATA = os.path.join(E2E, "rl/gsm8k_train_4k.jsonl")

CHAT_PROMPTS = [
    "Hi! Who are you?",
    "Write a short poem about the ocean.",
    "What is the capital of France?",
]


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}", flush=True)


def run_stage(run_dir: str, stage: str) -> None:
    banner(f"LOADING {stage} @ {run_dir}")
    engine = Engine.from_run_dir(run_dir, stage=stage, tokenizer_path=TOKENIZER)
    print(f"loaded stage={engine.stage} step={engine.step}", flush=True)

    banner(f"[{stage}] CHAT SMOKE")
    for q in CHAT_PROMPTS:
        out = engine.chat(
            [{"role": "user", "content": q}],
            max_new_tokens=200, temperature=0.7, top_k=50, top_p=0.95, seed=0,
        )
        print(f"\n[user] {q}\n[model] {out}", flush=True)

    banner(f"[{stage}] GSM8K RAW GENERATIONS (max_new_tokens=512)")
    rows = []
    with open(GSM8K_DATA, "r", encoding="utf-8") as f:
        for _, line in zip(range(5), f):
            rows.append(json.loads(line))
    n_boxed = 0
    for i, r in enumerate(rows):
        out = engine.chat(
            gsm8k.build_prompt(r["question"]),
            max_new_tokens=512, temperature=0.2, top_k=50, top_p=0.95, seed=0,
        )
        has_boxed = "\\boxed" in out
        n_boxed += int(has_boxed)
        print(f"\n--- Q{i}: {r['question'][:100]}...")
        print(f"--- gold: {r['answer'][-80:]!r}")
        print(f"--- boxed_in_output: {has_boxed}  output_chars: {len(out)}")
        print(out, flush=True)
    print(f"\n[{stage}] boxed in {n_boxed}/5 generations", flush=True)


def main() -> int:
    run_stage(os.path.join(E2E, "runs/sft_math2"), "sft")
    run_stage(os.path.join(E2E, "runs/rl_math2"), "rl")
    banner("SMOKE TEST COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
