"""Generate a JAXChat markdown report card from run artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


METRIC_RE = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)\s*:\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _last_metrics(log_path: Path) -> dict:
    out = {}
    if not log_path.exists():
        return out
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "[METRICS" in line or "Timing summary" in line:
                for k, v in METRIC_RE.findall(line):
                    try:
                        out[k] = float(v)
                    except ValueError:
                        pass
    return out


def _core_summary(data: dict) -> str:
    core = data.get("core") if isinstance(data, dict) else None
    if not isinstance(core, dict):
        return "missing"
    acc = core.get("_mean_accuracy") or core.get("accuracy")
    n = core.get("_total_n")
    tasks = core.get("_task_count")
    if acc is None:
        return "missing"
    suffix = f" ({n} examples, {tasks} tasks)" if n and tasks else ""
    return f"{float(acc):.3f}{suffix}"


def _metric(data: dict, key: str) -> str:
    value = data.get(key)
    if isinstance(value, dict):
        if "accuracy" in value:
            return f"{float(value['accuracy']):.3f}"
        if "error" in value:
            return "error"
    return "missing"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    root = Path(args.run_root)
    base_eval = _load_json(root / "runs" / "base" / "base_eval.json")
    sft_eval = _load_json(root / "runs" / "sft" / "chat_eval.json")
    rl_eval = _load_json(root / "runs" / "rl" / "chat_eval.json")
    base_log = _last_metrics(root / "runs" / "base" / "train.log")
    sft_log = _last_metrics(root / "runs" / "sft" / "train.log")
    rl_log = _last_metrics(root / "runs" / "rl" / "train.log")

    out_path = Path(args.out) if args.out else root / "report.md"
    lines = [
        "# JAXChat Report Card",
        "",
        f"- Run root: `{root}`",
        f"- Base val BPB: `{base_eval.get('val_bpb', 'missing')}`",
        f"- Base CORE: `{_core_summary(base_eval)}`",
        "",
        "## Chat Metrics",
        "",
        "| Stage | CORE | GSM8K | MMLU | HumanEval |",
        "|---|---:|---:|---:|---:|",
        f"| SFT | {_core_summary(sft_eval)} | {_metric(sft_eval, 'gsm8k')} | {_metric(sft_eval, 'mmlu')} | {_metric(sft_eval, 'humaneval')} |",
        f"| RL | {_core_summary(rl_eval)} | {_metric(rl_eval, 'gsm8k')} | {_metric(rl_eval, 'mmlu')} | {_metric(rl_eval, 'humaneval')} |",
        "",
        "## Timing",
        "",
        "| Stage | train_loop_s | tok/sec | mfu |",
        "|---|---:|---:|---:|",
        f"| Base | {base_log.get('train_loop_s', 'missing')} | {base_log.get('aggregate_tok_s', 'missing')} | {base_log.get('mfu_proxy', 'missing')} |",
        f"| SFT | {sft_log.get('train_loop_s', 'missing')} | - | - |",
        f"| RL | {rl_log.get('train_loop_s', 'missing')} | - | - |",
        "",
        "## Badges",
        "",
        f"- Pretraining: {'complete' if base_eval else 'missing'}",
        f"- Chat eval: {'complete' if sft_eval or rl_eval else 'missing'}",
        f"- GRPO: {'complete' if rl_eval else 'not run'}",
    ]
    os.makedirs(out_path.parent, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
