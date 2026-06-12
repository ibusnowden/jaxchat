"""Fit and plot the Chinchilla IsoFLOP sweep results.

Reads val_bpb from each run's ``base_eval.json`` (falls back to the last
``val_bpb = X`` line in ``log.txt``), groups by FLOP budget, fits a quadratic
in log10(N) per budget to find the compute-optimal (N*, D*), then fits log-log
power laws N* ∝ C^a and D* ∝ C^b.  Saves a 3-panel figure mirroring screenshot 1
plus a results CSV.

Usage::

    uv run python -m scripts.fit_chinchilla \\
        --runs-root /project/inniang/jaxchat/data/124m_rtx_run/runs/chinchilla \\
        --out-dir   /project/inniang/jaxchat/data/124m_rtx_run/runs/chinchilla/_fit

The script is read-only with respect to the runs themselves — safe to re-run
while the sweep is still in flight; missing runs are simply skipped (and
reported).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jaxchat-matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.chinchilla_grid import (  # noqa: E402
    DATA_POOL_TOKENS,
    GridPoint,
    WANDB_GROUP_ISOFLOP,
    WANDB_GROUP_MINISERIES,
    enumerate_grid,
)


# train.log lines look like:
#   [METRICS (1 step stale)] step: 50  |  val_bpb: 1.234  |  loss: ...
# (Logger.log in jaxchat/model.py:176.)
METRICS_LINE_RE = re.compile(r"\[METRICS[^\]]*\]\s*(.*)")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)\s*:\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")


def _read_val_bpb_from_eval_json(run_dir: str) -> tuple[Optional[float], Optional[dict]]:
    path = os.path.join(run_dir, "base_eval.json")
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = data.get("val_bpb")
        core = data.get("core")
        return (float(v) if v is not None else None, core)
    except (OSError, json.JSONDecodeError, ValueError):
        return None, None


def _read_train_log_history(run_dir: str) -> list[dict]:
    """Parse train.log and return a list of {step, ...} dicts for each [METRICS] line."""
    candidates = [
        os.path.join(run_dir, "train.log"),
        os.path.join(run_dir, "base", "train.log"),
    ]
    rows: list[dict] = []
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = METRICS_LINE_RE.search(line)
                    if not m:
                        continue
                    body = m.group(1)
                    pairs = dict(KV_RE.findall(body))
                    if "step" in pairs:
                        try:
                            row = {"step": int(float(pairs["step"]))}
                        except ValueError:
                            continue
                        for k, v in pairs.items():
                            if k == "step":
                                continue
                            try:
                                row[k] = float(v)
                            except ValueError:
                                pass
                        rows.append(row)
            if rows:
                return rows
        except OSError:
            continue
    return rows


def _read_timing_metrics(run_dir: str) -> dict:
    """Return final timing metrics from train.log, if present."""
    candidates = [
        os.path.join(run_dir, "train.log"),
        os.path.join(run_dir, "base", "train.log"),
    ]
    timing_keys = {
        "compile_s",
        "data_open_s",
        "train_loop_s",
        "eval_s_total",
        "checkpoint_s",
        "total_wall_s",
        "steady_state_step_s",
        "aggregate_tok_s",
        "mfu_proxy",
    }
    latest: dict = {}
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if "train_loop_s" not in line and "total_wall_s" not in line:
                        continue
                    pairs = dict(KV_RE.findall(line))
                    parsed = {}
                    for key in timing_keys:
                        if key not in pairs:
                            continue
                        try:
                            parsed[key] = float(pairs[key])
                        except ValueError:
                            pass
                    if parsed:
                        latest = parsed
        except OSError:
            continue
    return latest


def _last_val_bpb_from_history(rows: list[dict]) -> Optional[float]:
    for row in reversed(rows):
        if "val_bpb" in row:
            return float(row["val_bpb"])
    return None


def _core_metrics(core: object) -> dict:
    if not isinstance(core, dict):
        return {"accuracy": None, "stderr": None, "n": None, "task_count": None}

    tasks = core.get("tasks") if isinstance(core.get("tasks"), dict) else core
    scored = []
    for value in tasks.values():
        if not isinstance(value, dict) or "accuracy" not in value:
            continue
        try:
            acc = float(value["accuracy"])
            n = int(value.get("n", 0))
            stderr = value.get("stderr")
            if stderr is None:
                stderr = math.sqrt(acc * (1.0 - acc) / n) if n > 0 else None
            scored.append({"accuracy": acc, "n": n, "stderr": stderr})
        except (TypeError, ValueError):
            pass

    accuracy = None
    for key in ("_mean_accuracy", "accuracy"):
        if key in core:
            try:
                accuracy = float(core[key])
                break
            except (TypeError, ValueError):
                accuracy = None
    if accuracy is None and scored:
        accuracy = sum(v["accuracy"] for v in scored) / len(scored)

    stderr = None
    if "_mean_stderr" in core:
        try:
            stderr = float(core["_mean_stderr"])
        except (TypeError, ValueError):
            stderr = None
    if stderr is None and scored:
        stderr = math.sqrt(sum(float(v["stderr"] or 0.0) ** 2 for v in scored)) / len(scored)

    total_n = None
    if "_total_n" in core:
        try:
            total_n = int(core["_total_n"])
        except (TypeError, ValueError):
            total_n = None
    if total_n is None and scored:
        total_n = sum(v["n"] for v in scored)

    task_count = None
    if "_task_count" in core:
        try:
            task_count = int(core["_task_count"])
        except (TypeError, ValueError):
            task_count = None
    if task_count is None and scored:
        task_count = len(scored)

    return {"accuracy": accuracy, "stderr": stderr, "n": total_n, "task_count": task_count}


def load_results(runs_root: str, grid: list[GridPoint]) -> list[dict]:
    out: list[dict] = []
    for p in grid:
        run_dir = os.path.join(runs_root, p.run_name)
        val_bpb, core = _read_val_bpb_from_eval_json(run_dir)
        source = "base_eval.json"
        history = _read_train_log_history(run_dir)
        timing = _read_timing_metrics(run_dir)
        train_loop_s = timing.get("train_loop_s")
        total_wall_s = timing.get("total_wall_s")
        if val_bpb is None:
            val_bpb = _last_val_bpb_from_history(history)
            source = "train.log" if val_bpb is not None else "MISSING"
        # Extract CORE accuracy plus uncertainty if base_eval.json includes it.
        core_metrics = _core_metrics(core)
        out.append({
            "task_id": p.task_id,
            "kind": p.kind,
            "run_name": p.run_name,
            "flop_budget": p.flop_budget,
            "depth": p.depth,
            "params": p.params,
            "actual_train_tokens": p.actual_train_tokens,
            "actual_flops": p.actual_flops,
            "val_bpb": val_bpb,
            "core_acc": core_metrics["accuracy"],
            "core_stderr": core_metrics["stderr"],
            "core_n": core_metrics["n"],
            "core_task_count": core_metrics["task_count"],
            "source": source,
            "run_dir": run_dir,
            "history": history,  # for Plot 3 (loss vs cumulative FLOPs)
            "tokens_per_step": p.tokens_per_step,
            "train_time_hours": (train_loop_s / 3600.0) if train_loop_s is not None else None,
            "total_wall_hours": (total_wall_s / 3600.0) if total_wall_s is not None else None,
            "steady_state_step_s": timing.get("steady_state_step_s"),
            "aggregate_tok_s": timing.get("aggregate_tok_s"),
        })
    return out


def _fit_parabola(log_n: list[float], vals: list[float]) -> tuple[float, float, float, float]:
    """Quadratic fit y = a*x^2 + b*x + c.  Returns (a, b, c, x_min)."""
    n = len(log_n)
    if n < 3:
        raise ValueError(f"Need >=3 points for parabolic fit, got {n}.")
    # Normal equations via small NumPy-free closed form (avoids hard NumPy dep).
    import numpy as np
    x = np.asarray(log_n, dtype=float)
    y = np.asarray(vals, dtype=float)
    a, b, c = np.polyfit(x, y, 2)
    if abs(a) < 1e-12:
        x_min = float("nan")
    else:
        x_min = float(-b / (2.0 * a))
    return float(a), float(b), float(c), x_min


def _power_law_fit(c_vals: list[float], y_vals: list[float]) -> tuple[float, float]:
    """log10(y) = alpha * log10(C) + beta.  Returns (alpha, beta)."""
    import numpy as np
    log_c = np.log10(np.asarray(c_vals, dtype=float))
    log_y = np.log10(np.asarray(y_vals, dtype=float))
    alpha, beta = np.polyfit(log_c, log_y, 1)
    return float(alpha), float(beta)


def _flops_label(c: float) -> str:
    """Compact scientific notation without rounding 1.5e18 to 2e18."""
    if c <= 0:
        return "0"
    exp = int(math.floor(math.log10(c)))
    mant = c / (10 ** exp)
    mant_s = f"{mant:.2g}"
    if mant_s == "1":
        return f"1e{exp}"
    return f"{mant_s}e{exp}"


def fit_chinchilla(results: list[dict]) -> dict:
    """Group IsoFLOP runs by FLOP budget, fit parabolas, then fit cross-budget power laws."""
    # Group complete runs by FLOP budget (IsoFLOP only — miniseries are off-isoline).
    by_budget: dict[float, list[dict]] = {}
    for r in results:
        if r["kind"] != "isoflop":
            continue
        if r["val_bpb"] is None:
            continue
        by_budget.setdefault(float(r["flop_budget"]), []).append(r)

    per_budget: list[dict] = []
    for c in sorted(by_budget):
        rows = sorted(by_budget[c], key=lambda r: r["params"])
        if len(rows) < 3:
            print(f"[warn] budget C={c:.2e} has only {len(rows)} runs; skipping parabolic fit.")
            continue
        log_n = [math.log10(r["params"]) for r in rows]
        vals = [r["val_bpb"] for r in rows]
        a, b, cc, log_n_star = _fit_parabola(log_n, vals)
        n_star = 10.0 ** log_n_star
        d_star = c / (6.0 * n_star)
        loss_star = a * log_n_star ** 2 + b * log_n_star + cc
        per_budget.append({
            "flop_budget": c,
            "n_runs": len(rows),
            "fit_a": a,
            "fit_b": b,
            "fit_c": cc,
            "n_star": n_star,
            "d_star": d_star,
            "loss_at_min": loss_star,
            "rows": rows,
        })

    out: dict = {"per_budget": per_budget}
    if len(per_budget) >= 2:
        cs = [bd["flop_budget"] for bd in per_budget]
        n_stars = [bd["n_star"] for bd in per_budget]
        d_stars = [bd["d_star"] for bd in per_budget]
        n_alpha, n_beta = _power_law_fit(cs, n_stars)
        d_alpha, d_beta = _power_law_fit(cs, d_stars)
        out["n_power_law"] = {"alpha": n_alpha, "log10_prefactor": n_beta}
        out["d_power_law"] = {"alpha": d_alpha, "log10_prefactor": d_beta}
    return out


def write_csv(results: list[dict], path: str) -> None:
    fields = ["task_id", "kind", "run_name", "flop_budget", "depth", "params",
              "actual_train_tokens", "actual_flops", "val_bpb", "core_acc",
              "core_stderr", "core_n", "core_task_count", "train_time_hours",
              "total_wall_hours", "steady_state_step_s", "aggregate_tok_s", "source"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in fields})


def plot_isoflop(fit: dict, png_path: str) -> None:
    """Plot 1 (and small-scale Plot 4): 3-panel IsoFLOP + power laws."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not fit.get("per_budget"):
        print("[plot] no fitted budgets; skipping IsoFLOP figure.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cs_sorted = [bd["flop_budget"] for bd in fit["per_budget"]]
    cmap = plt.get_cmap("viridis")
    colors = {c: cmap(i / max(1, len(cs_sorted) - 1)) for i, c in enumerate(cs_sorted)}

    # Panel 1: IsoFLOP curves.
    ax = axes[0]
    for bd in fit["per_budget"]:
        c = bd["flop_budget"]
        col = colors[c]
        rows = bd["rows"]
        ns = np.array([r["params"] for r in rows])
        vs = np.array([r["val_bpb"] for r in rows])
        ax.plot(ns, vs, "o", color=col, label=_flops_label(c))
        log_n_range = np.linspace(math.log10(ns.min()) - 0.05, math.log10(ns.max()) + 0.05, 100)
        ys = bd["fit_a"] * log_n_range ** 2 + bd["fit_b"] * log_n_range + bd["fit_c"]
        ax.plot(10 ** log_n_range, ys, "--", color=col, lw=1, alpha=0.7)
        ax.plot(bd["n_star"], bd["loss_at_min"], "*", color=col, markersize=14,
                markeredgecolor="black", markeredgewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Effective Parameters")
    ax.set_ylabel("Validation Loss (bpb)")
    ax.set_title("IsoFLOP Curves")
    ax.legend(title="FLOPs", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # Panel 2: N* vs C.
    ax = axes[1]
    cs = np.array([bd["flop_budget"] for bd in fit["per_budget"]])
    ns = np.array([bd["n_star"] for bd in fit["per_budget"]])
    cols = [colors[c] for c in cs]
    ax.scatter(cs, ns, c=cols, s=60, zorder=3)
    if "n_power_law" in fit:
        alpha = fit["n_power_law"]["alpha"]
        beta = fit["n_power_law"]["log10_prefactor"]
        cline = np.logspace(math.log10(cs.min()) - 0.2, math.log10(cs.max()) + 0.2, 50)
        ax.plot(cline, 10 ** (alpha * np.log10(cline) + beta), "--", color="red",
                label=f"N $\\propto$ C$^{{{alpha:.2f}}}$")
        ax.legend()
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("FLOPs"); ax.set_ylabel("Optimal Parameters")
    ax.set_title("Optimal Model Size")
    ax.grid(True, which="both", alpha=0.3)

    # Panel 3: D* vs C.
    ax = axes[2]
    ds = np.array([bd["d_star"] for bd in fit["per_budget"]])
    ax.scatter(cs, ds, c=cols, s=60, zorder=3)
    if "d_power_law" in fit:
        alpha = fit["d_power_law"]["alpha"]
        beta = fit["d_power_law"]["log10_prefactor"]
        cline = np.logspace(math.log10(cs.min()) - 0.2, math.log10(cs.max()) + 0.2, 50)
        ax.plot(cline, 10 ** (alpha * np.log10(cline) + beta), "--", color="red",
                label=f"D $\\propto$ C$^{{{alpha:.2f}}}$")
        ax.legend()
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("FLOPs"); ax.set_ylabel("Optimal Tokens")
    ax.set_title("Optimal Training Tokens")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"jaxchat IsoFLOP fit — {WANDB_GROUP_ISOFLOP}", y=1.02)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {png_path}")


def plot_miniseries_loss(results: list[dict], png_path: str) -> None:
    """Plot 3: val_bpb vs cumulative training FLOPs, one curve per miniseries depth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    series = [r for r in results if r["kind"] == "miniseries" and r["history"]]
    if not series:
        print("[plot] no miniseries history yet; skipping loss-vs-FLOPs figure.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("plasma")
    depths = sorted({r["depth"] for r in series})
    for i, depth in enumerate(depths):
        color = cmap(i / max(1, len(depths) - 1))
        for r in series:
            if r["depth"] != depth:
                continue
            steps_v = [row["step"] for row in r["history"] if "val_bpb" in row]
            vals = [row["val_bpb"] for row in r["history"] if "val_bpb" in row]
            if not steps_v:
                continue
            steps_v = np.asarray(steps_v)
            vals = np.asarray(vals)
            flops = 6.0 * r["params"] * (steps_v * r["tokens_per_step"])
            ax.plot(flops, vals, "-", color=color, lw=1.5, label=f"d{depth} ({r['params']/1e6:.0f}M)")
    ax.set_xscale("log")
    ax.set_xlabel("Total Training FLOPs")
    ax.set_ylabel("val_bpb")
    ax.set_title("Depth miniseries — loss vs training FLOPs")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {png_path}")


def plot_miniseries_core(results: list[dict], png_path: str) -> None:
    """Plot 2: final CORE score vs training FLOPs and elapsed training time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    miniseries_pts = sorted(
        [
            r for r in results
            if r["kind"] == "miniseries" and r["core_acc"] is not None and r["actual_flops"] > 0
        ],
        key=lambda r: r["actual_flops"],
    )
    if not miniseries_pts:
        print("[plot] no CORE scores yet; skipping CORE-vs-FLOPs figure.")
        return

    total_ns = sorted({int(p["core_n"]) for p in miniseries_pts if p.get("core_n")})
    task_counts = sorted({int(p["core_task_count"]) for p in miniseries_pts if p.get("core_task_count")})
    subtitle = ""
    if total_ns and task_counts:
        n_text = str(total_ns[0]) if len(total_ns) == 1 else f"{total_ns[0]}-{total_ns[-1]}"
        t_text = str(task_counts[0]) if len(task_counts) == 1 else f"{task_counts[0]}-{task_counts[-1]}"
        subtitle = f"{n_text} examples across {t_text} tasks per point"

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    color = "seagreen"

    def add_panel(ax, *, x_key: str, xlabel: str, title: str) -> None:
        panel_pts = [p for p in miniseries_pts if p.get(x_key) is not None and float(p[x_key]) > 0]
        if not panel_pts:
            ax.text(0.5, 0.5, "timing unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("CORE Score")
            return

        xs = np.asarray([float(p[x_key]) for p in panel_pts], dtype=float)
        ys = np.asarray([float(p["core_acc"]) for p in panel_pts], dtype=float)
        yerr = np.asarray(
            [float(p["core_stderr"] or 0.0) for p in panel_pts],
            dtype=float,
        )
        depths = [int(p["depth"]) for p in panel_pts]

        ax.plot(xs, ys, "-", color=color, lw=1.2, alpha=0.8, zorder=2)
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            fmt="o",
            color=color,
            ecolor="mediumseagreen",
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=8,
            elinewidth=1.0,
            capsize=3,
            label="JAXChat miniseries (d10-d20)",
            zorder=3,
        )
        for x, y, depth in zip(xs, ys, depths):
            ax.annotate(f"d{depth}", (x, y), xytext=(4, 3), textcoords="offset points",
                        fontsize=8, color="dimgray")

        fit_mask = np.asarray([depth >= 12 for depth in depths], dtype=bool)
        if fit_mask.sum() >= 2 and np.all(xs[fit_mask] > 0):
            slope, intercept = np.polyfit(np.log10(xs[fit_mask]), ys[fit_mask], 1)
            fit_x = np.logspace(
                math.log10(xs[fit_mask].min()) - 0.05,
                math.log10(xs[fit_mask].max()) + 0.25,
                100,
            )
            fit_y = slope * np.log10(fit_x) + intercept
            ax.plot(fit_x, fit_y, "-", color="forestgreen", lw=2.0,
                    label="Fit (d>=12)", zorder=1)

        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CORE Score")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")

    add_panel(
        axes[0],
        x_key="actual_flops",
        xlabel="Training FLOPs",
        title="CORE vs Training FLOPs",
    )
    add_panel(
        axes[1],
        x_key="train_time_hours",
        xlabel="Training Time (hours, 8xRTX6000)",
        title="CORE vs Training Time",
    )

    ys = [float(p["core_acc"]) for p in miniseries_pts]
    yerrs = [float(p["core_stderr"] or 0.0) for p in miniseries_pts]
    y_min = max(0.0, min(y - e for y, e in zip(ys, yerrs)) - 0.03)
    y_max = min(0.5, max(y + e for y, e in zip(ys, yerrs)) + 0.04)
    axes[0].set_ylim(y_min, y_max)
    if subtitle:
        fig.suptitle(f"JAXChat CORE scaling ({subtitle})", y=1.02)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {png_path}")


def _print_summary(results: list[dict], fit: dict) -> None:
    iso = [r for r in results if r["kind"] == "isoflop"]
    mini = [r for r in results if r["kind"] == "miniseries"]
    done_iso = sum(1 for r in iso if r["val_bpb"] is not None)
    done_mini = sum(1 for r in mini if r["val_bpb"] is not None)
    print(f"\n=== Chinchilla sweep results: "
          f"isoflop {done_iso}/{len(iso)},  miniseries {done_mini}/{len(mini)} ===\n")
    print(f"{'task':>4}  {'kind':<10}  {'run_name':<28}  {'val_bpb':>9}  "
          f"{'core':>6}  {'source':<14}")
    for r in results:
        v = f"{r['val_bpb']:.4f}" if r["val_bpb"] is not None else "—"
        c = f"{r['core_acc']:.3f}" if r["core_acc"] is not None else "—"
        print(f"  {r['task_id']:>4}  {r['kind']:<10}  {r['run_name']:<28}  "
              f"{v:>9}  {c:>6}  {r['source']:<14}")
    if not fit.get("per_budget"):
        print("\n(no IsoFLOP fits yet — need >=3 isoflop runs per budget)")
    else:
        print("\n=== Per-budget IsoFLOP fits ===")
        print(f"{'C':>10}  {'#runs':>5}  {'N*':>10}  {'D*':>10}  {'loss*':>7}")
        for bd in fit["per_budget"]:
            print(f"  {bd['flop_budget']:>10.2e}  {bd['n_runs']:>5}  "
                  f"{bd['n_star']:>10.3e}  {bd['d_star']:>10.3e}  {bd['loss_at_min']:>7.4f}")
        if "n_power_law" in fit:
            a = fit["n_power_law"]["alpha"]
            print(f"\nN* power law:  N ∝ C^{a:.3f}   (Chinchilla: 0.50, screenshot1: 0.54)")
        if "d_power_law" in fit:
            a = fit["d_power_law"]["alpha"]
            print(f"D* power law:  D ∝ C^{a:.3f}   (Chinchilla: 0.50, screenshot1: 0.49)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit and plot the Chinchilla sweep.")
    parser.add_argument("--runs-root",
                        default="/project/inniang/jaxchat/data/124m_rtx_run/runs/chinchilla")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write fit.csv + plots (default: <runs-root>/_fit).")
    args = parser.parse_args(argv)

    grid = enumerate_grid()
    results = load_results(args.runs_root, grid)
    fit = fit_chinchilla(results)

    out_dir = args.out_dir or os.path.join(args.runs_root, "_fit")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "fit.csv")
    write_csv(results, csv_path)
    print(f"[csv] wrote {csv_path}")
    plot_isoflop(fit, os.path.join(out_dir, "plot1_isoflop.png"))
    plot_miniseries_loss(results, os.path.join(out_dir, "plot3_miniseries_loss.png"))
    plot_miniseries_core(results, os.path.join(out_dir, "plot2_miniseries_core.png"))
    _print_summary(results, fit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
