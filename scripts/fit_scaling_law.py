"""Fit scaling laws to the PretrainingBench sweep.

Consumes ``scripts.pretrainingbench`` grid points + each run's ``base_eval.json``
(or ``train.log`` val_bpb history) and fits three scaling laws:

1. **Kaplan** (arXiv:2001.08361):  L(N, D) = E + A·N^{-α} + B·D^{-β}
   Joint non-linear least squares over all completed (N, D, L) points.

2. **Chinchilla** (arXiv:2203.15556):  group points into IsoFLOP budgets, fit a
   parabola in log10(N) per budget, read off N*(C) and D*(C), then fit power laws
   N* ∝ C^{a_N}, D* ∝ C^{a_D}  (Chinchilla predicts ≈0.50 for both).

3. **Muennighoff data-constrained** (arXiv:2305.16264):  when the corpus is
   repeated (D > pool), effective tokens decay.  We fit
       L(N, D) = E + A·N^{-α} + B·D_eff^{-β}
   with  D_eff = D_u + R_n·D_r,  D_u = min(D, pool),  D_r = max(D - pool, 0),
   R_n = 1 - (1 - D_r/D_u)^{n}  (n = floor(D/D_u) repeats; R_n→1 as n→∞ but
   saturates because repeat tokens lose value).  This is the analytic
   repetition-decay from §3 of the paper, reduced to a single scalar R_n per
   point so the joint fit stays well-posed.

Outputs:  ``fit.csv`` (all points + fits), ``scaling_laws.json`` (fitted
exponents + predictions), and four plots:
  - plot_kaplan.png        L vs N (one curve per token budget) + Kaplan fit
  - plot_chinchilla.png    IsoFLOP curves + N*(C), D*(C) power laws
  - plot_data_constrained.png  L vs effective tokens, coloured by repeat count
  - plot_loss_vs_flops.png L vs cumulative FLOPs (the "compute frontier")

Usage::

    uv run python -m scripts.fit_scaling_law \\
        --runs-root /project/inniang/jaxchat/data/pbench/runs \\
        --out-dir   /project/inniang/jaxchat/data/pbench/_fit

The script is read-only with respect to the runs; missing runs are skipped.
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

from scripts.pretrainingbench import (  # noqa: E402
    BenchPoint,
    enumerate_grid,
)

# Reuse the train.log parser from fit_chinchilla so both tools read the same
# [METRICS ...] val_bpb lines identically.
from scripts.fit_chinchilla import (  # noqa: E402
    _read_val_bpb_from_eval_json,
    _read_train_log_history,
    _last_val_bpb_from_history,
    _read_timing_metrics,
)


DATA_POOL_TOKENS = 2_940_000_000  # fineweb32k_real_29; override via --data-pool-tokens


def load_results(runs_root: str, grid: list[BenchPoint], data_pool_tokens: int) -> list[dict]:
    out: list[dict] = []
    for p in grid:
        run_dir = os.path.join(runs_root, p.run_name)
        val_bpb, core = _read_val_bpb_from_eval_json(run_dir)
        source = "base_eval.json"
        history = _read_train_log_history(run_dir)
        timing = _read_timing_metrics(run_dir)
        if val_bpb is None:
            val_bpb = _last_val_bpb_from_history(history)
            source = "train.log" if val_bpb is not None else "MISSING"
        d = p.actual_train_tokens
        d_u = min(d, data_pool_tokens)
        d_r = max(d - data_pool_tokens, 0)
        n_repeats = d // max(d_u, 1)
        # Muennighoff repetition-decay scalar R_n in [0, 1].
        if d_r <= 0 or d_u <= 0:
            r_n = 1.0
        else:
            ratio = min(d_r / d_u, 1.0)
            r_n = 1.0 - (1.0 - ratio) ** n_repeats
        d_eff = d_u + r_n * d_r
        out.append({
            "task_id": p.task_id,
            "depth": p.depth,
            "vocab": p.vocab,
            "params": p.params,
            "non_emb_params": p.non_emb_params,
            "actual_train_tokens": d,
            "unique_tokens": d_u,
            "repeat_tokens": d_r,
            "n_repeats": n_repeats,
            "r_n": r_n,
            "effective_tokens": d_eff,
            "actual_flops": p.actual_flops,
            "val_bpb": val_bpb,
            "source": source,
            "run_dir": run_dir,
            "history": history,
            "tokens_per_step": p.tokens_per_step,
            "train_time_hours": (timing.get("train_loop_s") / 3600.0) if timing.get("train_loop_s") else None,
        })
    return out


# ---------------------------------------------------------------------------
# Law fits
# ---------------------------------------------------------------------------

def _completed(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["val_bpb"] is not None and r["val_bpb"] > 0]


def fit_kaplan(rows: list[dict]) -> dict:
    """L(N, D) = E + A·N^{-α} + B·D^{-β}.  Joint fit via scipy curve_fit."""
    import numpy as np
    from scipy.optimize import curve_fit

    pts = _completed(rows)
    if len(pts) < 4:
        return {"status": "insufficient_points", "n_points": len(pts)}
    N = np.array([float(r["params"]) for r in pts])
    D = np.array([float(r["actual_train_tokens"]) for r in pts])
    L = np.array([float(r["val_bpb"]) for r in pts])

    def kaplan(X, E, A, alpha, B, beta):
        n, d = X
        return E + A * np.power(n, -alpha) + B * np.power(d, -beta)

    # Initial guess: E ~ min loss, A,B ~ 1, alpha,beta ~ 0.1 (Kaplan) / 0.34,0.28 (Chinchilla-ish).
    p0 = [float(L.min()) * 0.5, 1.0, 0.10, 1.0, 0.10]
    bounds = ([0.0, 0.0, 1e-3, 0.0, 1e-3], [10.0, 1e3, 2.0, 1e3, 2.0])
    try:
        popt, pcov = curve_fit(kaplan, (N, D), L, p0=p0, bounds=bounds, maxfev=20000)
        E, A, alpha, B, beta = popt
        perr = np.sqrt(np.diag(pcov))
        pred = kaplan((N, D), *popt)
        rmse = float(np.sqrt(np.mean((pred - L) ** 2)))
        return {
            "status": "ok",
            "n_points": len(pts),
            "E": float(E), "A": float(A), "alpha": float(alpha),
            "B": float(B), "beta": float(beta),
            "E_err": float(perr[0]), "A_err": float(perr[1]), "alpha_err": float(perr[2]),
            "B_err": float(perr[3]), "beta_err": float(perr[4]),
            "rmse": rmse,
        }
    except (RuntimeError, ValueError) as exc:
        return {"status": "fit_failed", "n_points": len(pts), "error": str(exc)}


def fit_chinchilla(rows: list[dict]) -> dict:
    """IsoFLOP parabolas → N*(C), D*(C) power laws.  Reuses the fit_chinchilla
    approach: group by FLOP budget (here the per-model actual_flops), fit a
    parabola in log10(N) per budget, read the minimum."""
    import numpy as np

    pts = _completed(rows)
    if len(pts) < 3:
        return {"status": "insufficient_points", "n_points": len(pts)}

    # Group by rounded FLOP budget (each depth is its own budget here — we have
    # one point per depth, so IsoFLOP curves need >=3 depths per budget).  For a
    # single-vocab sweep each depth has a unique FLOP, so we bin into log10(FLOPs)
    # buckets of width 0.5 to get >=3 points per bucket.
    log_flops = np.log10([float(r["actual_flops"]) for r in pts])
    bins = np.floor(log_flops * 2) / 2  # 0.5-wide bins
    by_budget: dict[float, list[dict]] = {}
    for b, r in zip(bins, pts):
        by_budget.setdefault(float(b), []).append(r)

    per_budget = []
    for b in sorted(by_budget):
        grp = sorted(by_budget[b], key=lambda r: r["params"])
        if len(grp) < 3:
            continue
        log_n = np.log10([float(r["params"]) for r in grp])
        vals = [float(r["val_bpb"]) for r in grp]
        a, bb, cc = np.polyfit(log_n, vals, 2)
        log_n_star = -bb / (2.0 * a) if abs(a) > 1e-12 else float("nan")
        n_star = 10.0 ** log_n_star
        c = 10.0 ** b
        d_star = c / (6.0 * n_star) if n_star > 0 else float("nan")
        loss_star = a * log_n_star ** 2 + bb * log_n_star + cc
        per_budget.append({
            "log10_flops": b, "flops": c, "n_runs": len(grp),
            "n_star": float(n_star), "d_star": float(d_star), "loss_at_min": float(loss_star),
        })

    out: dict = {"status": "ok" if per_budget else "no_buckets", "per_budget": per_budget,
                 "n_points": len(pts)}
    if len(per_budget) >= 2:
        cs = np.array([bd["flops"] for bd in per_budget])
        ns = np.array([bd["n_star"] for bd in per_budget])
        ds = np.array([bd["d_star"] for bd in per_budget])
        a_n, b_n = np.polyfit(np.log10(cs), np.log10(ns), 1)
        a_d, b_d = np.polyfit(np.log10(cs), np.log10(ds), 1)
        out["n_power_law"] = {"alpha": float(a_n), "log10_prefactor": float(b_n)}
        out["d_power_law"] = {"alpha": float(a_d), "log10_prefactor": float(b_d)}
    return out


def fit_data_constrained(rows: list[dict], data_pool_tokens: int) -> dict:
    """Muennighoff data-constrained: L(N, D) = E + A·N^{-α} + B·D_eff^{-β},
    D_eff = D_u + R_n·D_r.  Same form as Kaplan but with D replaced by D_eff,
    which down-weights repeated tokens."""
    import numpy as np
    from scipy.optimize import curve_fit

    pts = _completed(rows)
    if len(pts) < 4:
        return {"status": "insufficient_points", "n_points": len(pts)}
    N = np.array([float(r["params"]) for r in pts])
    D_eff = np.array([float(r["effective_tokens"]) for r in pts])
    L = np.array([float(r["val_bpb"]) for r in pts])

    def muennighoff(X, E, A, alpha, B, beta):
        n, d = X
        return E + A * np.power(n, -alpha) + B * np.power(d, -beta)

    p0 = [float(L.min()) * 0.5, 1.0, 0.10, 1.0, 0.10]
    bounds = ([0.0, 0.0, 1e-3, 0.0, 1e-3], [10.0, 1e3, 2.0, 1e3, 2.0])
    try:
        popt, pcov = curve_fit(muennighoff, (N, D_eff), L, p0=p0, bounds=bounds, maxfev=20000)
        E, A, alpha, B, beta = popt
        perr = np.sqrt(np.diag(pcov))
        pred = muennighoff((N, D_eff), *popt)
        rmse = float(np.sqrt(np.mean((pred - L) ** 2)))
        # How much do repeat tokens help vs treating D literally?  Compare to a
        # Kaplan fit on raw D to see if D_eff reduces the loss residual.
        return {
            "status": "ok",
            "n_points": len(pts),
            "data_pool_tokens": data_pool_tokens,
            "E": float(E), "A": float(A), "alpha": float(alpha),
            "B": float(B), "beta": float(beta),
            "E_err": float(perr[0]), "A_err": float(perr[1]), "alpha_err": float(perr[2]),
            "B_err": float(perr[3]), "beta_err": float(perr[4]),
            "rmse": rmse,
            "n_repeated_points": int(sum(1 for r in pts if r["repeat_tokens"] > 0)),
        }
    except (RuntimeError, ValueError) as exc:
        return {"status": "fit_failed", "n_points": len(pts), "error": str(exc)}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_kaplan(rows: list[dict], kaplan: dict, png_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pts = _completed(rows)
    if not pts:
        print("[plot] no completed points; skipping Kaplan figure.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter([r["params"] for r in pts], [r["val_bpb"] for r in pts],
                    c=[r["actual_train_tokens"] for r in pts], cmap="viridis", s=60,
                    edgecolors="black", linewidths=0.5, zorder=3)
    plt.colorbar(sc, label="train tokens")
    if kaplan.get("status") == "ok":
        N = np.logspace(math.log10(min(r["params"] for r in pts)) - 0.1,
                        math.log10(max(r["params"] for r in pts)) + 0.1, 80)
        D_fixed = float(np.median([r["actual_train_tokens"] for r in pts]))
        L = (kaplan["E"] + kaplan["A"] * N ** (-kaplan["alpha"])
             + kaplan["B"] * D_fixed ** (-kaplan["beta"]))
        ax.plot(N, L, "--", color="red", lw=1.5,
                label=f"Kaplan fit (D={D_fixed/1e9:.2f}B)\nα={kaplan['alpha']:.3f} β={kaplan['beta']:.3f}")
        ax.legend(fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (N)")
    ax.set_ylabel("Validation loss (bpb)")
    ax.set_title("Kaplan: L(N, D) = E + A/N^α + B/D^β")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {png_path}")


def plot_chinchilla(chinchilla: dict, png_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    per = chinchilla.get("per_budget", [])
    if len(per) < 2:
        print("[plot] not enough IsoFLOP buckets; skipping Chinchilla figure.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cs = np.array([bd["flops"] for bd in per])
    ns = np.array([bd["n_star"] for bd in per])
    ds = np.array([bd["d_star"] for bd in per])
    axes[0].scatter(cs, ns, c="tab:blue", s=60, zorder=3)
    if "n_power_law" in chinchilla:
        a = chinchilla["n_power_law"]["alpha"]
        b = chinchilla["n_power_law"]["log10_prefactor"]
        cl = np.logspace(math.log10(cs.min()) - 0.2, math.log10(cs.max()) + 0.2, 50)
        axes[0].plot(cl, 10 ** (a * np.log10(cl) + b), "--", color="red",
                     label=f"N* ∝ C^{a:.2f}  (Chinchilla=0.50)")
        axes[0].legend(fontsize=8)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("FLOPs (C)"); axes[0].set_ylabel("Optimal params N*")
    axes[0].set_title("Chinchilla N*(C)"); axes[0].grid(True, which="both", alpha=0.3)

    axes[1].scatter(cs, ds, c="tab:orange", s=60, zorder=3)
    if "d_power_law" in chinchilla:
        a = chinchilla["d_power_law"]["alpha"]
        b = chinchilla["d_power_law"]["log10_prefactor"]
        cl = np.logspace(math.log10(cs.min()) - 0.2, math.log10(cs.max()) + 0.2, 50)
        axes[1].plot(cl, 10 ** (a * np.log10(cl) + b), "--", color="red",
                     label=f"D* ∝ C^{a:.2f}  (Chinchilla=0.50)")
        axes[1].legend(fontsize=8)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("FLOPs (C)"); axes[1].set_ylabel("Optimal tokens D*")
    axes[1].set_title("Chinchilla D*(C)"); axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {png_path}")


def plot_data_constrained(rows: list[dict], mue: dict, png_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pts = _completed(rows)
    if not pts:
        print("[plot] no completed points; skipping data-constrained figure.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    repeats = [r["n_repeats"] for r in pts]
    sc = ax.scatter([r["effective_tokens"] for r in pts], [r["val_bpb"] for r in pts],
                    c=repeats, cmap="plasma", s=60, edgecolors="black", linewidths=0.5, zorder=3)
    plt.colorbar(sc, label="epochs over pool (D/D_u)")
    ax.set_xscale("log")
    ax.set_xlabel("Effective tokens D_eff = D_u + R_n·D_r")
    ax.set_ylabel("Validation loss (bpb)")
    title = "Muennighoff data-constrained"
    if mue.get("status") == "ok":
        title += f"  (α={mue['alpha']:.3f}, β={mue['beta']:.3f}, rmse={mue['rmse']:.4f})"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {png_path}")


def plot_loss_vs_flops(rows: list[dict], png_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pts = _completed(rows)
    if not pts:
        print("[plot] no completed points; skipping loss-vs-FLOPs figure.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([r["actual_flops"] for r in pts], [r["val_bpb"] for r in pts],
               c=[r["params"] for r in pts], cmap="viridis", s=60,
               edgecolors="black", linewidths=0.5, zorder=3)
    for r in pts:
        ax.annotate(f"d{r['depth']}", (r["actual_flops"], r["val_bpb"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=7, color="dimgray")
    ax.set_xscale("log")
    ax.set_xlabel("Training FLOPs (C = 6·N·D)")
    ax.set_ylabel("Validation loss (bpb)")
    ax.set_title("Compute frontier: loss vs FLOPs")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {png_path}")


def write_csv(rows: list[dict], path: str) -> None:
    fields = ["task_id", "depth", "vocab", "params", "non_emb_params",
              "actual_train_tokens", "unique_tokens", "repeat_tokens", "n_repeats",
              "r_n", "effective_tokens", "actual_flops", "val_bpb", "source",
              "tokens_per_step", "train_time_hours"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _print_summary(rows: list[dict], kaplan: dict, chinchilla: dict, mue: dict) -> None:
    done = _completed(rows)
    print(f"\n=== PretrainingBench results: {len(done)}/{len(rows)} runs complete ===\n")
    print(f"{'task':>4}  {'depth':>5}  {'params':>10}  {'tokens':>10}  "
          f"{'epochs':>5}  {'D_eff':>10}  {'val_bpb':>9}  {'source':<14}")
    for r in rows:
        v = f"{r['val_bpb']:.4f}" if r["val_bpb"] is not None else "—"
        print(f"  {r['task_id']:>4}  {r['depth']:>5}  {r['params']/1e6:9.2f}M  "
              f"{r['actual_train_tokens']/1e6:9.1f}M  {r['n_repeats']:>5}  "
              f"{r['effective_tokens']/1e6:9.1f}M  {v:>9}  {r['source']:<14}")
    print("\n=== Kaplan ===")
    if kaplan.get("status") == "ok":
        print(f"  L = E + A/N^α + B/D^β")
        print(f"  E={kaplan['E']:.4f}  A={kaplan['A']:.4f}  α={kaplan['alpha']:.3f}±{kaplan['alpha_err']:.3f}  "
              f"B={kaplan['B']:.4f}  β={kaplan['beta']:.3f}±{kaplan['beta_err']:.3f}  rmse={kaplan['rmse']:.4f}")
        print(f"  (Kaplan 2020: α≈0.076, β≈0.095;  Chinchilla 2022: α≈0.34, β≈0.28)")
    else:
        print(f"  {kaplan}")
    print("\n=== Chinchilla (IsoFLOP) ===")
    if chinchilla.get("per_budget"):
        for bd in chinchilla["per_budget"]:
            print(f"  C={bd['flops']:.2e}  N*={bd['n_star']:.3e}  D*={bd['d_star']:.3e}  loss*={bd['loss_at_min']:.4f}")
        if "n_power_law" in chinchilla:
            print(f"  N* ∝ C^{chinchilla['n_power_law']['alpha']:.3f}  (Chinchilla=0.50)")
            print(f"  D* ∝ C^{chinchilla['d_power_law']['alpha']:.3f}  (Chinchilla=0.50)")
    else:
        print(f"  {chinchilla}")
    print("\n=== Muennighoff data-constrained ===")
    if mue.get("status") == "ok":
        print(f"  L = E + A/N^α + B/D_eff^β  (D_eff = D_u + R_n·D_r)")
        print(f"  E={mue['E']:.4f}  A={mue['A']:.4f}  α={mue['alpha']:.3f}  B={mue['B']:.4f}  β={mue['beta']:.3f}  "
              f"rmse={mue['rmse']:.4f}  repeated_points={mue['n_repeated_points']}")
    else:
        print(f"  {mue}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit Kaplan/Chinchilla/Muennighoff scaling laws.")
    parser.add_argument("--runs-root", default="/project/inniang/jaxchat/data/pbench/runs")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--vocab", type=int, default=32768)
    parser.add_argument("--depths", default="2-40:2")
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--data-pool-tokens", type=int, default=DATA_POOL_TOKENS)
    parser.add_argument("--chinchilla-ratio", type=float, default=20.0)
    args = parser.parse_args(argv)

    from scripts.pretrainingbench import _parse_depths
    depths = _parse_depths(args.depths)
    if args.max_depth is not None:
        depths = tuple(d for d in depths if d <= args.max_depth)
    grid = enumerate_grid(depths=depths, vocab=args.vocab,
                          data_pool_tokens=args.data_pool_tokens,
                          chinchilla_ratio=args.chinchilla_ratio)
    rows = load_results(args.runs_root, grid, args.data_pool_tokens)
    kaplan = fit_kaplan(rows)
    chinchilla = fit_chinchilla(rows)
    mue = fit_data_constrained(rows, args.data_pool_tokens)

    out_dir = args.out_dir or os.path.join(args.runs_root, "_fit")
    os.makedirs(out_dir, exist_ok=True)
    write_csv(rows, os.path.join(out_dir, "fit.csv"))
    with open(os.path.join(out_dir, "scaling_laws.json"), "w", encoding="utf-8") as f:
        json.dump({"kaplan": kaplan, "chinchilla": chinchilla,
                   "muennighoff": mue, "n_points": len(_completed(rows)),
                   "data_pool_tokens": args.data_pool_tokens}, f, indent=2, sort_keys=True)
        f.write("\n")
    plot_kaplan(rows, kaplan, os.path.join(out_dir, "plot_kaplan.png"))
    plot_chinchilla(chinchilla, os.path.join(out_dir, "plot_chinchilla.png"))
    plot_data_constrained(rows, mue, os.path.join(out_dir, "plot_data_constrained.png"))
    plot_loss_vs_flops(rows, os.path.join(out_dir, "plot_loss_vs_flops.png"))
    _print_summary(rows, kaplan, chinchilla, mue)
    print(f"\n[csv] {os.path.join(out_dir, 'fit.csv')}")
    print(f"[json] {os.path.join(out_dir, 'scaling_laws.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
