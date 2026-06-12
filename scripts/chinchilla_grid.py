"""Chinchilla-style IsoFLOP grid for the jaxchat 124m-modern family.

Replicates the small-scale compute-optimal sweep shown in the screenshots:
  * Several FLOP budgets C
  * Several model sizes (depths) per budget, bracketing the predicted Chinchilla
    optimum N* ≈ sqrt(C/120)
  * Token budget chosen so D = C / (6 * N), so each (C, depth) point sits on
    its target IsoFLOP curve.

This module is the single source of truth for the grid.  SLURM submission and
the fit/plot script both consume it.

CLI::

    python -m scripts.chinchilla_grid --print-grid         # human-readable table
    python -m scripts.chinchilla_grid --count              # total #tasks
    python -m scripts.chinchilla_grid --task-id 7          # KEY=VALUE for one task
    python -m scripts.chinchilla_grid --task-id 7 --shell  # eval-able assignments

The grid version is bumped whenever the (C, depth) points change so old W&B
groups stay frozen.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import shlex
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from jaxchat.model import Config, expected_parameter_breakdown  # noqa: E402
from jaxchat.presets import PRESETS  # noqa: E402


GRID_VERSION = "v2"
WANDB_GROUP_ISOFLOP = f"chinchilla-isoflop-{GRID_VERSION}"
WANDB_GROUP_MINISERIES = f"chinchilla-miniseries-{GRID_VERSION}"
WANDB_GROUP = WANDB_GROUP_ISOFLOP  # back-compat alias for fit_chinchilla
BASE_PRESET = "124m-modern"


# === IsoFLOP sweep (Screenshots 1 + 4 style) ===
# (FLOP budget, [depths]).  6 depths per budget bracket Chinchilla N* ≈ √(C/120):
#   C=1e17 -> N*≈29M    {4,6,8,10,12,14}   (params 41/67/99/139/189/251 M)
#   C=3e17 -> N*≈50M    {4,6,8,10,12,14}   (same depths, bigger D)
#   C=1e18 -> N*≈91M    {6,8,10,12,14,16}  (params 67/99/139/189/251/327 M)
#   C=1.5e18 -> N*≈112M {8,10,12,14,16,18} (params 99/139/189/251/327/419 M)
#
# The top budget is capped at 1.5e18 so every IsoFLOP point stays within one
# pass over the 2.94B-token FineWeb 32K-BPE pool while still bracketing its
# predicted optimum.
ISOFLOP_GRID: tuple[tuple[float, tuple[int, ...]], ...] = (
    (1e17, (4, 6, 8, 10, 12, 14)),
    (3e17, (4, 6, 8, 10, 12, 14)),
    (1e18, (6, 8, 10, 12, 14, 16)),
    (1.5e18, (8, 10, 12, 14, 16, 18)),
)

# === Depth miniseries (Screenshots 2 + 3 style) ===
# Each entry trains one depth to a FIXED token budget.  Matches the nanochat
# d10..d20 series; jaxchat's even-depth constraint (model.py:350) means we step
# every 2 depths, giving 6 lines instead of 11.  1.31B tokens matches the
# existing SOTA depth-16-long single point (val_bpb 0.7662) so the depth-16
# curve overlays cleanly on prior work.
MINISERIES_DEPTHS: tuple[int, ...] = (10, 12, 14, 16, 18, 20)
MINISERIES_TARGET_TOKENS: int = 1_310_720_000  # ~1.31B, matches existing SOTA budget

# Soft upper bound on tokens per run before we flag a row as "data-wall-bound".
DATA_POOL_TOKENS = 2_940_000_000


def skip_connections_for_depth(depth: int) -> tuple[tuple[int, int], ...]:
    """Keep only preset skip pairs that are valid for this depth."""
    base = PRESETS[BASE_PRESET]
    return tuple(
        (int(src), int(dst))
        for src, dst in base.skip_connections
        if 0 <= int(src) < int(dst) < depth
    )


def n_kv_heads_for_depth(depth: int) -> int:
    """Use 2 KV heads when it divides n_heads; otherwise fall back to MQA."""
    base = PRESETS[BASE_PRESET]
    cfg = dataclasses.replace(base, depth=depth)
    preferred = int(base.n_kv_heads)
    if preferred > 0 and cfg.n_heads % preferred == 0:
        return preferred
    return 1


def _build_config_for_depth(depth: int) -> Config:
    base = PRESETS[BASE_PRESET]
    return dataclasses.replace(
        base,
        depth=depth,
        target_train_tokens=0,
        n_train_iters=0,
        n_kv_heads=n_kv_heads_for_depth(depth),
        skip_connections=skip_connections_for_depth(depth),
    )


def params_for_depth(depth: int) -> int:
    cfg = _build_config_for_depth(depth)
    return int(expected_parameter_breakdown(cfg)["total"])


def tokens_per_step_for_depth(depth: int) -> int:
    cfg = _build_config_for_depth(depth)
    return int(cfg.tokens_per_step)


@dataclasses.dataclass(frozen=True)
class GridPoint:
    task_id: int
    kind: str  # "isoflop" or "miniseries"
    flop_budget: float  # nominal target for isoflop; 0.0 for miniseries (compute set by tokens)
    depth: int
    params: int
    tokens_per_step: int
    target_train_tokens: int
    n_train_iters: int
    actual_train_tokens: int
    actual_flops: float
    run_name: str
    wandb_group: str
    n_kv_heads: int
    skip_connections: tuple[tuple[int, int], ...]

    def shell_env(self) -> str:
        return (
            f"TASK_ID={self.task_id}\n"
            f"KIND={self.kind}\n"
            f"FLOP_BUDGET={self.flop_budget:.4e}\n"
            f"DEPTH={self.depth}\n"
            f"PARAMS={self.params}\n"
            f"TOKENS_PER_STEP={self.tokens_per_step}\n"
            f"TARGET_TRAIN_TOKENS={self.target_train_tokens}\n"
            f"N_TRAIN_ITERS={self.n_train_iters}\n"
            f"ACTUAL_TRAIN_TOKENS={self.actual_train_tokens}\n"
            f"ACTUAL_FLOPS={self.actual_flops:.4e}\n"
            f"RUN_NAME={shlex.quote(self.run_name)}\n"
            f"WANDB_GROUP_TASK={shlex.quote(self.wandb_group)}\n"
            f"N_KV_HEADS={self.n_kv_heads}\n"
            f"SKIP_CONNECTIONS={shlex.quote(repr(self.skip_connections))}\n"
        )


def _round_target_to_step(target_tokens: float, tokens_per_step: int) -> tuple[int, int, int]:
    n_iters = max(1, math.ceil(target_tokens / tokens_per_step))
    actual = n_iters * tokens_per_step
    return int(target_tokens), int(n_iters), int(actual)


def _format_run_name(flop_budget: float, depth: int) -> str:
    # e.g. chinchilla-C1e18-d10 (compact, sortable, W&B-safe)
    exp = int(round(math.log10(flop_budget)))
    mant = flop_budget / 10 ** exp
    if abs(mant - 1.0) < 1e-3:
        c_str = f"1e{exp}"
    elif abs(mant - round(mant)) < 1e-3:
        c_str = f"{int(round(mant))}e{exp}"
    else:
        c_str = f"{mant:.1f}e{exp}"
    return f"chinchilla-{GRID_VERSION}-C{c_str}-d{depth:02d}"


def enumerate_grid() -> list[GridPoint]:
    points: list[GridPoint] = []
    task_id = 0
    # --- IsoFLOP sweep ---
    for flop_budget, depths in ISOFLOP_GRID:
        for depth in depths:
            N = params_for_depth(depth)
            tps = tokens_per_step_for_depth(depth)
            target_tokens = flop_budget / (6.0 * N)
            target_int, n_iters, actual = _round_target_to_step(target_tokens, tps)
            actual_flops = 6.0 * N * actual
            run_name = _format_run_name(flop_budget, depth)
            n_kv_heads = n_kv_heads_for_depth(depth)
            skip_connections = skip_connections_for_depth(depth)
            points.append(
                GridPoint(
                    task_id=task_id,
                    kind="isoflop",
                    flop_budget=flop_budget,
                    depth=depth,
                    params=N,
                    tokens_per_step=tps,
                    target_train_tokens=target_int,
                    n_train_iters=n_iters,
                    actual_train_tokens=actual,
                    actual_flops=actual_flops,
                    run_name=run_name,
                    wandb_group=WANDB_GROUP_ISOFLOP,
                    n_kv_heads=n_kv_heads,
                    skip_connections=skip_connections,
                )
            )
            task_id += 1
    # --- Depth miniseries ---
    target_tokens = MINISERIES_TARGET_TOKENS
    for depth in MINISERIES_DEPTHS:
        N = params_for_depth(depth)
        tps = tokens_per_step_for_depth(depth)
        target_int, n_iters, actual = _round_target_to_step(target_tokens, tps)
        actual_flops = 6.0 * N * actual
        run_name = f"miniseries-{GRID_VERSION}-d{depth:02d}"
        n_kv_heads = n_kv_heads_for_depth(depth)
        skip_connections = skip_connections_for_depth(depth)
        points.append(
            GridPoint(
                task_id=task_id,
                kind="miniseries",
                flop_budget=0.0,  # nominal-only; not on a FLOP isoline
                depth=depth,
                params=N,
                tokens_per_step=tps,
                target_train_tokens=target_int,
                n_train_iters=n_iters,
                actual_train_tokens=actual,
                actual_flops=actual_flops,
                run_name=run_name,
                wandb_group=WANDB_GROUP_MINISERIES,
                n_kv_heads=n_kv_heads,
                skip_connections=skip_connections,
            )
        )
        task_id += 1
    return points


def _fmt_int(x: int) -> str:
    if x >= 1e9:
        return f"{x / 1e9:6.2f}B"
    if x >= 1e6:
        return f"{x / 1e6:6.1f}M"
    if x >= 1e3:
        return f"{x / 1e3:6.1f}K"
    return f"{x:6d}"


def _fmt_flops(x: float) -> str:
    return f"{x:8.2e}"


def print_grid(points: list[GridPoint]) -> None:
    print(f"# Chinchilla grid ({GRID_VERSION})")
    print(f"# isoflop group: {WANDB_GROUP_ISOFLOP}")
    print(f"# miniseries group: {WANDB_GROUP_MINISERIES}  ({MINISERIES_TARGET_TOKENS/1e9:.2f}B tok fixed)")
    print(
        f"# {'task':>4}  {'kind':<10}  {'C (FLOPs)':>10}  {'depth':>5}  {'params':>8}  "
        f"{'tokens':>9}  {'iters':>6}  {'epochs':>6}  run_name"
    )
    total_flops = 0.0
    last_kind = None
    for p in points:
        if last_kind is not None and p.kind != last_kind:
            print(f"# ---")
        last_kind = p.kind
        epochs = p.actual_train_tokens / DATA_POOL_TOKENS
        flag = "*" if epochs > 1.0 else " "
        c_str = _fmt_flops(p.actual_flops) if p.kind == "miniseries" else _fmt_flops(p.flop_budget)
        print(
            f"  {p.task_id:>4}  {p.kind:<10}  {c_str:>10}  {p.depth:>5}  "
            f"{_fmt_int(p.params):>8}  {_fmt_int(p.actual_train_tokens):>9}  "
            f"{p.n_train_iters:>6}  {epochs:>5.2f}{flag}  {p.run_name}"
        )
        total_flops += p.actual_flops
    print(f"# total tasks: {len(points)}   total FLOPs: {total_flops:.2e}")
    print(f"# data pool: {DATA_POOL_TOKENS/1e9:.2f}B tokens  (* = >1 epoch)")


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chinchilla IsoFLOP grid generator.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--print-grid", action="store_true")
    g.add_argument("--count", action="store_true")
    g.add_argument("--task-id", type=int)
    parser.add_argument("--shell", action="store_true",
                        help="With --task-id, emit KEY=VALUE lines (suitable for `eval`).")
    args = parser.parse_args(argv)

    points = enumerate_grid()

    if args.print_grid:
        print_grid(points)
        return 0
    if args.count:
        print(len(points))
        return 0
    if args.task_id is not None:
        if args.task_id < 0 or args.task_id >= len(points):
            print(f"ERROR: task-id {args.task_id} out of range [0, {len(points)})", file=sys.stderr)
            return 2
        p = points[args.task_id]
        if args.shell:
            sys.stdout.write(p.shell_env())
        else:
            for k, v in dataclasses.asdict(p).items():
                print(f"{k}={v}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
