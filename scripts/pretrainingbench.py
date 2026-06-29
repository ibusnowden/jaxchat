"""PretrainingBench scaling-law sweep for 1×H100.

Defines a 20-model compute-optimal sweep (Kaplan/Chinchilla/Muennighoff) over a
single clean pretraining corpus, sized for one H100.  The grid is the single
source of truth consumed by the SLURM driver (``runs/h100_pretrainingbench.sh``)
and the scaling-law fit (``scripts/fit_scaling_law.py``).

Design notes
------------
* **One clean corpus, one vocab.**  All points share the 32K-BPE FineWeb-Edu
  re-tokenization (``data/fineweb32k_real_29``, ~2.94B tokens) so the embedding
  tax is constant across the sweep and the only free axis is model width.  This
  is the standard setup for a clean scaling law (Kaplan 2020, Chinchilla 2022).
  The smallest 32K model is ~13M (depth 2); pass ``--vocab 4096`` to push the
  floor toward ~2M if a smaller tokenizer/data shard is available.
* **20 models, even depths 2..40.**  ``d_model = depth*64`` (Config invariant),
  so depth 2..40 spans d_model 128..2560 and (at vocab 32K) ~13M..~13.8B
  params.  Each point gets a Chinchilla-optimal token budget
  ``D = min(ratio * non_embedding_params, data_pool)`` — the data-pool cap is
  the data-constrained regime (Muennighoff 2023) where small models repeat the
  corpus.
* **1×H100 feasibility flags.**  Every point reports estimated wall-hours
  (from ``6*N*D`` FLOPs at a configurable H100 MFU) and a memory estimate, so
  the driver can skip points that won't fit a single 80 GB H100 in the budget.

CLI
---
    python -m scripts.pretrainingbench --print-grid
    python -m scripts.pretrainingbench --count
    python -m scripts.pretrainingbench --task-id 7
    python -m scripts.pretrainingbench --task-id 7 --shell
    python -m scripts.pretrainingbench --print-grid --vocab 4096 --max-depth 22

References
----------
* Kaplan et al. 2020 — L(N,D) = E + A/N^α + B/D^β            (arXiv:2001.08361)
* Hoffmann et al. 2022 (Chinchilla) — compute-optimal N*,D*     (arXiv:2203.15556)
* Muennighoff et al. 2023 — data-constrained scaling             (arXiv:2305.16264)
* Bi & Lin 2024 — PretrainingBench                                (arXiv:2506.10972)
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


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

GRID_VERSION = "v1"
WANDB_GROUP = f"pretrainingbench-{GRID_VERSION}"
BASE_PRESET = "124m-modern"

# 20 even depths 2..40 → d_model 128..2560.  At vocab=32K this spans ~13M..~13.8B
# params; at vocab=4K it spans ~2M..~2.4B.  The driver runs the feasible subset.
DEFAULT_DEPTHS: tuple[int, ...] = tuple(range(2, 42, 2))

# FineWeb-Edu 32K-BPE re-tokenized pool (29 shards).  See presets.py.
DEFAULT_DATA_POOL_TOKENS = 2_940_000_000

# Chinchilla token-to-parameter ratio on *non-embedding* params (the transformer
# matrices + value embeds + token-feature tables).  20 is the Chinchilla
# compute-optimal ratio; the repo's default train_token_ratio is 10.5.
DEFAULT_CHINCHILLA_RATIO = 20.0

# 1×H100 bf16 peak ~989 TFLOP/s (sparsity-2 dense).  Sustained MuonAdamW MFU on
# this stack is ~0.40-0.50; 0.45 is the default for wall-time estimates.
H100_PEAK_TFLOPS = 989.0
DEFAULT_H100_MFU = 0.45

# Memory guard: params (bf16, 2 B) + MuonAdamW state (m + v ≈ 4 B/param for the
# Adam groups; Muon matrices carry m only) + activations.  ~6 B/param is a safe
# budget; 80 GB H100 → ~1.5B params before OOM risk.
DEFAULT_MEM_CAP_PARAMS = 1_500_000_000

# Per-model wall-time guard for the default grid print (hours).  Points above
# this are flagged infeasible for a single H100 within the budget.
DEFAULT_PER_MODEL_HOURS = 6.0


def _build_config(depth: int, vocab: int, tokens_per_step: int | None) -> Config:
    """Build a 124m-modern Config at (depth, vocab) with the scaling-law feature set.

    The sweep holds the *modern* feature set fixed (WSD, deepnorm, grad-clip,
    z-loss, GQA, long-short attn, bigram, cross-doc mask) and varies only width.
    Bigram/cross-doc/long-short are kept ON to match the SOTA preset — the
    scaling law is over the *production* architecture, not a stripped baseline.
    """
    base = PRESETS[BASE_PRESET]
    cfg = dataclasses.replace(
        base,
        depth=depth,
        vocab_size=vocab,
        target_train_tokens=0,
        n_train_iters=0,
        n_kv_heads=n_kv_heads_for_depth(depth, base),
        skip_connections=skip_connections_for_depth(depth, base),
    )
    if tokens_per_step is not None:
        cfg = dataclasses.replace(cfg, tokens_per_step=tokens_per_step)
    return cfg


def n_kv_heads_for_depth(depth: int, base: Config) -> int:
    """2 KV heads when it divides n_heads; otherwise MQA (1)."""
    cfg = dataclasses.replace(base, depth=depth)
    preferred = int(base.n_kv_heads)
    if preferred > 0 and cfg.n_heads % preferred == 0:
        return preferred
    return 1


def skip_connections_for_depth(depth: int, base: Config) -> tuple[tuple[int, int], ...]:
    """Keep only preset skip pairs valid for this depth (src < dst < depth)."""
    return tuple(
        (int(src), int(dst))
        for src, dst in base.skip_connections
        if 0 <= int(src) < int(dst) < depth
    )


@dataclasses.dataclass(frozen=True)
class BenchPoint:
    task_id: int
    depth: int
    vocab: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    skip_connections: tuple[tuple[int, int], ...]
    params: int
    non_emb_params: int
    chinchilla_tokens: int            # ratio * non_emb (uncapped)
    target_train_tokens: int          # min(chinchilla_tokens, data_pool)
    n_train_iters: int
    actual_train_tokens: int
    actual_flops: float
    data_repeats: float               # actual_train_tokens / data_pool
    tokens_per_step: int
    est_wall_hours: float             # 6*N*D / (MFU*peak*1e12) / 3600
    mem_gb: float                     # ~6 bytes/param
    feasible_1h100: bool
    run_name: str
    wandb_group: str

    def shell_env(self) -> str:
        skip = repr(self.skip_connections)
        return (
            f"TASK_ID={self.task_id}\n"
            f"DEPTH={self.depth}\n"
            f"VOCAB={self.vocab}\n"
            f"D_MODEL={self.d_model}\n"
            f"N_HEADS={self.n_heads}\n"
            f"N_KV_HEADS={self.n_kv_heads}\n"
            f"PARAMS={self.params}\n"
            f"NON_EMB_PARAMS={self.non_emb_params}\n"
            f"CHINCHILLA_TOKENS={self.chinchilla_tokens}\n"
            f"TARGET_TRAIN_TOKENS={self.target_train_tokens}\n"
            f"N_TRAIN_ITERS={self.n_train_iters}\n"
            f"ACTUAL_TRAIN_TOKENS={self.actual_train_tokens}\n"
            f"ACTUAL_FLOPS={self.actual_flops:.4e}\n"
            f"DATA_REPEATS={self.data_repeats:.4f}\n"
            f"TOKENS_PER_STEP={self.tokens_per_step}\n"
            f"EST_WALL_HOURS={self.est_wall_hours:.3f}\n"
            f"MEM_GB={self.mem_gb:.2f}\n"
            f"FEASIBLE_1H100={'1' if self.feasible_1h100 else '0'}\n"
            f"RUN_NAME={shlex.quote(self.run_name)}\n"
            f"WANDB_GROUP_TASK={shlex.quote(self.wandb_group)}\n"
            f"SKIP_CONNECTIONS={shlex.quote(skip)}\n"
        )


def _round_target_to_step(target_tokens: float, tokens_per_step: int) -> tuple[int, int, int]:
    n_iters = max(1, math.ceil(target_tokens / tokens_per_step))
    actual = n_iters * tokens_per_step
    return int(target_tokens), int(n_iters), int(actual)


def _fmt_int(x: int) -> str:
    if x >= 1e9:
        return f"{x / 1e9:7.3f}B"
    if x >= 1e6:
        return f"{x / 1e6:7.2f}M"
    if x >= 1e3:
        return f"{x / 1e3:7.1f}K"
    return f"{x:7d}"


def _fmt_hours(h: float) -> str:
    if h < 1.0:
        return f"{h * 60:6.1f}m"
    if h < 48.0:
        return f"{h:6.2f}h"
    return f"{h / 24:6.2f}d"


def enumerate_grid(
    depths: tuple[int, ...] = DEFAULT_DEPTHS,
    vocab: int = 32768,
    data_pool_tokens: int = DEFAULT_DATA_POOL_TOKENS,
    chinchilla_ratio: float = DEFAULT_CHINCHILLA_RATIO,
    tokens_per_step: int | None = None,
    h100_mfu: float = DEFAULT_H100_MFU,
    mem_cap_params: int = DEFAULT_MEM_CAP_PARAMS,
    per_model_hours: float = DEFAULT_PER_MODEL_HOURS,
) -> list[BenchPoint]:
    """Build the PretrainingBench sweep (one BenchPoint per depth)."""
    effective_tflops = H100_PEAK_TFLOPS * h100_mfu
    points: list[BenchPoint] = []
    for task_id, depth in enumerate(depths):
        cfg = _build_config(depth, vocab, tokens_per_step)
        bd = expected_parameter_breakdown(cfg)
        n_params = bd["total"]
        non_emb = bd["total"] - bd["wte"] - bd["lm_head"]
        chinchilla_tokens = int(non_emb * chinchilla_ratio)
        target_tokens = min(chinchilla_tokens, data_pool_tokens)
        tps = int(cfg.tokens_per_step)
        _, n_iters, actual = _round_target_to_step(target_tokens, tps)
        actual_flops = 6.0 * n_params * actual
        data_repeats = actual / data_pool_tokens
        wall_h = actual_flops / (effective_tflops * 1e12) / 3600.0
        mem_gb = (n_params * 6.0) / 1e9
        feasible = (wall_h <= per_model_hours) and (n_params <= mem_cap_params)
        points.append(
            BenchPoint(
                task_id=task_id,
                depth=depth,
                vocab=vocab,
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                skip_connections=skip_connections_for_depth(depth, PRESETS[BASE_PRESET]),
                params=n_params,
                non_emb_params=non_emb,
                chinchilla_tokens=chinchilla_tokens,
                target_train_tokens=target_tokens,
                n_train_iters=n_iters,
                actual_train_tokens=actual,
                actual_flops=actual_flops,
                data_repeats=data_repeats,
                tokens_per_step=tps,
                est_wall_hours=wall_h,
                mem_gb=mem_gb,
                feasible_1h100=feasible,
                run_name=f"pbench-{GRID_VERSION}-d{depth:02d}-v{vocab}",
                wandb_group=WANDB_GROUP,
            )
        )
    return points


def print_grid(points: list[BenchPoint]) -> None:
    print(f"# PretrainingBench grid ({GRID_VERSION})  —  1×H100, vocab={points[0].vocab}, "
          f"{len(points)} models")
    print(f"# wandb group: {WANDB_GROUP}")
    print(f"# Chinchilla ratio: {DEFAULT_CHINCHILLA_RATIO:.1f}× non-embedding params  "
          f"  data pool: {DEFAULT_DATA_POOL_TOKENS / 1e9:.2f}B tokens")
    print(f"# H100 MFU: {DEFAULT_H100_MFU:.2f}  (peak {H100_PEAK_TFLOPS:.0f} TFLOP/s)  "
          f"mem cap: {DEFAULT_MEM_CAP_PARAMS / 1e9:.2f}B params  per-model: {DEFAULT_PER_MODEL_HOURS:.1f}h")
    print(
        f"# {'task':>4}  {'depth':>5}  {'d_model':>7}  {'params':>10}  {'non_emb':>10}  "
        f"{'tokens':>10}  {'iters':>7}  {'repeats':>7}  {'FLOPs':>10}  {'wall':>8}  "
        f"{'mem':>5}  {'ok':>2}  run_name"
    )
    total_flops = 0.0
    total_wall = 0.0
    n_feasible = 0
    for p in points:
        flag = "✓" if p.feasible_1h100 else "✗"
        print(
            f"  {p.task_id:>4}  {p.depth:>5}  {p.d_model:>7}  {_fmt_int(p.params):>10}  "
            f"{_fmt_int(p.non_emb_params):>10}  {_fmt_int(p.actual_train_tokens):>10}  "
            f"{p.n_train_iters:>7}  {p.data_repeats:>6.2f}x  {p.actual_flops:>10.2e}  "
            f"{_fmt_hours(p.est_wall_hours):>8}  {p.mem_gb:>4.1f}G  {flag:>2}  {p.run_name}"
        )
        total_flops += p.actual_flops
        total_wall += p.est_wall_hours
        n_feasible += int(p.feasible_1h100)
    print(f"# total: {len(points)} models, {n_feasible} feasible on 1×H100, "
          f"sum FLOPs {total_flops:.2e}, sum wall {_fmt_hours(total_wall)}")


def _parse_depths(s: str) -> tuple[int, ...]:
    """Accept '2,4,...,40' (expanded) or '2-40:2' (start-end:step)."""
    s = s.strip()
    if ":" in s and "-" in s:
        rng, step = s.split(":")
        step = int(step)
        start, end = rng.split("-")
        return tuple(range(int(start), int(end) + 1, step))
    if "," in s:
        return tuple(int(x) for x in s.split(","))
    raise SystemExit(f"--depths expects 'a,b,c' or 'start-end:step', got {s!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PretrainingBench 1×H100 scaling-law sweep grid.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--print-grid", action="store_true")
    g.add_argument("--count", action="store_true")
    g.add_argument("--task-id", type=int)
    parser.add_argument("--shell", action="store_true", help="With --task-id, emit KEY=VALUE lines.")
    parser.add_argument("--vocab", type=int, default=32768, help="Tokenizer vocab (default 32768).")
    parser.add_argument("--depths", default="2-40:2",
                        help="Comma list '2,4,6' or range 'start-end:step' (default 2-40:2 → 20 models).")
    parser.add_argument("--max-depth", type=int, default=None, help="Clip depths to <= this ( Convenience).")
    parser.add_argument("--data-pool-tokens", type=int, default=DEFAULT_DATA_POOL_TOKENS)
    parser.add_argument("--chinchilla-ratio", type=float, default=DEFAULT_CHINCHILLA_RATIO)
    parser.add_argument("--tokens-per-step", type=int, default=None,
                        help="Override tokens/step (default: depth-scaled from preset).")
    parser.add_argument("--h100-mfu", type=float, default=DEFAULT_H100_MFU)
    parser.add_argument("--mem-cap-params", type=int, default=DEFAULT_MEM_CAP_PARAMS)
    parser.add_argument("--per-model-hours", type=float, default=DEFAULT_PER_MODEL_HOURS)
    args = parser.parse_args(argv)

    depths = _parse_depths(args.depths)
    if args.max_depth is not None:
        depths = tuple(d for d in depths if d <= args.max_depth)
    if not depths:
        raise SystemExit("No depths after filtering; check --depths/--max-depth.")

    points = enumerate_grid(
        depths=depths,
        vocab=args.vocab,
        data_pool_tokens=args.data_pool_tokens,
        chinchilla_ratio=args.chinchilla_ratio,
        tokens_per_step=args.tokens_per_step,
        h100_mfu=args.h100_mfu,
        mem_cap_params=args.mem_cap_params,
        per_model_hours=args.per_model_hours,
    )

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
                if isinstance(v, float):
                    print(f"{k}={v:.6e}")
                else:
                    print(f"{k}={v}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
