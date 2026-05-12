"""Schedule helpers for sequence length, batch size, and dataset mixing.

Provides functions that compute the current (seq_len, batch_size, n_grad_accum)
at a given training step, given a schedule specification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SchedulePoint:
    """A point in a schedule: at ``step``, use these values."""
    step: int
    seq_len: int
    batch_size: int  # total batch size (not micro_batch)


def resolve_schedule(
    step: int,
    points: tuple[SchedulePoint, ...],
    default_seq_len: int,
    default_batch_size: int,
) -> tuple[int, int]:
    """Resolve (seq_len, batch_size) at a given step from a schedule.

    If ``points`` is empty, returns the defaults.
    If step is before the first point, uses that point's values.
    Otherwise, linearly interpolates between the nearest surrounding points
    (with integer rounding), or uses the last point's values if beyond.
    """
    if not points:
        return default_seq_len, default_batch_size

    # Sort by step
    sorted_pts = sorted(points, key=lambda p: p.step)

    # Before or at first point
    if step <= sorted_pts[0].step:
        return sorted_pts[0].seq_len, sorted_pts[0].batch_size

    # After last point
    if step >= sorted_pts[-1].step:
        return sorted_pts[-1].seq_len, sorted_pts[-1].batch_size

    # Interpolate between surrounding points
    for i in range(len(sorted_pts) - 1):
        a = sorted_pts[i]
        b = sorted_pts[i + 1]
        if a.step <= step < b.step:
            frac = (step - a.step) / max(b.step - a.step, 1)
            seq_len = int(round(a.seq_len + frac * (b.seq_len - a.seq_len)))
            batch_size = int(round(a.batch_size + frac * (b.batch_size - a.batch_size)))
            return seq_len, batch_size

    return sorted_pts[-1].seq_len, sorted_pts[-1].batch_size


def get_shape_for_step(
    step: int,
    config: Any,
) -> tuple[int, int, int]:
    """Compute (seq_len, batch_size, n_grad_accum) at a given step.

    Supports:
    - Sequence length warmup (min_seq_len → max_seq_len)
    - Batch size schedule via batch_schedule_points
    - Sequence length schedule via seq_schedule_points
    - Joint schedule via joint_schedule_points (overrides individual)
    """
    micro_batch_size = config.micro_batch_size
    tokens_per_step = config.tokens_per_step

    # Joint schedule (highest priority)
    joint_points = getattr(config, "joint_schedule_points", ())
    if joint_points:
        seq_len, batch_size = resolve_schedule(
            step, joint_points, config.max_seq_len,
            tokens_per_step // config.max_seq_len
        )
        batch_size = max(batch_size, micro_batch_size)
        n_grad_accum = max(batch_size // micro_batch_size, 1)
        return int(seq_len), int(batch_size), int(n_grad_accum)

    # Separate seq_len and batch_size schedules
    seq_points = getattr(config, "seq_schedule_points", ())
    batch_points = getattr(config, "batch_schedule_points", ())

    if seq_points:
        seq_len, _ = resolve_schedule(step, seq_points, config.max_seq_len, 0)
    else:
        seq_len = _warmup_seq_len(step, config)

    if batch_points:
        _, batch_size = resolve_schedule(step, batch_points, 0, tokens_per_step // seq_len)
    else:
        batch_size = tokens_per_step // seq_len

    batch_size = max(batch_size, micro_batch_size)
    # Ensure batch_size is divisible by micro_batch_size (required for grad_accum)
    batch_size = (batch_size // micro_batch_size) * micro_batch_size
    n_grad_accum = max(batch_size // micro_batch_size, 1)
    tokens_actual = batch_size * seq_len
    # Warn if we're losing more than 5% of tokens_per_step
    # (this is fine — the warmup has variable lengths)

    return int(seq_len), int(batch_size), int(n_grad_accum)


def _warmup_seq_len(step: int, config: Any) -> int:
    """Compute sequence length with warmup from min_seq_len to max_seq_len.

    Quantizes to multiples of 64 to minimize distinct JIT compilation shapes.
    Only returns distinct values every 64 steps of warmup.
    """
    min_seq = config.min_seq_len
    max_seq = config.max_seq_len
    warmup_steps = getattr(config, "sequence_warmup_intervals", 0)

    if warmup_steps <= 0 or max_seq <= min_seq:
        return max_seq

    if step >= warmup_steps:
        return max_seq

    frac = step / max(warmup_steps, 1)
    raw_seq = min_seq + frac * (max_seq - min_seq)
    # Quantize to multiple of 64 to reduce JIT compilation count
    seq_len = int(round(raw_seq / 64.0)) * 64
    return max(min_seq, min(seq_len, max_seq))


def get_train_shape_counts(config) -> dict[tuple[int, int, int], int]:
    """Count how many steps use each (seq_len, batch_size, grad_accum) shape."""
    counts: dict[tuple[int, int, int], int] = {}
    for step in range(config.n_train_iters):
        shape = get_shape_for_step(step, config)
        counts[shape] = counts.get(shape, 0) + 1
    return counts


def format_shape_summary(shape_counts: dict[tuple[int, int, int], int]) -> str:
    parts = []
    for (seq_len, batch_size, grad_accum), count in sorted(shape_counts.items()):
        parts.append(
            f"(seq_len={seq_len}, batch={batch_size}, grad_accum={grad_accum}) x {count}"
        )
    return "; ".join(parts)


def get_eval_shape(config) -> tuple[int, int, int]:
    """Get the fixed shape used for evaluation.
    
    Uses config.val_tokens to determine batch size, falling back to
    tokens_per_step when val_tokens is not set. Caps batch at a reasonable
    size to avoid OOM on the lm_head projection (batch * seq * vocab).
    """
    seq_len = config.max_seq_len
    val_tokens = getattr(config, "val_tokens", 0)
    if val_tokens > 0:
        batch_size = max(val_tokens // seq_len, 1)
    else:
        batch_size = config.tokens_per_step // seq_len
    # Cap eval batch to avoid OOM on lm_head projection
    max_safe_batch = max(131072 // seq_len, 1)  # ~256K tokens max
    batch_size = min(batch_size, max_safe_batch)
    n_grad_accum = max(batch_size // config.micro_batch_size, 1)
    return int(seq_len), int(batch_size), int(n_grad_accum)
