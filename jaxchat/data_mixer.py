"""Data mixing and document-boundary handling for jaxchat.

Supports:
- Multi-dataset mixing with per-step weights
- Dataset swaps at specified training steps
- Document boundary marking and loss masking

Note: This module intentionally does NOT import from jaxchat.model to avoid
circular imports. model.py imports pure functions from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Dataset schedule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetEntry:
    """A dataset source with optional override glob."""
    glob: str
    weight: float = 1.0
    label: str = ""


@dataclass(frozen=True)
class DatasetSchedulePoint:
    """At this step, use these dataset weights."""
    step: int
    entries: tuple[DatasetEntry, ...]


def resolve_dataset_weights(
    step: int,
    schedule: tuple[DatasetSchedulePoint, ...],
    default_entries: tuple[DatasetEntry, ...],
) -> tuple[DatasetEntry, ...]:
    """Resolve dataset entries and weights at a given step."""
    if not schedule:
        return default_entries

    sorted_pts = sorted(schedule, key=lambda p: p.step)
    applicable = default_entries
    for pt in sorted_pts:
        if step >= pt.step:
            applicable = pt.entries
        else:
            break
    return applicable


# ---------------------------------------------------------------------------
# Document boundary masking
# ---------------------------------------------------------------------------

def build_doc_boundary_mask(
    idx: jax.Array,
    doc_sep_id: int,
) -> jax.Array:
    """Build a mask where 0 = cross-document boundary, 1 = within-document.

    For each position i (predicting token at i+1), the mask is 0 if
    token at i is the document separator (meaning the label at i is the
    start of a new document).  Otherwise it's 1.

    Note on ``doc_sep_id``: the packed-sequence format from
    ``data/cached_fineweb.py`` starts each packed sequence with a BOS token,
    not a dedicated document-separator token.  ``Config.doc_sep_id`` defaults
    to 0, which matches the BOS id of the fineweb tokenizers (``<|bos|>`` is
    token id 1 in the 32k/65k tokenizers, but the 8k d4 tokenizer uses id 0).
    If you switch to a tokenizer whose BOS id is not 0, you must set
    ``Config.doc_sep_id`` to that BOS id (or to a real separator token id) —
    otherwise cross-document masking will silently mask the wrong positions
    (or no positions at all).
    """
    is_boundary = (idx == doc_sep_id).astype(jnp.float32)
    return 1.0 - is_boundary


def maybe_mask_loss(
    loss_per_token: jax.Array,
    idx: jax.Array,
    doc_sep_id: int,
    cross_document_mask: bool,
) -> jax.Array:
    """Mask loss at document boundaries if enabled."""
    if not cross_document_mask:
        return loss_per_token
    mask = build_doc_boundary_mask(idx, doc_sep_id)
    return loss_per_token * mask


def mean_loss_masked(
    token_nll: jax.Array,
    idx: jax.Array,
    doc_sep_id: int,
    cross_document_mask: bool,
) -> jax.Array:
    """Compute mean loss, optionally masking cross-document boundaries."""
    if not cross_document_mask:
        return jnp.mean(token_nll)
    mask = build_doc_boundary_mask(idx, doc_sep_id)
    masked = token_nll * mask
    return jnp.sum(masked) / jnp.maximum(jnp.sum(mask), 1.0)
