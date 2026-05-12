"""Advanced optimizers for jaxchat: Muon variants, SOAP, schedule-free AdamW.

This module provides factory functions that return ``(init_fn, update_fn)``
pairs compatible with the :class:`jaxchat.model.Optimizer` NamedTuple protocol.

Each optimizer takes ``(config, params, mesh)`` and returns the optimizer
object plus its initial state.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_flatten_with_path, tree_leaves, tree_map, tree_unflatten

Pytree = Any


# ---------------------------------------------------------------------------
# Helpers shared across optimizers
# ---------------------------------------------------------------------------

def _path_to_names(path) -> tuple[str, ...]:
    names = []
    for entry in path:
        if hasattr(entry, "key"):
            names.append(str(entry.key))
        elif hasattr(entry, "idx"):
            names.append(str(entry.idx))
        elif hasattr(entry, "name"):
            names.append(str(entry.name))
        else:
            names.append(str(entry))
    return tuple(names)


def _label_param(names: tuple[str, ...], leaf: jax.Array) -> str:
    if names and names[0] == "wte":
        return "adam_embed"
    if names and names[0] == "value_embeds":
        return "adam_embed"
    if names and names[0] == "lm_head":
        return "adam_lm_head"
    if names and names[0] == "resid_lambdas":
        return "adam_resid"
    if names and names[0] == "x0_lambdas":
        return "adam_x0"
    if names and names[0] in ("lm_head_norm", "skip_lambdas"):
        return "adam_resid"
    if leaf.ndim == 2:
        return "muon"
    raise ValueError(f"Unexpected parameter leaf at {names} with shape {leaf.shape}")


def get_lr_scale(step, n_warmup_iters, n_warmdown_iters, n_train_iters, schedule: str = "linear"):
    """Compute LR multiplier based on schedule type.

    Supported schedules:
      - "linear": warmup → constant → linear warmdown (original)
      - "cosine": warmup → cosine decay → warmdown
      - "wsd":    warmup → constant → rapid linear decay (Warmup-Stable-Decay)
    """
    step = step.astype(jnp.float32)
    n_warmup = float(n_warmup_iters)
    n_total = float(n_train_iters)
    n_warmdown = float(n_warmdown_iters)
    warmdown_start = max(n_total - n_warmdown, 0.0)

    warmup_scale = jnp.where(step < n_warmup, (step + 1.0) / max(n_warmup, 1.0), 1.0)

    # Decay phase
    if schedule == "cosine":
        # Cosine decay from n_warmup to n_warmdown_start
        progress = (step - n_warmup) / max(warmdown_start - n_warmup, 1.0)
        decay_scale = 0.5 * (1.0 + jnp.cos(jnp.clip(progress, 0.0, 1.0) * jnp.pi))
    elif schedule == "wsd":
        # Stable phase: constant 1.0 until warmdown_start
        # Then rapid linear decay
        warmdown_progress = (step - warmdown_start) / max(n_warmdown, 1.0)
        decay_scale = jnp.maximum(1.0 - warmdown_progress, 0.0)
    else:  # linear
        decay_scale = jnp.maximum((n_total - step) / max(n_warmdown, 1.0), 0.0)

    # If no warmdown, decay_scale stays at 1.0 via the safe path
    if n_warmdown <= 0:
        decay_scale = jnp.where(step >= n_warmup, 1.0, 1.0)
    else:
        decay_scale = jnp.where(step < n_warmup, 1.0,
                                jnp.where(step < warmdown_start, 1.0, decay_scale))

    return warmup_scale * decay_scale


def get_weight_decay_scale(step, n_train_iters):
    step = step.astype(jnp.float32)
    return jnp.maximum(1.0 - step / float(max(n_train_iters, 1)), 0.0)


def scaled_group_lr(base_lr: float, d_model: int) -> float:
    return base_lr / math.sqrt(d_model / 768.0)


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization (shared by Muon variants)
# ---------------------------------------------------------------------------

def newton_schulz_orthogonalize(g, steps: int, eps: float):
    """Polar decomposition via Newton-Schulz iterations."""
    transpose = g.shape[-2] > g.shape[-1]
    x = jnp.swapaxes(g, -1, -2) if transpose else g
    x = x.astype(jnp.float32)
    x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps)
    for _ in range(steps):
        gram = jnp.matmul(jnp.swapaxes(x, -1, -2), x)
        x = 1.5 * x - 0.5 * jnp.matmul(x, gram)
    x = jnp.swapaxes(x, -1, -2) if transpose else x
    return x.astype(g.dtype)


# ---------------------------------------------------------------------------
# Power iteration for top-eigenvalue estimation (used by SOAP, NorMuon)
# ---------------------------------------------------------------------------

def power_iteration(matrix: jax.Array, n_iters: int = 5) -> jax.Array:
    """Estimate top eigenvalue via power iteration.

    Args:
        matrix: shape (..., M, N)
        n_iters: number of power iterations

    Returns:
        Top eigenvalue estimate, scalar per batch.
    """
    # Start with a random vector of size N (columns of G)
    vec = jnp.ones_like(matrix[..., 0, :])  # shape (..., N)
    for _ in range(n_iters):
        # v = G^T G v
        # G: (..., M, N), v: (..., N) -> Gv: (..., M) -> G^T(Gv): (..., N)
        gv = jnp.einsum('...ij,...j->...i', matrix, vec)  # (..., M)
        vec = jnp.einsum('...ji,...j->...i', matrix.conj(), gv)  # (..., N)
        vec = vec / (jnp.linalg.norm(vec, axis=-1, keepdims=True) + 1e-10)
    # Rayleigh quotient: (v^T G^T G v) / (v^T v)
    gv = jnp.einsum('...ij,...j->...i', matrix, vec)  # (..., M)
    eigenvalue = jnp.sum(gv ** 2, axis=-1) / (jnp.sum(vec ** 2, axis=-1) + 1e-10)
    return eigenvalue


# ---------------------------------------------------------------------------
# Original DistMuonAdamW (ported from model.py for modular access)
# ---------------------------------------------------------------------------

def make_muon_adamw(config, params, mesh):
    """Original DistMuonAdamW: AdamW for embed/lm_head/scalars, Muon for matrices."""
    del mesh
    from jaxchat.model import Config

    path_leaves, treedef = tree_flatten_with_path(params)
    flat_paths = tuple(_path_to_names(path) for path, _ in path_leaves)
    flat_params = tuple(leaf for _, leaf in path_leaves)
    labels = tuple(_label_param(path, leaf) for path, leaf in zip(flat_paths, flat_params, strict=True))

    adam_groups = {
        "adam_embed": tuple(i for i, label in enumerate(labels) if label == "adam_embed"),
        "adam_lm_head": tuple(i for i, label in enumerate(labels) if label == "adam_lm_head"),
        "adam_resid": tuple(i for i, label in enumerate(labels) if label == "adam_resid"),
        "adam_x0": tuple(i for i, label in enumerate(labels) if label == "adam_x0"),
    }
    muon_groups = {}
    for idx, label in enumerate(labels):
        if label != "muon":
            continue
        muon_groups.setdefault(flat_params[idx].shape, []).append(idx)
    muon_group_specs = tuple((shape, tuple(indices)) for shape, indices in sorted(muon_groups.items()))

    adam_hparams = {
        "adam_embed": {
            "lr": scaled_group_lr(config.embed_lr_base, config.d_model),
            "beta1": config.embed_beta1,
            "beta2": config.adam_beta2,
            "weight_decay": config.weight_decay,
        },
        "adam_lm_head": {
            "lr": scaled_group_lr(config.lm_head_lr_base, config.d_model),
            "beta1": config.lm_head_beta1,
            "beta2": config.adam_beta2,
            "weight_decay": config.weight_decay,
        },
        "adam_resid": {
            "lr": config.scalar_resid_lr,
            "beta1": config.scalar_beta1,
            "beta2": config.adam_beta2,
            "weight_decay": 0.0,
        },
        "adam_x0": {
            "lr": config.scalar_x0_lr,
            "beta1": config.x0_beta1,
            "beta2": config.adam_beta2,
            "weight_decay": 0.0,
        },
    }

    def init_fn(params):
        state = {"step": jnp.array(0, dtype=jnp.int32)}
        for name, indices in adam_groups.items():
            state[name] = {
                "m": tuple(jnp.zeros_like(flat_params[i], dtype=jnp.float32) for i in indices),
                "v": tuple(jnp.zeros_like(flat_params[i], dtype=jnp.float32) for i in indices),
            }
        muon_state = []
        for shape, indices in muon_group_specs:
            muon_state.append(
                {
                    "m": jnp.zeros((len(indices),) + shape, dtype=jnp.float32),
                    "row_v": jnp.zeros((len(indices), shape[0]), dtype=jnp.float32),
                    "col_v": jnp.zeros((len(indices), shape[1]), dtype=jnp.float32),
                }
            )
        state["muon"] = tuple(muon_state)
        return state

    def get_muon_momentum(step):
        frac = jnp.minimum(step / float(max(config.muon_momentum_warmup_steps, 1)), 1.0)
        return config.muon_warmup_momentum_init + frac * (
            config.muon_warmup_momentum_final - config.muon_warmup_momentum_init
        )

    def update_fn(grads, params, state):
        grad_leaves = tree_flatten(grads)[0]
        param_leaves = tree_flatten(params)[0]
        new_leaves = list(param_leaves)
        new_state = {"step": state["step"] + 1}
        step = state["step"].astype(jnp.float32)
        lr_scale = get_lr_scale(
            state["step"], config.n_warmup_iters, config.n_warmdown_iters,
            config.n_train_iters, config.lr_schedule
        )
        wd_scale = get_weight_decay_scale(state["step"], config.n_train_iters)

        # --- AdamW groups ---
        for name, indices in adam_groups.items():
            hparams = adam_hparams[name]
            lr = jnp.asarray(hparams["lr"], dtype=jnp.float32) * lr_scale
            beta1 = jnp.asarray(hparams["beta1"], dtype=jnp.float32)
            beta2 = jnp.asarray(hparams["beta2"], dtype=jnp.float32)
            group_m = []
            group_v = []
            old_group_state = state[name]
            for local_idx, leaf_idx in enumerate(indices):
                grad = grad_leaves[leaf_idx].astype(jnp.float32)
                param = param_leaves[leaf_idx].astype(jnp.float32)
                m_prev = old_group_state["m"][local_idx]
                v_prev = old_group_state["v"][local_idx]
                m = beta1 * m_prev + (1.0 - beta1) * grad
                v = beta2 * v_prev + (1.0 - beta2) * jnp.square(grad)
                m_hat = m / (1.0 - beta1 ** (step + 1.0))
                v_hat = v / (1.0 - beta2 ** (step + 1.0))
                update_value = m_hat / (jnp.sqrt(v_hat) + config.adam_eps)
                if hparams["weight_decay"] > 0.0:
                    update_value = update_value + hparams["weight_decay"] * wd_scale * param
                new_param = param - lr * update_value
                new_leaves[leaf_idx] = new_param.astype(param_leaves[leaf_idx].dtype)
                group_m.append(m)
                group_v.append(v)
            new_state[name] = {"m": tuple(group_m), "v": tuple(group_v)}

        # --- Muon groups ---
        muon_state = []
        momentum = get_muon_momentum(step).astype(jnp.float32)
        muon_lr = jnp.asarray(config.muon_base_lr, dtype=jnp.float32) * lr_scale
        for (shape, indices), group_state in zip(muon_group_specs, state["muon"], strict=True):
            grad_stack = jnp.stack([grad_leaves[i].astype(jnp.float32) for i in indices], axis=0)
            param_stack = jnp.stack([param_leaves[i].astype(jnp.float32) for i in indices], axis=0)
            m_prev = group_state["m"]
            row_prev = group_state["row_v"]
            col_prev = group_state["col_v"]

            row_stat = jnp.mean(jnp.square(grad_stack), axis=-1)
            col_stat = jnp.mean(jnp.square(grad_stack), axis=-2)
            row_v = config.muon_beta2 * row_prev + (1.0 - config.muon_beta2) * row_stat
            col_v = config.muon_beta2 * col_prev + (1.0 - config.muon_beta2) * col_stat
            row_scale = row_v / (jnp.mean(row_v, axis=-1, keepdims=True) + config.muon_eps)
            factored_var = row_scale[..., :, None] * col_v[..., None, :]
            precond_grad = grad_stack / jnp.sqrt(factored_var + config.muon_eps)

            m = momentum * m_prev + (1.0 - momentum) * precond_grad
            nesterov = precond_grad + momentum * m
            ortho = newton_schulz_orthogonalize(nesterov, config.muon_polar_iters, config.muon_eps)
            scale = math.sqrt(max(1.0, shape[0] / shape[1]))
            decay_mask = (grad_stack * param_stack) >= 0
            decay = config.weight_decay * wd_scale * decay_mask.astype(jnp.float32) * param_stack
            new_param_stack = param_stack - muon_lr * (scale * ortho + decay)
            for offset, leaf_idx in enumerate(indices):
                new_leaves[leaf_idx] = new_param_stack[offset].astype(param_leaves[leaf_idx].dtype)
            muon_state.append({"m": m, "row_v": row_v, "col_v": col_v})
        new_state["muon"] = tuple(muon_state)
        return tree_unflatten(treedef, new_leaves), new_state

    return init_fn, update_fn


# ---------------------------------------------------------------------------
# NorMuon: Muon with spectral normalization of updates
# ---------------------------------------------------------------------------

def make_normuon(config, params, mesh):
    """NorMuon: Muon with per-parameter spectral normalization.

    Instead of plain orthogonalization, NorMuon normalizes the gradient by
    its spectral norm estimate (top singular value) before applying the
    Newton-Schulz orthogonalization step.
    """
    del mesh
    path_leaves, treedef = tree_flatten_with_path(params)
    flat_paths = tuple(_path_to_names(path) for path, _ in path_leaves)
    flat_params = tuple(leaf for _, leaf in path_leaves)
    labels = tuple(_label_param(path, leaf) for path, leaf in zip(flat_paths, flat_params, strict=True))

    adam_groups = {
        "adam_embed": tuple(i for i, label in enumerate(labels) if label == "adam_embed"),
        "adam_lm_head": tuple(i for i, label in enumerate(labels) if label == "adam_lm_head"),
        "adam_resid": tuple(i for i, label in enumerate(labels) if label == "adam_resid"),
        "adam_x0": tuple(i for i, label in enumerate(labels) if label == "adam_x0"),
    }
    muon_groups = {}
    for idx, label in enumerate(labels):
        if label != "muon":
            continue
        muon_groups.setdefault(flat_params[idx].shape, []).append(idx)
    muon_group_specs = tuple((shape, tuple(indices)) for shape, indices in sorted(muon_groups.items()))

    adam_hparams = {
        "adam_embed": {
            "lr": scaled_group_lr(config.embed_lr_base, config.d_model),
            "beta1": config.embed_beta1,
            "beta2": config.adam_beta2,
            "weight_decay": config.weight_decay,
        },
        "adam_lm_head": {
            "lr": scaled_group_lr(config.lm_head_lr_base, config.d_model),
            "beta1": config.lm_head_beta1,
            "beta2": config.adam_beta2,
            "weight_decay": config.weight_decay,
        },
        "adam_resid": {
            "lr": config.scalar_resid_lr,
            "beta1": config.scalar_beta1,
            "beta2": config.adam_beta2,
            "weight_decay": 0.0,
        },
        "adam_x0": {
            "lr": config.scalar_x0_lr,
            "beta1": config.x0_beta1,
            "beta2": config.adam_beta2,
            "weight_decay": 0.0,
        },
    }

    def init_fn(params):
        state = {"step": jnp.array(0, dtype=jnp.int32)}
        for name, indices in adam_groups.items():
            state[name] = {
                "m": tuple(jnp.zeros_like(flat_params[i], dtype=jnp.float32) for i in indices),
                "v": tuple(jnp.zeros_like(flat_params[i], dtype=jnp.float32) for i in indices),
            }
        muon_state = []
        for shape, indices in muon_group_specs:
            muon_state.append(
                {
                    "m": jnp.zeros((len(indices),) + shape, dtype=jnp.float32),
                    "row_v": jnp.zeros((len(indices), shape[0]), dtype=jnp.float32),
                    "col_v": jnp.zeros((len(indices), shape[1]), dtype=jnp.float32),
                }
            )
        state["muon"] = tuple(muon_state)
        return state

    def get_muon_momentum(step):
        frac = jnp.minimum(step / float(max(config.muon_momentum_warmup_steps, 1)), 1.0)
        return config.muon_warmup_momentum_init + frac * (
            config.muon_warmup_momentum_final - config.muon_warmup_momentum_init
        )

    def update_fn(grads, params, state):
        grad_leaves = tree_flatten(grads)[0]
        param_leaves = tree_flatten(params)[0]
        new_leaves = list(param_leaves)
        new_state = {"step": state["step"] + 1}
        step = state["step"].astype(jnp.float32)
        lr_scale = get_lr_scale(
            state["step"], config.n_warmup_iters, config.n_warmdown_iters,
            config.n_train_iters, config.lr_schedule
        )
        wd_scale = get_weight_decay_scale(state["step"], config.n_train_iters)

        # --- AdamW groups (same as MuonAdamW) ---
        for name, indices in adam_groups.items():
            hparams = adam_hparams[name]
            lr = jnp.asarray(hparams["lr"], dtype=jnp.float32) * lr_scale
            beta1 = jnp.asarray(hparams["beta1"], dtype=jnp.float32)
            beta2 = jnp.asarray(hparams["beta2"], dtype=jnp.float32)
            group_m = []
            group_v = []
            old_group_state = state[name]
            for local_idx, leaf_idx in enumerate(indices):
                grad = grad_leaves[leaf_idx].astype(jnp.float32)
                param = param_leaves[leaf_idx].astype(jnp.float32)
                m_prev = old_group_state["m"][local_idx]
                v_prev = old_group_state["v"][local_idx]
                m = beta1 * m_prev + (1.0 - beta1) * grad
                v = beta2 * v_prev + (1.0 - beta2) * jnp.square(grad)
                m_hat = m / (1.0 - beta1 ** (step + 1.0))
                v_hat = v / (1.0 - beta2 ** (step + 1.0))
                update_value = m_hat / (jnp.sqrt(v_hat) + config.adam_eps)
                if hparams["weight_decay"] > 0.0:
                    update_value = update_value + hparams["weight_decay"] * wd_scale * param
                new_param = param - lr * update_value
                new_leaves[leaf_idx] = new_param.astype(param_leaves[leaf_idx].dtype)
                group_m.append(m)
                group_v.append(v)
            new_state[name] = {"m": tuple(group_m), "v": tuple(group_v)}

        # --- NorMuon groups (spectral-normalized Muon) ---
        muon_state = []
        momentum = get_muon_momentum(step).astype(jnp.float32)
        muon_lr = jnp.asarray(config.muon_base_lr, dtype=jnp.float32) * lr_scale
        for (shape, indices), group_state in zip(muon_group_specs, state["muon"], strict=True):
            grad_stack = jnp.stack([grad_leaves[i].astype(jnp.float32) for i in indices], axis=0)
            param_stack = jnp.stack([param_leaves[i].astype(jnp.float32) for i in indices], axis=0)
            m_prev = group_state["m"]
            row_prev = group_state["row_v"]
            col_prev = group_state["col_v"]

            row_stat = jnp.mean(jnp.square(grad_stack), axis=-1)
            col_stat = jnp.mean(jnp.square(grad_stack), axis=-2)
            row_v = config.muon_beta2 * row_prev + (1.0 - config.muon_beta2) * row_stat
            col_v = config.muon_beta2 * col_prev + (1.0 - config.muon_beta2) * col_stat
            row_scale = row_v / (jnp.mean(row_v, axis=-1, keepdims=True) + config.muon_eps)
            factored_var = row_scale[..., :, None] * col_v[..., None, :]
            precond_grad = grad_stack / jnp.sqrt(factored_var + config.muon_eps)

            # NorMuon: spectral normalization of update
            spectral_norm = power_iteration(precond_grad, n_iters=3)
            spectral_norm = spectral_norm[..., None, None]  # broadcast
            normalized_grad = precond_grad / (spectral_norm + config.muon_eps)

            m = momentum * m_prev + (1.0 - momentum) * normalized_grad
            nesterov = normalized_grad + momentum * m

            # Orthogonalize the spectrally-normalized update
            ortho = newton_schulz_orthogonalize(nesterov, config.muon_polar_iters, config.muon_eps)
            scale = math.sqrt(max(1.0, shape[0] / shape[1]))
            decay_mask = (grad_stack * param_stack) >= 0
            decay = config.weight_decay * wd_scale * decay_mask.astype(jnp.float32) * param_stack
            new_param_stack = param_stack - muon_lr * (scale * ortho + decay)
            for offset, leaf_idx in enumerate(indices):
                new_leaves[leaf_idx] = new_param_stack[offset].astype(param_leaves[leaf_idx].dtype)
            muon_state.append({"m": m, "row_v": row_v, "col_v": col_v})
        new_state["muon"] = tuple(muon_state)
        return tree_unflatten(treedef, new_leaves), new_state

    return init_fn, update_fn


# ---------------------------------------------------------------------------
# SOAP: Shampoo (Adam in special bases) with power-iteration
# ---------------------------------------------------------------------------

def make_soap(config, params, mesh):
    """SOAP optimizer: maintains eigenbasis of gradient moments via power iteration.

    Groups: embed + value_embeds use plain AdamW. lm_head, matrices use SOAP.
    Scalars use plain AdamW.

    Reference: https://arxiv.org/abs/2409.11321
    """
    del mesh
    path_leaves, treedef = tree_flatten_with_path(params)
    flat_paths = tuple(_path_to_names(path) for path, _ in path_leaves)
    flat_params = tuple(leaf for _, leaf in path_leaves)
    labels = tuple(_label_param(path, leaf) for path, leaf in zip(flat_paths, flat_params, strict=True))

    soap_rank = getattr(config, "soap_rank", 32)
    soap_update_freq = getattr(config, "soap_update_freq", 10)
    soap_beta2 = getattr(config, "soap_beta2", 0.95)

    adam_groups = {
        "adam_embed": tuple(i for i, label in enumerate(labels) if label == "adam_embed"),
        "adam_lm_head": tuple(i for i, label in enumerate(labels) if label == "adam_lm_head"),
        "adam_resid": tuple(i for i, label in enumerate(labels) if label == "adam_resid"),
        "adam_x0": tuple(i for i, label in enumerate(labels) if label == "adam_x0"),
    }
    soap_groups = {}
    for idx, label in enumerate(labels):
        if label == "muon":
            soap_groups.setdefault(flat_params[idx].shape, []).append(idx)
    soap_group_specs = tuple((shape, tuple(indices)) for shape, indices in sorted(soap_groups.items()))

    adam_hparams = {
        "adam_embed": {
            "lr": scaled_group_lr(config.embed_lr_base, config.d_model),
            "beta1": config.embed_beta1,
            "beta2": config.adam_beta2,
            "weight_decay": config.weight_decay,
        },
        "adam_lm_head": {
            "lr": scaled_group_lr(config.lm_head_lr_base, config.d_model),
            "beta1": config.lm_head_beta1,
            "beta2": config.adam_beta2,
            "weight_decay": config.weight_decay,
        },
        "adam_resid": {
            "lr": config.scalar_resid_lr,
            "beta1": config.scalar_beta1,
            "beta2": config.adam_beta2,
            "weight_decay": 0.0,
        },
        "adam_x0": {
            "lr": config.scalar_x0_lr,
            "beta1": config.x0_beta1,
            "beta2": config.adam_beta2,
            "weight_decay": 0.0,
        },
    }

    def _soap_init_for_shape(shape, n_matrices):
        """Initialize SOAP state for matrices of a given shape."""
        M, N = shape
        rank = min(soap_rank, M, N)
        return {
            "m_left": jnp.zeros((n_matrices, M, rank), dtype=jnp.float32),
            "m_right": jnp.zeros((n_matrices, N, rank), dtype=jnp.float32),
            "v_left": jnp.zeros((n_matrices, M, rank), dtype=jnp.float32),
            "v_right": jnp.zeros((n_matrices, N, rank), dtype=jnp.float32),
            "Q_left": jnp.eye(M, rank, dtype=jnp.float32)[None, :, :].repeat(n_matrices, axis=0),
            "Q_right": jnp.eye(N, rank, dtype=jnp.float32)[None, :, :].repeat(n_matrices, axis=0),
            "step": jnp.zeros((n_matrices,), dtype=jnp.int32),
        }

    def init_fn(params):
        state = {"step": jnp.array(0, dtype=jnp.int32)}
        for name, indices in adam_groups.items():
            state[name] = {
                "m": tuple(jnp.zeros_like(flat_params[i], dtype=jnp.float32) for i in indices),
                "v": tuple(jnp.zeros_like(flat_params[i], dtype=jnp.float32) for i in indices),
            }
        for shape, indices in soap_group_specs:
            state[f"soap_{shape}"] = _soap_init_for_shape(shape, len(indices))
        return state

    def _compute_soap_update(grad, global_step, soap_state_for_group):
        """Compute Adam in the eigenbasis of G^T G."""
        M, N = grad.shape
        rank = soap_state_for_group["Q_left"].shape[-1]
        Q_left = soap_state_for_group["Q_left"]  # (M, rank)
        Q_right = soap_state_for_group["Q_right"]  # (N, rank)

        # Project gradient into eigenbasis
        grad_proj = jnp.einsum('mi,mn,nj->ij', Q_left.T, grad, Q_right)  # (rank, rank)

        # Adam in eigenbasis
        beta1 = jnp.asarray(config.embed_beta1 if labels[0] == "adam_embed" else 0.9, dtype=jnp.float32)
        gs = global_step.astype(jnp.float32)
        m_prev_left = soap_state_for_group["m_left"]
        m_prev_right = soap_state_for_group["m_right"]
        v_prev_left = soap_state_for_group["v_left"]
        v_prev_right = soap_state_for_group["v_right"]

        # Factorized moment accumulation
        m_left = beta1 * m_prev_left + (1.0 - beta1) * grad @ Q_right
        m_right = beta1 * m_prev_right + (1.0 - beta1) * grad.T @ Q_left
        v_left = soap_beta2 * v_prev_left + (1.0 - soap_beta2) * (grad @ Q_right) ** 2
        v_right = soap_beta2 * v_prev_right + (1.0 - soap_beta2) * (grad.T @ Q_left) ** 2

        m_hat_left = m_left / (1.0 - beta1 ** (gs + 1.0))
        m_hat_right = m_right / (1.0 - beta1 ** (gs + 1.0))
        v_hat_left = v_left / (1.0 - soap_beta2 ** (gs + 1.0))
        v_hat_right = v_right / (1.0 - soap_beta2 ** (gs + 1.0))

        update = (m_hat_left / (jnp.sqrt(v_hat_left) + config.adam_eps)) @ (m_hat_right / (jnp.sqrt(v_hat_right) + config.adam_eps)).T
        return update, {"m_left": m_left, "m_right": m_right, "v_left": v_left, "v_right": v_right}

    def _update_eigenbasis(grad, soap_state_for_group):
        """Update eigenbasis estimate via power iteration on G^T G."""
        M, N = grad.shape
        rank = soap_state_for_group["Q_left"].shape[-1]
        Q_left_old = soap_state_for_group["Q_left"]
        Q_right_old = soap_state_for_group["Q_right"]

        # Power iteration on G @ G^T for left basis
        vec_left = Q_left_old
        for _ in range(3):
            vec_left = grad @ (grad.T @ vec_left)
            vec_left = vec_left / (jnp.linalg.norm(vec_left, axis=-2, keepdims=True) + 1e-10)

        # Power iteration on G^T @ G for right basis
        vec_right = Q_right_old
        for _ in range(3):
            vec_right = grad.T @ (grad @ vec_right)
            vec_right = vec_right / (jnp.linalg.norm(vec_right, axis=-2, keepdims=True) + 1e-10)

        return {"Q_left": vec_left, "Q_right": vec_right}

    def update_fn(grads, params, state):
        grad_leaves = tree_flatten(grads)[0]
        param_leaves = tree_flatten(params)[0]
        new_leaves = list(param_leaves)
        new_state = {"step": state["step"] + 1}
        step = state["step"].astype(jnp.float32)
        lr_scale = get_lr_scale(
            state["step"], config.n_warmup_iters, config.n_warmdown_iters,
            config.n_train_iters, config.lr_schedule
        )
        wd_scale = get_weight_decay_scale(state["step"], config.n_train_iters)

        # --- AdamW groups ---
        for name, indices in adam_groups.items():
            hparams = adam_hparams[name]
            lr = jnp.asarray(hparams["lr"], dtype=jnp.float32) * lr_scale
            beta1 = jnp.asarray(hparams["beta1"], dtype=jnp.float32)
            beta2 = jnp.asarray(hparams["beta2"], dtype=jnp.float32)
            group_m = []
            group_v = []
            old_group_state = state[name]
            for local_idx, leaf_idx in enumerate(indices):
                grad = grad_leaves[leaf_idx].astype(jnp.float32)
                param = param_leaves[leaf_idx].astype(jnp.float32)
                m_prev = old_group_state["m"][local_idx]
                v_prev = old_group_state["v"][local_idx]
                m = beta1 * m_prev + (1.0 - beta1) * grad
                v = beta2 * v_prev + (1.0 - beta2) * jnp.square(grad)
                m_hat = m / (1.0 - beta1 ** (step + 1.0))
                v_hat = v / (1.0 - beta2 ** (step + 1.0))
                update_value = m_hat / (jnp.sqrt(v_hat) + config.adam_eps)
                if hparams["weight_decay"] > 0.0:
                    update_value = update_value + hparams["weight_decay"] * wd_scale * param
                new_param = param - lr * update_value
                new_leaves[leaf_idx] = new_param.astype(param_leaves[leaf_idx].dtype)
                group_m.append(m)
                group_v.append(v)
            new_state[name] = {"m": tuple(group_m), "v": tuple(group_v)}

        # --- SOAP groups ---
        soap_lr = jnp.asarray(config.muon_base_lr, dtype=jnp.float32) * lr_scale
        for (shape, indices) in soap_group_specs:
            group_key = f"soap_{shape}"
            old_soap_state = state[group_key]
            new_soap_m_left = []
            new_soap_m_right = []
            new_soap_v_left = []
            new_soap_v_right = []
            new_Q_left = []
            new_Q_right = []

            for local_idx, leaf_idx in enumerate(indices):
                grad = grad_leaves[leaf_idx].astype(jnp.float32)
                param = param_leaves[leaf_idx].astype(jnp.float32)

                # Per-matrix SOAP state
                per_mat_state = {
                    "m_left": old_soap_state["m_left"][local_idx],
                    "m_right": old_soap_state["m_right"][local_idx],
                    "v_left": old_soap_state["v_left"][local_idx],
                    "v_right": old_soap_state["v_right"][local_idx],
                    "Q_left": old_soap_state["Q_left"][local_idx],
                    "Q_right": old_soap_state["Q_right"][local_idx],
                }

                # Occasionally update eigenbasis
                should_update = (state["step"] % soap_update_freq == 0)
                per_mat_state = jax.lax.cond(
                    should_update,
                    lambda g, s: _update_eigenbasis(g, s),
                    lambda g, s: s,
                    grad, per_mat_state,
                )

                update, new_moments = _compute_soap_update(grad, state["step"], per_mat_state)

                decay = config.weight_decay * wd_scale * param
                new_param = param - soap_lr * (update + decay)
                new_leaves[leaf_idx] = new_param.astype(param_leaves[leaf_idx].dtype)

                new_soap_m_left.append(new_moments["m_left"])
                new_soap_m_right.append(new_moments["m_right"])
                new_soap_v_left.append(new_moments["v_left"])
                new_soap_v_right.append(new_moments["v_right"])
                new_Q_left.append(per_mat_state["Q_left"])
                new_Q_right.append(per_mat_state["Q_right"])

            new_state[group_key] = {
                "m_left": jnp.stack(new_soap_m_left, axis=0),
                "m_right": jnp.stack(new_soap_m_right, axis=0),
                "v_left": jnp.stack(new_soap_v_left, axis=0),
                "v_right": jnp.stack(new_soap_v_right, axis=0),
                "Q_left": jnp.stack(new_Q_left, axis=0),
                "Q_right": jnp.stack(new_Q_right, axis=0),
                "step": old_soap_state["step"] + 1,
            }

        return tree_unflatten(treedef, new_leaves), new_state

    return init_fn, update_fn


# ---------------------------------------------------------------------------
# Optimizer registry
# ---------------------------------------------------------------------------

OPTIMIZER_REGISTRY: dict[str, Callable] = {
    "muon_adamw": make_muon_adamw,
    "normuon": make_normuon,
    "soap": make_soap,
}


def create_optimizer(config, params, mesh):
    """Factory: return ``(init_fn, update_fn)`` based on ``config.optimizer``.

    ``config.optimizer`` must be one of the keys in ``OPTIMIZER_REGISTRY``.
    Defaults to ``"muon_adamw"`` for backward compatibility.
    """
    name = getattr(config, "optimizer", "muon_adamw")
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer {name!r}. "
            f"Available: {list(OPTIMIZER_REGISTRY)}"
        )
    return OPTIMIZER_REGISTRY[name](config, params, mesh)
