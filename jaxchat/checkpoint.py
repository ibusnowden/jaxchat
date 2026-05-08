"""Unified checkpoint save/load for the staged jaxchat pipeline.

A run directory looks like::

    {run_dir}/
        latest.txt                       # name of the freshest stage
        base/
            latest_checkpoint.txt        # absolute path to the freshest base ckpt
            state_step000200.pkl
            state_step000400.pkl
        sft/
            latest_checkpoint.txt
            state_step000050.pkl
        rl/
            latest_checkpoint.txt
            state_step000020.pkl

Each ``state_step{step:06d}.pkl`` is a single pickle file with the schema below.
"""

from __future__ import annotations

import datetime as _dt
import os
import pickle
from typing import Any, Iterable

import jax

Pytree = Any
SCHEMA_VERSION = 1
VALID_STAGES = ("base", "sft", "rl")


def _ensure_stage(stage: str) -> str:
    if stage not in VALID_STAGES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {VALID_STAGES}")
    return stage


def _stage_dir(run_dir: str, stage: str) -> str:
    return os.path.join(run_dir, _ensure_stage(stage))


def _atomic_write_text(path: str, text: str) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp_path, path)


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def save(
    *,
    stage: str,
    step: int,
    params: Pytree,
    opt_state: Pytree | None,
    config: Any,
    run_dir: str,
    tokenizer_path: str = "",
    rng_seed: int | None = None,
    parent: dict | None = None,
) -> str:
    """Persist a checkpoint for ``stage`` and update the latest pointers.

    Returns the absolute path to the written ``.pkl`` file.
    """

    stage = _ensure_stage(stage)
    run_dir = os.path.abspath(run_dir)
    stage_dir = _stage_dir(run_dir, stage)
    os.makedirs(stage_dir, exist_ok=True)

    params_host = jax.device_get(params)
    opt_state_host = jax.device_get(opt_state) if opt_state is not None else None

    state = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "step": int(step),
        "params": params_host,
        "opt_state": opt_state_host,
        "config": config,
        "rng_seed": int(rng_seed) if rng_seed is not None else None,
        "tokenizer_path": tokenizer_path,
        "parent": parent,
        "saved_at": _now_iso(),
    }

    save_path = os.path.join(stage_dir, f"state_step{int(step):06d}.pkl")
    tmp_path = f"{save_path}.tmp"
    with open(tmp_path, "wb") as handle:
        pickle.dump(state, handle)
    os.replace(tmp_path, save_path)

    _atomic_write_text(os.path.join(stage_dir, "latest_checkpoint.txt"), save_path + "\n")
    _atomic_write_text(os.path.join(run_dir, "latest.txt"), stage + "\n")
    return save_path


def _read_latest_pointer(run_dir: str, stage: str) -> str:
    pointer = os.path.join(_stage_dir(run_dir, stage), "latest_checkpoint.txt")
    if not os.path.exists(pointer):
        raise FileNotFoundError(f"No latest_checkpoint.txt for stage {stage!r} in {run_dir}")
    with open(pointer, "r", encoding="utf-8") as handle:
        path = handle.read().strip()
    if not path:
        raise RuntimeError(f"latest_checkpoint.txt at {pointer} was empty")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint referenced by {pointer} is missing: {path}")
    return path


def _read_top_pointer(run_dir: str) -> str:
    pointer = os.path.join(run_dir, "latest.txt")
    if not os.path.exists(pointer):
        raise FileNotFoundError(
            f"No latest.txt at {run_dir}; specify stage explicitly via --stage"
        )
    with open(pointer, "r", encoding="utf-8") as handle:
        stage = handle.read().strip()
    if not stage:
        raise RuntimeError(f"latest.txt at {pointer} was empty")
    return _ensure_stage(stage)


def load_latest(run_dir: str, stage: str | None = None) -> dict:
    """Load the freshest checkpoint, optionally constrained to a stage."""

    run_dir = os.path.abspath(run_dir)
    if stage is None:
        stage = _read_top_pointer(run_dir)
    path = _read_latest_pointer(run_dir, stage)
    return load_path(path)


def load_path(path: str) -> dict:
    with open(path, "rb") as handle:
        state = pickle.load(handle)
    state = _migrate(state)
    state.setdefault("_checkpoint_path", path)
    return state


def list_checkpoints(run_dir: str, stage: str | None = None) -> list[tuple[str, int, str]]:
    """List ``(stage, step, path)`` triples discovered under ``run_dir``."""

    stages: Iterable[str] = (stage,) if stage is not None else VALID_STAGES
    results: list[tuple[str, int, str]] = []
    for stg in stages:
        stage_dir = _stage_dir(run_dir, stg)
        if not os.path.isdir(stage_dir):
            continue
        for entry in sorted(os.listdir(stage_dir)):
            if not entry.startswith("state_step") or not entry.endswith(".pkl"):
                continue
            try:
                step = int(entry[len("state_step") : -len(".pkl")])
            except ValueError:
                continue
            results.append((stg, step, os.path.join(stage_dir, entry)))
    return results


def init_from_parent(
    parent_run_dir: str,
    parent_stage: str,
) -> tuple[Pytree, Any, dict]:
    """Convenience helper used by SFT/RL bootstrap.

    Returns ``(params, config, parent_meta)``.  ``parent_meta`` has the keys
    ``stage`` and ``ckpt_path`` and is meant to be threaded into ``save(parent=...)``.
    """

    state = load_latest(parent_run_dir, parent_stage)
    parent_meta = {
        "stage": state["stage"],
        "ckpt_path": state.get("_checkpoint_path", ""),
        "step": state["step"],
    }
    return state["params"], state["config"], parent_meta


def _migrate(state: dict) -> dict:
    """Minimal forward-compat shim for older checkpoint dictionaries."""

    if "schema_version" not in state:
        # Legacy ``Logger.dump`` shape: {step, params, opt_state, config}.
        state = {
            "schema_version": SCHEMA_VERSION,
            "stage": "base",
            "step": int(state.get("step", 0)),
            "params": state["params"],
            "opt_state": state.get("opt_state"),
            "config": state.get("config"),
            "rng_seed": None,
            "tokenizer_path": "",
            "parent": None,
            "saved_at": "",
        }
    elif state["schema_version"] > SCHEMA_VERSION:
        raise RuntimeError(
            f"Checkpoint schema_version={state['schema_version']} is newer than "
            f"this code understands ({SCHEMA_VERSION}); upgrade jaxchat."
        )
    return state


__all__ = [
    "SCHEMA_VERSION",
    "VALID_STAGES",
    "save",
    "load_latest",
    "load_path",
    "list_checkpoints",
    "init_from_parent",
]
