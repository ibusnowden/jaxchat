"""Inference engine for jaxchat.

Loads a staged checkpoint via :mod:`jaxchat.checkpoint`, reconstructs the JAX
runtime, and exposes simple ``generate`` / ``chat`` / ``score_continuation``
helpers.

Notes:
    - This v1 has no KV cache.  Each generated token re-runs the full
      ``gpt_forward`` over a buffer padded to ``config.max_seq_len`` so JIT
      can be reused across prompt lengths.  That is fast enough for the d4
      (~11M-param) target on a single H100 and avoids the complexity of a
      cache layout that would couple to the attention dispatcher in fa3.py.
    - Tokenizer is loaded via :func:`jaxchat.tokenizer.load_tokenizer`.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.tree_util import tree_map

from jax.sharding import NamedSharding, PartitionSpec as P

from jaxchat import checkpoint as ckpt_lib
from jaxchat.model import (
    get_mesh,
    get_weight_sharding,
    gpt_forward,
    precompute_rope,
)
from jaxchat.tokenizer import load_tokenizer
from jaxchat.tools import execute_python


def _logits_at(params, precomputed_params, config, embedding_out_sharding, idx_padded, pos):
    logits = gpt_forward(params, idx_padded, precomputed_params, config, embedding_out_sharding)
    return logits[0, pos]


def _logprobs_full(params, precomputed_params, config, embedding_out_sharding, idx_padded):
    return gpt_forward(params, idx_padded, precomputed_params, config, embedding_out_sharding)


def _apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        out = np.full_like(logits, -np.inf)
        out[int(np.argmax(logits))] = 0.0
        return out
    return logits / float(temperature)


def _top_k_top_p_filter(logits: np.ndarray, top_k: int | None, top_p: float | None) -> np.ndarray:
    out = logits.copy()
    if top_k is not None and top_k > 0 and top_k < out.shape[0]:
        kth = np.partition(out, -top_k)[-top_k]
        out = np.where(out < kth, -np.inf, out)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_idx = np.argsort(out)[::-1]
        sorted_logits = out[sorted_idx]
        sorted_logits = sorted_logits - np.max(sorted_logits)
        probs = np.exp(sorted_logits)
        probs = probs / np.sum(probs)
        cdf = np.cumsum(probs)
        cutoff = np.searchsorted(cdf, top_p) + 1
        keep = sorted_idx[:cutoff]
        mask = np.full_like(out, -np.inf)
        mask[keep] = out[keep]
        out = mask
    return out


def _sample(rng: np.random.Generator, logits: np.ndarray) -> int:
    finite = np.isfinite(logits)
    if not finite.any():
        return int(np.argmax(logits))
    logits = logits - np.max(logits[finite])
    probs = np.where(finite, np.exp(logits), 0.0)
    s = probs.sum()
    if s <= 0.0:
        return int(np.argmax(logits))
    probs = probs / s
    return int(rng.choice(probs.shape[0], p=probs))


class Engine:
    """Stateless wrapper around a sharded jaxchat checkpoint."""

    def __init__(self, *, params, precomputed_params, config, mesh, tokenizer, embedding_out_sharding, stage: str, step: int):
        self.params = params
        self.precomputed_params = precomputed_params
        self.config = config
        self.mesh = mesh
        self.tokenizer = tokenizer
        self.embedding_out_sharding = embedding_out_sharding
        self.stage = stage
        self.step = step
        self._pad_id = int(tokenizer.get_bos_token_id())
        self._jit_logits_at = jit(
            _logits_at,
            static_argnames=("config", "embedding_out_sharding"),
        )
        self._jit_logprobs_full = jit(
            _logprobs_full,
            static_argnames=("config", "embedding_out_sharding"),
        )
        self._n_params = None  # lazily computed by model_info()

    def model_info(self) -> dict:
        """Return a JSON-serializable summary of the loaded model for UI/servers."""
        c = self.config

        def g(name, default=None):
            return getattr(c, name, default)

        if self._n_params is None:
            try:
                self._n_params = int(
                    sum(int(np.prod(leaf.shape)) for leaf in jax.tree_util.tree_leaves(self.params))
                )
            except Exception:
                self._n_params = 0

        def _fmt_params(n: int) -> str:
            for unit, val in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
                if n >= val:
                    return f"{n / val:.2f}{unit}"
            return str(n)

        return {
            "stage": self.stage,
            "step": int(self.step),
            "n_params": self._n_params,
            "n_params_human": _fmt_params(self._n_params),
            "depth": g("depth"),
            "n_layers": g("n_layers"),
            "d_model": g("d_model"),
            "n_heads": g("n_heads"),
            "n_kv_heads": g("n_kv_heads"),
            "n_recurrence": g("n_recurrence"),
            "vocab_size": g("vocab_size"),
            "max_seq_len": g("max_seq_len"),
            "tokenizer_name": g("tokenizer_name"),
            "logit_softcap": g("logit_softcap"),
            "dtype": str(g("dtype")),
        }

    @classmethod
    def from_run_dir(
        cls,
        run_dir: str,
        *,
        stage: str | None = None,
        tokenizer_path: str | None = None,
        single_device: bool = False,
    ) -> "Engine":
        state = ckpt_lib.load_latest(run_dir, stage=stage)
        return cls.from_state(state, tokenizer_path=tokenizer_path, single_device=single_device)

    @classmethod
    def from_path(
        cls, ckpt_path: str, *, tokenizer_path: str | None = None, single_device: bool = False
    ) -> "Engine":
        state = ckpt_lib.load_path(ckpt_path)
        return cls.from_state(state, tokenizer_path=tokenizer_path, single_device=single_device)

    @classmethod
    def from_state(
        cls, state: dict, *, tokenizer_path: str | None = None, single_device: bool = False
    ) -> "Engine":
        config = state["config"]
        if single_device:
            # Collapse any data-parallel mesh to a single-device mesh (one device
            # per axis) so a checkpoint trained across N GPUs can be served for
            # inference on one device (a single GPU, or CPU). We keep the axis
            # names so any sharding spec that references them still resolves; with
            # one device per axis every sharding degenerates to replicated.
            new_shape = tuple(1 for _ in config.mesh_shape) if config.mesh_shape else (1,)
            new_names = config.mesh_axis_names if config.mesh_axis_names else ("dp",)
            config = dataclasses.replace(config, mesh_shape=new_shape, mesh_axis_names=new_names)
        mesh = get_mesh(config)
        weight_sharding = get_weight_sharding(config, mesh)
        params = tree_map(lambda leaf: jax.device_put(jnp.asarray(leaf), weight_sharding), state["params"])
        with mesh:
            precomputed_params = precompute_rope(config, mesh)
        # Match train_base / eval_base: the embedding/output activations use
        # ``config.activation_sharding`` (replicated batch for the 124m presets)
        # so single-sample inference works on multi-GPU meshes where batch=1
        # would not divide ``dp``.
        embedding_out_sharding = NamedSharding(mesh, P(*config.activation_sharding))
        tok_path = tokenizer_path or state.get("tokenizer_path") or config.tokenizer_json
        if not tok_path:
            raise RuntimeError(
                "No tokenizer_path on checkpoint and config.tokenizer_json is empty; "
                "pass tokenizer_path= explicitly."
            )
        tokenizer = load_tokenizer(tok_path)
        return cls(
            params=params,
            precomputed_params=precomputed_params,
            config=config,
            mesh=mesh,
            tokenizer=tokenizer,
            embedding_out_sharding=embedding_out_sharding,
            stage=str(state.get("stage", "base")),
            step=int(state.get("step", 0)),
        )

    def _pad_to_max(self, ids: list[int]) -> tuple[jax.Array, int]:
        T = self.config.max_seq_len
        if len(ids) > T:
            ids = ids[-T:]
        n = len(ids)
        buf = np.full((1, T), self._pad_id, dtype=np.int32)
        buf[0, :n] = np.asarray(ids, dtype=np.int32)
        return jnp.asarray(buf), n

    def _next_token_logits(self, ids: list[int]) -> np.ndarray:
        idx_padded, n = self._pad_to_max(ids)
        with self.mesh:
            logits = self._jit_logits_at(
                self.params,
                self.precomputed_params,
                self.config,
                self.embedding_out_sharding,
                idx_padded,
                n - 1,
            )
        return np.asarray(jax.device_get(logits))

    def prefill_ids(self, prompt_ids: list[int]) -> dict:
        """Create a decode cache object.

        The current cache stores token ids and preserves the future KV-cache API
        shape. Attention KV tensors can be added behind this interface without
        changing callers.
        """
        return {"ids": list(prompt_ids)}

    def decode_one(self, cache: dict, *, temperature: float = 0.8, top_k: int | None = 40,
                   top_p: float | None = None, seed: int = 0) -> tuple[int, dict]:
        rng = np.random.default_rng(seed)
        ids = list(cache.get("ids", []))
        logits = self._next_token_logits(ids)
        logits = _apply_temperature(logits, temperature)
        logits = _top_k_top_p_filter(logits, top_k, top_p)
        tok = _sample(rng, logits)
        ids.append(tok)
        return tok, {"ids": ids}

    def generate_ids(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int | None = 40,
        top_p: float | None = None,
        seed: int = 0,
        stop_token_ids: Iterable[int] | None = None,
    ) -> list[int]:
        rng = np.random.default_rng(seed)
        stop = set(int(t) for t in (stop_token_ids or ()))
        ids = list(prompt_ids)
        out: list[int] = []
        for _ in range(max_new_tokens):
            if len(ids) >= self.config.max_seq_len:
                break
            logits = self._next_token_logits(ids)
            logits = _apply_temperature(logits, temperature)
            logits = _top_k_top_p_filter(logits, top_k, top_p)
            tok = _sample(rng, logits)
            ids.append(tok)
            out.append(tok)
            if tok in stop:
                break
        return out

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int | None = 40,
        top_p: float | None = None,
        seed: int = 0,
    ) -> str:
        prompt_ids = [self._pad_id] + list(self.tokenizer.encode(prompt))
        out_ids = self.generate_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        return self.tokenizer.decode(out_ids)

    def chat(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        seed: int = 0,
    ) -> str:
        """Render ``messages`` (with a final user turn) and continue as the assistant."""

        if not messages or messages[-1]["role"] != "user":
            raise ValueError("chat() expects messages to end with a user turn.")
        # render_for_completion expects the trailing role to be 'assistant'; add a sentinel.
        primed = {"messages": messages + [{"role": "assistant", "content": ""}]}
        prompt_ids = self.tokenizer.render_for_completion(primed)

        end_id = self.tokenizer.encode_special("<|assistant_end|>")
        bos_id = self.tokenizer.get_bos_token_id()
        stop_ids = {tid for tid in (end_id, bos_id) if tid is not None}

        gen_ids = self.generate_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            stop_token_ids=stop_ids,
        )
        # Strip any trailing stop token (assistant_end or bos) — generate_ids
        # appends the stop token before breaking, and special tokens decode to
        # their literal "<|...|>" surface form (skip_special_tokens=False),
        # which would otherwise leak into the rendered reply.
        if gen_ids and gen_ids[-1] in stop_ids:
            gen_ids = gen_ids[:-1]
        return self.tokenizer.decode(gen_ids)

    def chat_stream(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        seed: int = 0,
    ):
        """Yield decoded text deltas for ``messages`` one token at a time.

        Mirrors :meth:`chat` semantics (render_for_completion priming, stop on
        ``<|assistant_end|>``/BOS) but streams the reply instead of returning it
        whole. Decodes the accumulated continuation each step so BPE merges across
        token boundaries are handled correctly; yields only the new suffix.
        """
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("chat_stream() expects messages to end with a user turn.")
        primed = {"messages": messages + [{"role": "assistant", "content": ""}]}
        prompt_ids = self.tokenizer.render_for_completion(primed)
        end_id = self.tokenizer.encode_special("<|assistant_end|>")
        bos_id = self.tokenizer.get_bos_token_id()
        stop_ids = {tid for tid in (end_id, bos_id) if tid is not None}

        rng = np.random.default_rng(seed)
        ids = list(prompt_ids)
        gen_ids: list[int] = []
        decoded_prev = ""
        for _ in range(max_new_tokens):
            if len(ids) >= self.config.max_seq_len:
                break
            logits = self._next_token_logits(ids)
            logits = _apply_temperature(logits, temperature)
            logits = _top_k_top_p_filter(logits, top_k, top_p)
            tok = _sample(rng, logits)
            ids.append(tok)
            if tok in stop_ids:
                break
            gen_ids.append(tok)
            text = self.tokenizer.decode(gen_ids)
            if len(text) > len(decoded_prev):
                yield text[len(decoded_prev):]
                decoded_prev = text

    def chat_with_tools(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        seed: int = 0,
        max_tool_rounds: int = 2,
        tool_timeout_s: float = 2.0,
    ) -> dict:
        """Chat and execute local Python tool calls emitted with special tokens."""
        working = list(messages)
        events: list[dict] = []
        reply = ""
        for round_idx in range(max(max_tool_rounds, 0) + 1):
            reply = self.chat(
                working,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed + round_idx,
            )
            start = reply.find("<|python_start|>")
            end = reply.find("<|python_end|>", start + 1)
            if start < 0 or end < 0:
                return {"reply": reply, "events": events}
            code = reply[start + len("<|python_start|>"):end].strip()
            result = execute_python(code, timeout_s=tool_timeout_s)
            events.append({"type": "python", "code": code, **result})
            tool_augmented = (
                reply[: end + len("<|python_end|>")]
                + "<|output_start|>"
                + result["output"]
                + "<|output_end|>"
            )
            working = working + [{"role": "assistant", "content": tool_augmented}]
        return {"reply": reply, "events": events}

    def score_continuation(self, ctx_ids: list[int], cont_ids: list[int]) -> float:
        """Sum log-probs of ``cont_ids`` conditioned on ``ctx_ids`` (natural log).

        Used by :mod:`tasks.core` for ranked-classification eval.
        """

        ids = list(ctx_ids) + list(cont_ids)
        idx_padded, n = self._pad_to_max(ids)
        with self.mesh:
            logits = self._jit_logprobs_full(
                self.params,
                self.precomputed_params,
                self.config,
                self.embedding_out_sharding,
                idx_padded,
            )
        logits = np.asarray(jax.device_get(logits))[0, : n - 1]  # predict pos t+1 from pos t
        # We need log-prob of cont_ids at positions [len(ctx)-1 .. len(ctx)+len(cont)-2]
        ctx_len = len(ctx_ids)
        cont_len = len(cont_ids)
        if cont_len == 0:
            return 0.0
        start = max(ctx_len - 1, 0)
        end = start + cont_len
        if end > logits.shape[0]:
            cont_len = logits.shape[0] - start
            end = start + cont_len
            cont_ids = cont_ids[:cont_len]
        slab = logits[start:end]  # shape (cont_len, vocab)
        slab = slab - np.max(slab, axis=-1, keepdims=True)
        probs = np.exp(slab)
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        gather = probs[np.arange(cont_len), np.asarray(cont_ids, dtype=np.int64)]
        return float(np.sum(np.log(np.maximum(gather, 1e-30))))


__all__ = ["Engine"]
