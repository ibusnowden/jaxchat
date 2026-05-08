"""Smoke-test the inference engine using a stub tokenizer + fresh init params."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
from jax.tree_util import tree_map  # noqa: E402

from jaxchat import checkpoint as ckpt_lib  # noqa: E402
from jaxchat.engine import Engine  # noqa: E402
from jaxchat.model import (  # noqa: E402
    get_data_parallel_sharding,
    get_mesh,
    get_weight_sharding,
    init_params,
    precompute_rope,
)
from jaxchat.presets import SMOKE  # noqa: E402


class _StubTokenizer:
    """Identity-style stub good enough for engine-level tests."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def get_bos_token_id(self) -> int:
        return 1

    def encode(self, text):
        if isinstance(text, list):
            return [self.encode(t) for t in text]
        return [(c % (self.vocab_size - 1)) + 1 for c in text.encode("utf-8")][:64]

    def decode(self, ids):
        return "".join(chr(int(i) % 128) for i in ids)

    def encode_special(self, name: str):
        if name == "<|assistant_end|>":
            return 2
        if name == "<|bos|>":
            return 1
        return 0

    def render_for_completion(self, conversation):
        # Concatenate all message contents, stripping the trailing assistant placeholder.
        msgs = conversation["messages"][:-1]
        text = "\n".join(m["content"] for m in msgs)
        return [self.get_bos_token_id()] + self.encode(text)


def _make_engine(stage: str = "base") -> Engine:
    config = SMOKE
    mesh = get_mesh(config)
    weight_sharding = get_weight_sharding(config, mesh)
    with mesh:
        params, _ = init_params(config, mesh)
        precomputed = precompute_rope(config, mesh)
    embedding_out_sharding = get_data_parallel_sharding(config, mesh, ndim=3)
    return Engine(
        params=tree_map(lambda x: jax.device_put(x, weight_sharding), params),
        precomputed_params=precomputed,
        config=config,
        mesh=mesh,
        tokenizer=_StubTokenizer(config.vocab_size),
        embedding_out_sharding=embedding_out_sharding,
        stage=stage,
        step=0,
    )


def test_generate_ids_produces_tokens():
    engine = _make_engine()
    out = engine.generate_ids([1, 2, 3, 4], max_new_tokens=8, temperature=0.8, top_k=20, seed=0)
    assert len(out) == 8
    assert all(0 <= t < engine.config.vocab_size for t in out)


def test_score_continuation_returns_float():
    engine = _make_engine()
    score = engine.score_continuation([1, 2, 3], [4, 5, 6])
    assert isinstance(score, float)
    assert score < 0.0  # log-probs are non-positive.


def test_engine_load_round_trip(tmp_path):
    engine = _make_engine()
    host_params = jax.device_get(engine.params)
    ckpt_lib.save(
        stage="base",
        step=0,
        params=host_params,
        opt_state=None,
        config=engine.config,
        run_dir=str(tmp_path),
    )
    state = ckpt_lib.load_latest(str(tmp_path), stage="base")
    assert state["step"] == 0
    # Quick equality on one tensor.
    np.testing.assert_array_equal(np.asarray(state["params"]["wte"]), np.asarray(host_params["wte"]))
