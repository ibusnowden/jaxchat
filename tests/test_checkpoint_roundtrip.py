"""Round-trip the unified checkpoint module."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from jaxchat import checkpoint as ckpt_lib
from jaxchat.presets import D4


def test_save_load_round_trip():
    params = {"wte": np.zeros((4, 8), dtype=np.float32), "lm_head": np.ones((8, 4), dtype=np.float32)}
    opt_state = {"step": np.array(7)}
    with tempfile.TemporaryDirectory() as run_dir:
        path = ckpt_lib.save(
            stage="base",
            step=10,
            params=params,
            opt_state=opt_state,
            config=D4,
            run_dir=run_dir,
            tokenizer_path="/tmp/fake.json",
            rng_seed=42,
        )
        assert os.path.basename(path) == "state_step000010.pkl"
        loaded = ckpt_lib.load_latest(run_dir)
        assert loaded["stage"] == "base"
        assert loaded["step"] == 10
        assert loaded["rng_seed"] == 42
        np.testing.assert_array_equal(loaded["params"]["wte"], params["wte"])
        np.testing.assert_array_equal(loaded["params"]["lm_head"], params["lm_head"])
        assert loaded["opt_state"]["step"] == 7

        # Stage chaining: write an SFT ckpt referring to the base parent.
        ckpt_lib.save(
            stage="sft",
            step=3,
            params=params,
            opt_state=None,
            config=D4,
            run_dir=run_dir,
            parent={"stage": "base", "ckpt_path": path, "step": 10},
        )
        # Top-level latest should now point at SFT.
        top = ckpt_lib.load_latest(run_dir)
        assert top["stage"] == "sft" and top["step"] == 3
        # Explicit base load still works.
        base_again = ckpt_lib.load_latest(run_dir, stage="base")
        assert base_again["step"] == 10
        # list_checkpoints returns both.
        listed = ckpt_lib.list_checkpoints(run_dir)
        stages = sorted(s for s, _, _ in listed)
        assert stages == ["base", "sft"]


def test_legacy_checkpoint_migration():
    """Old Logger.dump pickles must still load."""

    import pickle

    legacy = {
        "step": 5,
        "params": {"wte": np.zeros((2, 2), dtype=np.float32)},
        "opt_state": None,
        "config": D4,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "legacy.pkl")
        with open(path, "wb") as h:
            pickle.dump(legacy, h)
        loaded = ckpt_lib.load_path(path)
        assert loaded["schema_version"] == ckpt_lib.SCHEMA_VERSION
        assert loaded["stage"] == "base"
        assert loaded["step"] == 5
