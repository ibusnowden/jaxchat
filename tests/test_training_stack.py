import json
import os
import pickle
import subprocess
import tempfile
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from data.cached_fineweb import iter_tokenized_documents, load_tokenizer as load_preprocess_tokenizer
from jax.nn import dot_product_attention

from jaxchat.fa3 import attention, backend_decision, format_runtime_info
from jaxchat.model import (
    BIN_HEADER_BYTES,
    BIN_HEADER_INTS,
    BIN_MAGIC,
    BIN_VERSION,
    Config,
    eval_step,
    expected_parameter_breakdown,
    get_data_parallel_sharding,
    get_mesh,
    init_optimizer,
    init_params,
    load_dataset,
    parameter_breakdown_from_params,
    parameter_optimizer_labels,
    precompute_token_bytes,
    train_step,
)
from jaxchat.tokenizer import (
    HuggingFaceTokenizer,
    load_hf_tokenizer,
    normalize_dataset_specs,
    resolve_tokenizer_json_path,
)
from training.eval_base import evaluate_run
from training.eval_tokenizer import evaluate_tokenizer
from jaxchat.presets import DEFAULT_CONFIG
from training.train_base import (
    train_loop,
    validate_training_assets,
)


class StubLogger:
    def msg(self, msg: str) -> None:
        del msg


def write_bin(path: str, tokens: np.ndarray) -> None:
    header = np.zeros(BIN_HEADER_INTS, dtype=np.int32)
    header[0] = BIN_MAGIC
    header[1] = BIN_VERSION
    header[2] = int(tokens.size)
    with open(path, "wb") as handle:
        handle.write(header.tobytes())
        handle.write(tokens.astype(np.uint16).tobytes())


def make_dataset_tokens(*, seq_len: int, bos: int, num_sequences: int = 8) -> np.ndarray:
    sequences = []
    for i in range(num_sequences):
        payload = [((i * (seq_len - 1) + j) % 200) + 2 for j in range(seq_len - 1)]
        sequences.append(np.asarray([bos] + payload, dtype=np.uint16))
    return np.concatenate(sequences, axis=0)


class TrainingStackTests(unittest.TestCase):
    def test_exact_depth24_parameter_breakdown(self) -> None:
        cfg = Config(input_bin="train.bin", input_val_bin="val.bin")
        self.assertEqual(cfg.depth, 24)
        self.assertEqual(cfg.d_model, 1536)
        self.assertEqual(cfg.n_heads, 12)
        self.assertEqual(cfg.target_train_tokens, 14533312248)
        self.assertEqual(cfg.n_train_iters, 27721)
        self.assertEqual(
            expected_parameter_breakdown(cfg),
            {
                "wte": 50331648,
                "value_embeds": 603979776,
                "lm_head": 50331648,
                "transformer_matrices": 679481856,
                "scalars": 48,
                "total": 1384124976,
            },
        )

    def test_small_model_matches_formula_and_optimizer_routing(self) -> None:
        cfg = Config(
            input_bin="train.bin",
            input_val_bin="val.bin",
            depth=4,
            vocab_size=4096,
            min_seq_len=64,
            max_seq_len=64,
            tokens_per_step=1024,
            micro_batch_size=4,
            target_train_tokens=4096,
        )
        mesh = get_mesh(cfg)
        with mesh:
            params, _ = init_params(cfg, mesh)
            self.assertEqual(parameter_breakdown_from_params(params), expected_parameter_breakdown(cfg))
            labels = parameter_optimizer_labels(params)

        self.assertEqual(labels["wte"], "adam_embed")
        self.assertEqual(labels["value_embeds"], "adam_embed")
        self.assertEqual(labels["lm_head"], "adam_lm_head")
        self.assertEqual(labels["resid_lambdas"], "adam_resid")
        self.assertEqual(labels["x0_lambdas"], "adam_x0")
        self.assertEqual(labels["ve_gates/0"], "muon")
        self.assertEqual(labels["blocks/0/attn/wq"], "muon")
        self.assertTrue(all(label in {"adam_embed", "adam_lm_head", "adam_resid", "adam_x0", "muon"} for label in labels.values()))

    def test_attention_cpu_fallback_matches_sdpa(self) -> None:
        cfg = Config(
            input_bin="train.bin",
            input_val_bin="val.bin",
            depth=4,
            vocab_size=4096,
            min_seq_len=16,
            max_seq_len=16,
            tokens_per_step=256,
            micro_batch_size=4,
            target_train_tokens=1024,
            sliding_window_pattern=(4, 4, 4, 8),
        )
        q = jax.random.normal(jax.random.PRNGKey(0), (1, 8, cfg.n_heads, cfg.d_head), dtype=jnp.float32)
        k = jax.random.normal(jax.random.PRNGKey(1), (1, 8, cfg.n_heads, cfg.d_head), dtype=jnp.float32)
        v = jax.random.normal(jax.random.PRNGKey(2), (1, 8, cfg.n_heads, cfg.d_head), dtype=jnp.float32)
        out = attention(q, k, v, layer_idx=0, config=cfg, mesh=None)
        ref = dot_product_attention(
            q,
            k,
            v,
            scale=1.0 / np.sqrt(cfg.d_head),
            is_causal=True,
            local_window_size=(3, 0),
        )
        np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-5, atol=1e-5)
        decision = backend_decision(q, k, v, layer_idx=0, config=cfg, mesh=None)
        self.assertEqual(decision.backend, "sdpa")
        self.assertIn("sliding-window", decision.reason)

    def test_runtime_summary_reports_backend_surface(self) -> None:
        summary = format_runtime_info()
        self.assertIn("gpu_attention=", summary)
        self.assertIn("gpu_attention_mgpu=", summary)
        self.assertIn("mgpu_causal_training_safe=False", summary)

    def test_tokenizer_helpers_use_repo_32k_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = HuggingFaceTokenizer.train_from_iterator(
                ["hello world", "hello tokenizer", "world of jax"], vocab_size=256
            )
            tokenizer.save(tmpdir)
            by_dir = load_hf_tokenizer(tmpdir)
            by_json = load_hf_tokenizer(resolve_tokenizer_json_path(tmpdir))
            self.assertEqual(by_dir.get_bos_token_id(), by_json.get_bos_token_id())
            self.assertTrue(by_dir.encode("hello world"))
            docs = list(
                iter_tokenized_documents(
                    [{"text": "hello world"}, {"text": ""}, {"text": "jax"}],
                    by_dir,
                    "text",
                )
            )
            self.assertEqual(len(docs), 2)
            self.assertEqual(docs[0].dtype, np.uint16)

            loaded = load_preprocess_tokenizer(
                tokenizer_path_or_dir=tmpdir,
                train_if_missing=False,
                dataset_names="ignored",
                dataset_configs=None,
                split="train",
                text_field="text",
                vocab_size=256,
                max_documents=None,
            )
            self.assertEqual(loaded.get_bos_token_id(), by_dir.get_bos_token_id())

    def test_precompute_token_bytes_tolerates_missing_tokenizer_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                input_bin="train.bin",
                input_val_bin="val.bin",
                depth=4,
                vocab_size=32,
                tokenizer_json=os.path.join(tmpdir, "missing-tokenizer"),
                min_seq_len=16,
                max_seq_len=16,
                tokens_per_step=256,
                micro_batch_size=4,
                target_train_tokens=1024,
            )
            mesh = get_mesh(cfg)
            with mesh:
                token_bytes = precompute_token_bytes(cfg, mesh)

        np.testing.assert_array_equal(np.asarray(token_bytes), np.ones(cfg.vocab_size, dtype=np.int32))

    def test_eval_tokenizer_writes_expected_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer = HuggingFaceTokenizer.train_from_iterator(
                ["hello world", "hello tokenizer", "world of jax"], vocab_size=256
            )
            tokenizer.save(tmpdir)
            report = evaluate_tokenizer(
                tmpdir,
                dataset_names="ignored",
                sample_texts=("hello world", "jax tokenizer"),
                max_documents=2,
            )
            output_path = os.path.join(tmpdir, "tokenizer_eval.json")
            self.assertTrue(os.path.exists(output_path))
            with open(output_path, "r", encoding="utf-8") as handle:
                written = json.load(handle)
            self.assertTrue(written["roundtrip_matches"])
            self.assertEqual(written["bos_token"], "<|bos|>")
            self.assertEqual(written["vocab_size"], report["vocab_size"])
            self.assertGreater(written["avg_bytes_per_token"], 0.0)

    def test_default_training_paths_are_repo_root_anchored(self) -> None:
        self.assertTrue(os.path.isabs(DEFAULT_CONFIG.input_bin))
        self.assertTrue(os.path.isabs(DEFAULT_CONFIG.input_val_bin))
        self.assertTrue(os.path.isabs(DEFAULT_CONFIG.tokenizer_json))

    def test_normalize_dataset_specs_supports_combined_fineweb(self) -> None:
        self.assertEqual(
            normalize_dataset_specs("HuggingFaceFW/fineweb-edu,HuggingFaceFW/fineweb", None),
            (
                ("HuggingFaceFW/fineweb-edu", None),
                ("HuggingFaceFW/fineweb", None),
            ),
        )
        self.assertEqual(
            normalize_dataset_specs("a,b", "default"),
            (("a", "default"), ("b", "default")),
        )

    def test_validate_training_assets_fails_before_runtime_setup(self) -> None:
        cfg = Config(
            input_bin="/definitely/missing/train_*.bin",
            input_val_bin="/definitely/missing/val_000000.bin",
        )
        with self.assertRaises(FileNotFoundError):
            validate_training_assets(cfg)

    def test_train_loop_run_dir_writes_latest_checkpoint_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_len = 8
            bos = 1
            tokens = make_dataset_tokens(seq_len=seq_len, bos=bos)
            train_path = os.path.join(tmpdir, "train_000000.bin")
            val_path = os.path.join(tmpdir, "val_000000.bin")
            run_dir = os.path.join(tmpdir, "run")
            write_bin(train_path, tokens)
            write_bin(val_path, tokens)

            cfg = Config(
                input_bin=train_path,
                input_val_bin=val_path,
                depth=4,
                vocab_size=256,
                tokenizer_bos_id=bos,
                min_seq_len=seq_len,
                max_seq_len=seq_len,
                tokens_per_step=seq_len,
                micro_batch_size=1,
                target_train_tokens=seq_len,
                val_tokens=seq_len,
                val_loss_every=100,
                save_every=0,
            )
            train_loop(cfg, preset_name="tiny", run_dir=run_dir)

            train_log = os.path.join(run_dir, "train.log")
            base_latest = os.path.join(run_dir, "base", "latest_checkpoint.txt")
            top_latest = os.path.join(run_dir, "latest.txt")
            self.assertTrue(os.path.exists(train_log))
            self.assertTrue(os.path.exists(base_latest))
            self.assertTrue(os.path.exists(top_latest))
            with open(base_latest, "r", encoding="utf-8") as handle:
                checkpoint_path = handle.read().strip()
            self.assertTrue(os.path.exists(checkpoint_path))
            self.assertTrue(checkpoint_path.startswith(run_dir))
            with open(top_latest, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read().strip(), "base")

    def test_streaming_loader_preserves_bos_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_len = 8
            bos = 1
            sequences = []
            for i in range(8):
                sequences.append(np.asarray([bos, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7, i + 8], dtype=np.uint16))
            tokens = np.concatenate(sequences, axis=0)
            train_path = os.path.join(tmpdir, "train_000000.bin")
            val_path = os.path.join(tmpdir, "val_000000.bin")
            write_bin(train_path, tokens)
            write_bin(val_path, tokens)

            cfg = Config(
                input_bin=train_path,
                input_val_bin=val_path,
                depth=4,
                vocab_size=512,
                tokenizer_bos_id=bos,
                min_seq_len=seq_len,
                max_seq_len=seq_len,
                tokens_per_step=32,
                micro_batch_size=2,
                target_train_tokens=64,
            )
            mesh = get_mesh(cfg)
            with mesh:
                loader = load_dataset(cfg, StubLogger(), mesh, is_training=True)
                batched_x, batched_y = next(loader)
            self.assertEqual(batched_x.shape, (2, 2, seq_len))
            self.assertEqual(batched_y.shape, (2, 2, seq_len))
            flat_x = np.asarray(batched_x).reshape(-1, seq_len)
            self.assertTrue(np.all(flat_x[:, 0] == bos))

    def test_eval_base_writes_metrics_and_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_len = 32
            bos = 1
            tokens = make_dataset_tokens(seq_len=seq_len, bos=bos)
            train_path = os.path.join(tmpdir, "train_000000.bin")
            val_path = os.path.join(tmpdir, "val_000000.bin")
            tokenizer_dir = os.path.join(tmpdir, "tokenizer")
            run_dir = os.path.join(tmpdir, "run")
            os.makedirs(run_dir, exist_ok=True)
            write_bin(train_path, tokens)
            write_bin(val_path, tokens)

            tokenizer = HuggingFaceTokenizer.train_from_iterator(
                ["Once upon a time", "Today in science", "Write a story"], vocab_size=256
            )
            tokenizer.save(tokenizer_dir)
            tokenizer_json = resolve_tokenizer_json_path(tokenizer_dir)

            cfg = Config(
                input_bin=train_path,
                input_val_bin=val_path,
                depth=4,
                vocab_size=256,
                tokenizer_json=tokenizer_json,
                tokenizer_bos_id=bos,
                min_seq_len=seq_len,
                max_seq_len=seq_len,
                tokens_per_step=seq_len,
                micro_batch_size=1,
                target_train_tokens=seq_len,
                val_tokens=seq_len,
            )
            mesh = get_mesh(cfg)
            with mesh:
                params, _ = init_params(cfg, mesh)
            checkpoint_path = os.path.join(run_dir, "state_step000000.pkl")
            with open(checkpoint_path, "wb") as handle:
                pickle.dump(
                    {
                        "step": 0,
                        "params": jax.device_get(params),
                        "opt_state": None,
                        "config": cfg,
                    },
                    handle,
                )
            with open(os.path.join(run_dir, "latest_checkpoint.txt"), "w", encoding="utf-8") as handle:
                handle.write(checkpoint_path + "\n")

            report = evaluate_run(run_dir)
            self.assertTrue(np.isfinite(float(report["val_bpb"])))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "base_eval.json")))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "samples.txt")))
            self.assertGreater(report["samples"][0]["generated_token_count"], 0)

    def test_small_train_and_eval_step(self) -> None:
        cfg = Config(
            input_bin="train.bin",
            input_val_bin="val.bin",
            depth=4,
            vocab_size=4096,
            min_seq_len=16,
            max_seq_len=16,
            tokens_per_step=256,
            micro_batch_size=4,
            target_train_tokens=1024,
        )
        mesh = get_mesh(cfg)
        with mesh:
            params, rope = init_params(cfg, mesh)
            optimizer, opt_state = init_optimizer(cfg, params, mesh)
            embedding_out_sharding = get_data_parallel_sharding(cfg, mesh, ndim=3)
            token_bytes = precompute_token_bytes(cfg, mesh)
            batched_x = (jnp.arange(256, dtype=jnp.int32).reshape(4, 4, 16) % cfg.vocab_size)
            batched_y = (batched_x + 1) % cfg.vocab_size
            params, opt_state, metrics = train_step(
                cfg,
                params,
                rope,
                opt_state,
                optimizer,
                embedding_out_sharding,
                batched_x,
                batched_y,
            )
            bpb = eval_step(
                params,
                batched_x,
                batched_y,
                rope,
                token_bytes,
                cfg,
                embedding_out_sharding,
                get_data_parallel_sharding(cfg, mesh, ndim=1),
            )
        self.assertTrue(np.isfinite(float(metrics["loss"])))
        self.assertTrue(np.isfinite(float(bpb)))

    def test_job_scripts_have_valid_shell_syntax(self) -> None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for relpath in ("jobs/train_base.sh", "jobs/speedrun.sh"):
            subprocess.run(["bash", "-n", os.path.join(repo_root, relpath)], check=True)


if __name__ == "__main__":
    unittest.main()
