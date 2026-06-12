"""Final validation BPB and sample generation for the staged JAX speedrun."""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __package__ in {None, ""}:
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

import jaxchat.model as model_lib

model_lib.configure_jax_runtime()

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.tree_util import tree_map

from jaxchat import checkpoint as ckpt_lib
from jaxchat.model import (
    eval_step,
    get_data_parallel_sharding,
    get_eval_shape,
    get_mesh,
    get_weight_sharding,
    gpt_forward,
    load_dataset,
    precompute_rope,
    precompute_token_bytes,
)
from jaxchat.tokenizer import load_tokenizer


FIXED_PROMPTS = (
    "Once",
    "Today",
    "Science",
    "Write",
)
GENERATION_SEED = 42
GENERATION_TEMPERATURE = 0.8
GENERATION_TOP_K = 40
GENERATION_MAX_NEW_TOKENS = 128


class _EvalLogger:
    def msg(self, msg: str) -> None:
        print(msg)


def _load_latest_checkpoint(run_dir: str) -> tuple[str, dict]:
    """Load the freshest base-stage checkpoint via :mod:`jaxchat.checkpoint`.

    Falls back to a top-level ``latest_checkpoint.txt`` for legacy runs that
    predate the staged layout.
    """

    try:
        state = ckpt_lib.load_latest(run_dir, stage="base")
        return state.get("_checkpoint_path", ""), state
    except FileNotFoundError:
        legacy_marker = os.path.join(run_dir, "latest_checkpoint.txt")
        if not os.path.exists(legacy_marker):
            raise
        with open(legacy_marker, "r", encoding="utf-8") as handle:
            ckpt_path = handle.read().strip()
        state = ckpt_lib.load_path(ckpt_path)
        return ckpt_path, state


def _shard_params(params, config, mesh):
    weight_sharding = get_weight_sharding(config, mesh)
    return tree_map(lambda leaf: jax.device_put(jnp.asarray(leaf), weight_sharding), params)


def _top_k_sample(logits: np.ndarray, *, rng: np.random.Generator, top_k: int, temperature: float) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))
    k = max(1, min(int(top_k), logits.shape[0]))
    top_indices = np.argpartition(logits, -k)[-k:]
    top_logits = logits[top_indices] / float(temperature)
    top_logits = top_logits - np.max(top_logits)
    probs = np.exp(top_logits)
    probs = probs / np.sum(probs)
    return int(rng.choice(top_indices, p=probs))


def _generate_samples(params, precomputed_params, config, tokenizer, mesh) -> list[dict[str, object]]:
    rng = np.random.default_rng(GENERATION_SEED)
    # Match training: the embedding-gather output uses ``activation_sharding`` (the
    # 124m presets set it to (None, None, None) — replicated — because the per-call
    # batch dim is micro_batch_size, which is smaller than the device count and so
    # cannot be DP-sharded). Using get_data_parallel_sharding(ndim=3) here would put
    # 'dp' on the batch axis and fail (4 not divisible by 8).
    embedding_out_sharding = NamedSharding(mesh, P(*config.activation_sharding))
    bos_id = int(tokenizer.get_bos_token_id())
    samples: list[dict[str, object]] = []

    for prompt in FIXED_PROMPTS:
        prompt_ids = tokenizer.encode(prompt)
        token_ids = [bos_id] + list(prompt_ids)
        generated_ids: list[int] = []

        for _ in range(GENERATION_MAX_NEW_TOKENS):
            if len(token_ids) >= config.max_seq_len:
                break
            idx = jnp.asarray([token_ids], dtype=jnp.int32)
            logits = gpt_forward(params, idx, precomputed_params, config, embedding_out_sharding)
            next_token = _top_k_sample(
                np.asarray(jax.device_get(logits[0, -1])),
                rng=rng,
                top_k=GENERATION_TOP_K,
                temperature=GENERATION_TEMPERATURE,
            )
            token_ids.append(next_token)
            generated_ids.append(next_token)

        samples.append(
            {
                "prompt": prompt,
                "prompt_token_count": len(prompt_ids),
                "generated_token_count": len(generated_ids),
                "completion": tokenizer.decode(generated_ids),
            }
        )

    return samples


def evaluate_run(run_dir: str, *, generate_samples: bool = True) -> dict[str, object]:
    run_dir = os.path.abspath(run_dir)
    checkpoint_path, state = _load_latest_checkpoint(run_dir)
    config = state["config"]
    params = state["params"]
    step = int(state["step"])

    mesh = get_mesh(config)
    logger = _EvalLogger()
    # ``jax.set_mesh`` puts a *concrete* mesh in context (not just the abstract
    # one ``with mesh:`` provides) — eager ops in _generate_samples (e.g. the
    # SDPA mask's jnp.ones_like) need it, else: "Length of device assignment 1
    # is not equal to the size of the mesh 8 ... enter your jit into a mesh
    # context via jax.set_mesh".
    with mesh, jax.set_mesh(mesh):
        params = _shard_params(params, config, mesh)
        precomputed_params = precompute_rope(config, mesh)
        token_bytes = precompute_token_bytes(config, mesh)
        # See _generate_samples: embedding-gather output follows activation_sharding,
        # not a DP split of the (micro_batch_size,...) batch axis.
        embedding_out_sharding = NamedSharding(mesh, P(*config.activation_sharding))
        token_bytes_out_sharding = get_data_parallel_sharding(config, mesh, ndim=1)
        activation_sharding = NamedSharding(mesh, P(*config.activation_sharding))
        val_shape = get_eval_shape(config)

        jitted_eval_step = jit(
            eval_step,
            static_argnames=("config", "embedding_out_sharding", "token_bytes_out_sharding"),
        )
        dummy_x_shape = (val_shape[2], config.micro_batch_size, val_shape[0])
        dummy_x = jnp.zeros(dummy_x_shape, dtype=jnp.int32)
        dummy_y = jnp.zeros_like(dummy_x)
        dummy_x = jax.device_put(dummy_x, activation_sharding)
        dummy_y = jax.device_put(dummy_y, activation_sharding)
        compiled_eval_fn = jitted_eval_step.lower(
            params,
            dummy_x,
            dummy_y,
            precomputed_params,
            token_bytes,
            config,
            embedding_out_sharding,
            token_bytes_out_sharding,
        ).compile()

        val_loader = load_dataset(config, logger, mesh, is_training=False)
        val_loader.reset()
        bpb_accum = 0.0
        val_steps = 0
        for batched_x, batched_y in val_loader:
            bpb_accum += float(
                compiled_eval_fn(
                    params,
                    batched_x,
                    batched_y,
                    precomputed_params,
                    token_bytes,
                )
            )
            val_steps += 1
        if val_steps == 0:
            raise RuntimeError("Validation dataset produced no batches.")
        val_bpb = bpb_accum / val_steps

        logger.msg(f"val_bpb={val_bpb:.4f} (step {step})")
        # Eager autoregressive sampling re-traces per growing seq-len and dispatches
        # op-by-op across the 8-device mesh -> minutes.  It's a diagnostic; skip it
        # for ablation sweeps (--skip-generation) and never let it sink val_bpb.
        samples: list[dict[str, object]] = []
        if generate_samples:
            tokenizer = load_tokenizer(config.tokenizer_json)
            try:
                samples = _generate_samples(params, precomputed_params, config, tokenizer, mesh)
            except Exception as exc:
                logger.msg(f"[warn] sample generation failed ({type(exc).__name__}: {exc}); reporting val_bpb only.")
                samples = []

    samples_path = os.path.join(run_dir, "samples.txt")
    with open(samples_path, "w", encoding="utf-8") as handle:
        for idx, sample in enumerate(samples, start=1):
            handle.write(f"Prompt {idx}: {sample['prompt']}\n")
            handle.write(sample["completion"])
            handle.write("\n\n")

    report = {
        "run_dir": run_dir,
        "checkpoint_path": checkpoint_path,
        "step": step,
        "val_bpb": val_bpb,
        "generation_seed": GENERATION_SEED,
        "generation_temperature": GENERATION_TEMPERATURE,
        "generation_top_k": GENERATION_TOP_K,
        "generation_max_new_tokens": GENERATION_MAX_NEW_TOKENS,
        "samples_path": samples_path,
        "samples": samples,
    }
    output_path = os.path.join(run_dir, "base_eval.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the final staged JAX base-model checkpoint.")
    parser.add_argument("--run-dir", required=True, help="Deterministic run directory containing latest_checkpoint.txt")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Only compute val_bpb; skip the (slow) autoregressive sample generation.")
    args = parser.parse_args(argv)

    report = evaluate_run(args.run_dir, generate_samples=not args.skip_generation)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
