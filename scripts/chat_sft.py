"""Supervised fine-tuning entry point.

Loads the base checkpoint produced by ``scripts.base_train``, switches the loss
to a masked CE over assistant tokens (via :func:`jaxchat.model.sft_train_step`),
and saves a new checkpoint under ``stage="sft"``.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jaxchat.model as model_lib  # noqa: E402

model_lib.configure_jax_runtime()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import jit  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402
from jax.tree_util import tree_map  # noqa: E402

from jaxchat import checkpoint as ckpt_lib  # noqa: E402
from jaxchat import wandb_log as wb  # noqa: E402
from jaxchat.model import (  # noqa: E402
    Logger,
    count_parameters,
    get_mesh,
    get_weight_sharding,
    init_optimizer,
    init_params,
    precompute_rope,
    sft_train_step,
)
from jaxchat.tokenizer import load_tokenizer  # noqa: E402


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _render_examples(rows: list[dict], tokenizer, *, max_seq_len: int):
    pad_id = int(tokenizer.get_bos_token_id())
    rendered: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for row in rows:
        ids, mask = tokenizer.render_conversation(row, max_tokens=max_seq_len + 1)
        if len(ids) < 2 or sum(mask) == 0:
            continue
        ids = ids[: max_seq_len + 1]
        mask = mask[: max_seq_len + 1]
        x = np.full(max_seq_len, pad_id, dtype=np.int32)
        y = np.full(max_seq_len, pad_id, dtype=np.int32)
        m = np.zeros(max_seq_len, dtype=np.int32)
        n = len(ids) - 1
        if n <= 0:
            continue
        x[:n] = np.asarray(ids[:-1], dtype=np.int32)[:n]
        y[:n] = np.asarray(ids[1:], dtype=np.int32)[:n]
        # mask aligns with the next-token prediction, so use mask shifted.
        m[:n] = np.asarray(mask[1:], dtype=np.int32)[:n]
        rendered.append((x, y, m))
    return rendered


def _batch_iter(rendered, *, batch_size: int, n_grad_accum: int, seed: int):
    rng = np.random.default_rng(seed)
    full = batch_size * n_grad_accum
    if not rendered:
        raise RuntimeError("No SFT examples rendered; check that the data has assistant turns.")
    while True:
        idxs = rng.integers(0, len(rendered), size=full)
        xs = np.stack([rendered[i][0] for i in idxs], axis=0)
        ys = np.stack([rendered[i][1] for i in idxs], axis=0)
        ms = np.stack([rendered[i][2] for i in idxs], axis=0)
        xs = xs.reshape(n_grad_accum, batch_size, -1)
        ys = ys.reshape(n_grad_accum, batch_size, -1)
        ms = ms.reshape(n_grad_accum, batch_size, -1)
        yield xs, ys, ms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Supervised fine-tune a jaxchat base checkpoint.")
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--parent-stage", choices=("base", "sft"), default="base")
    parser.add_argument("--sft-data", required=True, help="JSONL with one {messages: [...]} per line.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-iters", type=int, default=200)
    parser.add_argument("--smoke-iters", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--n-grad-accum", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lr-scale", type=float, default=0.2, help="Multiplier on base LRs (Muon + Adam).")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer-json", default=None)
    args = parser.parse_args(argv)

    if args.smoke_iters is not None and args.smoke_iters > 0:
        args.n_iters = int(args.smoke_iters)

    state = ckpt_lib.load_latest(args.base_run_dir, stage=args.parent_stage)
    base_config = state["config"]
    parent_meta = {
        "stage": args.parent_stage,
        "ckpt_path": state.get("_checkpoint_path", ""),
        "step": int(state.get("step", 0)),
    }

    micro_batch_size = max(1, args.micro_batch_size)
    n_grad_accum = max(1, args.n_grad_accum)
    batch_size = micro_batch_size * n_grad_accum
    max_seq_len = max(64, args.max_seq_len)
    tokens_per_step = batch_size * max_seq_len

    sft_config = dataclasses.replace(
        base_config,
        min_seq_len=max_seq_len,
        max_seq_len=max_seq_len,
        tokens_per_step=tokens_per_step,
        micro_batch_size=micro_batch_size,
        target_train_tokens=tokens_per_step * args.n_iters,
        n_train_iters=args.n_iters,
        n_warmup_iters=max(1, args.n_iters // 20),
        f_warmdown_iters=0.5,
        val_loss_every=max(args.n_iters + 1, 1),
        save_every=args.save_every,
        embed_lr_base=base_config.embed_lr_base * args.lr_scale,
        lm_head_lr_base=base_config.lm_head_lr_base * args.lr_scale,
        muon_base_lr=base_config.muon_base_lr * args.lr_scale,
        scalar_resid_lr=base_config.scalar_resid_lr * args.lr_scale,
        scalar_x0_lr=base_config.scalar_x0_lr * args.lr_scale,
        weight_decay_base=0.0,
    )

    logger = Logger(run_dir=args.run_dir)
    if logger.is_master:
        wb.wandb_init(
            stage="sft",
            run_name=f"sft-{logger.run_id}",
            config_dict=wb.config_from_dataclass(
                sft_config,
                {"n_iters": args.n_iters, "lr_scale": args.lr_scale, "parent": parent_meta},
            ),
        )
    mesh = get_mesh(sft_config)

    tok_path = args.tokenizer_json or state.get("tokenizer_path") or sft_config.tokenizer_json
    if not tok_path:
        raise RuntimeError("No tokenizer path found; pass --tokenizer-json.")
    tokenizer = load_tokenizer(tok_path)

    logger.msg(f"Loading SFT data from {args.sft_data}")
    rows = _load_jsonl(args.sft_data)
    rendered = _render_examples(rows, tokenizer, max_seq_len=max_seq_len)
    logger.msg(f"Rendered {len(rendered)} SFT examples (skipped {len(rows) - len(rendered)}).")

    with mesh:
        weight_sharding = get_weight_sharding(sft_config, mesh)
        params, _ = init_params(sft_config, mesh)
        params = tree_map(lambda leaf: jax.device_put(jnp.asarray(leaf), weight_sharding), state["params"])
        precomputed_params = precompute_rope(sft_config, mesh)
        optimizer, opt_state = init_optimizer(sft_config, params, mesh)
        param_count = count_parameters(params)
        logger.msg(f"SFT preset | params: {param_count:,} | tokens_per_step: {tokens_per_step:,} | iters: {args.n_iters}")

        # Match train_base: embedding-output activations follow
        # ``config.activation_sharding`` (replicated batch on the 124m presets)
        # so micro_batch_size < dp doesn't trip a divisibility error.
        activation_sharding = NamedSharding(mesh, P(*sft_config.activation_sharding))
        embedding_out_sharding = activation_sharding

        jitted = jit(
            sft_train_step,
            static_argnames=("config", "optimizer", "embedding_out_sharding"),
            donate_argnums=(1, 3),
        )
        dummy = jnp.zeros((n_grad_accum, micro_batch_size, max_seq_len), dtype=jnp.int32)
        dummy = jax.device_put(dummy, activation_sharding)
        compile_start = time.perf_counter()
        compiled = jitted.lower(
            sft_config,
            params,
            precomputed_params,
            opt_state,
            optimizer,
            embedding_out_sharding,
            dummy,
            dummy,
            dummy,
        ).compile()
        logger.msg(f"sft compile_s: {time.perf_counter() - compile_start:.3f}")

        gen = _batch_iter(rendered, batch_size=micro_batch_size, n_grad_accum=n_grad_accum, seed=args.seed)
        train_start = time.perf_counter()
        for step in range(args.n_iters):
            xs, ys, ms = next(gen)
            xs_d = jax.device_put(jnp.asarray(xs), activation_sharding)
            ys_d = jax.device_put(jnp.asarray(ys), activation_sharding)
            ms_d = jax.device_put(jnp.asarray(ms), activation_sharding)
            params, opt_state, metrics = compiled(params, precomputed_params, opt_state, xs_d, ys_d, ms_d)
            if step % 10 == 9:
                metrics_view = {k: float(v) for k, v in metrics.items()}
                logger.log({"step": step, "time": datetime.datetime.now()} | metrics_view)
                wb.wandb_log({"step": step, **metrics_view})
            if args.save_every > 0 and step > 0 and step % args.save_every == 0 and step < args.n_iters - 1:
                save_path = ckpt_lib.save(
                    stage="sft",
                    step=step,
                    params=params,
                    opt_state=opt_state,
                    config=sft_config,
                    run_dir=logger.logdir,
                    tokenizer_path=tok_path,
                    rng_seed=args.seed,
                    parent=parent_meta,
                )
                logger.msg(f"Saved checkpoint to {save_path}")

        jax.block_until_ready(params)
        logger.msg(f"sft train_loop_s: {time.perf_counter() - train_start:.3f}")
        final_path = ckpt_lib.save(
            stage="sft",
            step=args.n_iters - 1,
            params=params,
            opt_state=opt_state,
            config=sft_config,
            run_dir=logger.logdir,
            tokenizer_path=tok_path,
            rng_seed=args.seed,
            parent=parent_meta,
        )
        logger.msg(f"Saved final SFT checkpoint to {final_path}")
        logger.flush()
        wb.wandb_finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
