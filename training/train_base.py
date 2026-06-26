"""Active JAX training orchestration for the depth-driven base model stack.

Modernized with:
- Configurable LR schedule (linear, cosine, WSD)
- Sequence length + batch size scheduling
- Gradient clipping
- Z-loss regularization
- Weight tying (delayed untying)
- Dataset schedule reporting
- Shape schedule reporting
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import glob
import os
import sys
import time
from typing import Callable

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __package__ in {None, ""}:
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

import jaxchat.model as model_lib

model_lib.configure_jax_runtime()

import jax
import jax.numpy as jnp
from jax import jit
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.tree_util import tree_flatten_with_path, tree_map

from jaxchat import checkpoint as ckpt_lib
from jaxchat import wandb_log as wb
from jaxchat.fa3 import backend_summary_for_config
from jaxchat.model import (
    Config,
    Logger,
    Pytree,
    count_parameters,
    estimate_mfu_proxy,
    eval_step,
    expected_parameter_breakdown,
    format_parameter_breakdown,
    format_shape_summary,
    get_data_parallel_sharding,
    get_eval_shape,
    get_mesh,
    get_train_shape_counts,
    init_optimizer,
    init_params,
    load_dataset,
    parameter_breakdown_from_params,
    precompute_token_bytes,
    train_step,
    maybe_untie_weights,
)
from jaxchat.optimizer import get_lr_scale
from jaxchat.schedules import get_shape_for_step


def _hours_for_throughput(total_tokens: int, tok_s: float) -> float:
    return float(total_tokens) / max(tok_s, 1.0) / 3600.0


def _reported_param_count(breakdown: dict[str, int]) -> int:
    return int(breakdown.get("total", 0) - breakdown.get("scalars", 0))


def _estimated_flops_per_token_for_report(config: Config, breakdown: dict[str, int]) -> float:
    # Match the nanochat-style accounting used by the target report: dense params
    # plus the token-feature table if the architecture excludes it from params.
    param_count = _reported_param_count(breakdown)
    if not config.bigram_hash_embed and config.vocab_size == 65_536 and config.depth == 20:
        param_count += int(config.bigram_hash_buckets * config.d_model)
    return 6.0 * float(param_count)


def log_nanochat_style_startup(config: Config, breakdown: dict[str, int], logger: Logger) -> None:
    seq_len = int(config.max_seq_len)
    global_batch = int(config.tokens_per_step // max(seq_len, 1))
    per_rank_batch = max(global_batch // max(jax.device_count(), 1), 1)
    grad_accum = max(global_batch // max(config.micro_batch_size, 1), 1)
    reported_params = _reported_param_count(breakdown)
    flops_per_token = _estimated_flops_per_token_for_report(config, breakdown)
    total_flops = flops_per_token * float(config.actual_train_tokens)
    ratio = float(config.actual_train_tokens) / max(float(reported_params), 1.0)
    lr_scale = 1.0 / (float(config.d_model) / 768.0) ** 0.5

    for line in (
        f"Vocab size: {config.vocab_size:,}",
        f"num_layers: {config.depth}",
        f"model_dim: {config.d_model}",
        f"num_heads: {config.n_heads}",
        f"num_kv_heads: {config.n_kv_heads}",
        f"Tokens / micro-batch / rank: {per_rank_batch} x {seq_len} = {per_rank_batch * seq_len:,}",
        f"Tokens / micro-batch: {config.tokens_per_step:,}",
        f"Total batch size {config.tokens_per_step:,} => gradient accumulation steps: {grad_accum}",
        f"Number of parameters: {reported_params:,}",
        f"Estimated FLOPs per token: {flops_per_token:.6e}",
        f"Calculated number of iterations from target data:param ratio: {config.n_train_iters:,}",
        f"Total number of training tokens: {config.actual_train_tokens:,}",
        f"Tokens : Params ratio: {ratio:.2f}",
        f"Total training FLOPs estimate: {total_flops:.6e}",
        f"Scaling the LR for the AdamW parameters ∝1/√({config.d_model}/768) = {lr_scale:.6f}",
    ):
        logger.msg(line)


def log_muon_group_summary(params: Pytree, logger: Logger) -> None:
    groups: dict[tuple[int, ...], int] = {}
    for path, leaf in tree_flatten_with_path(params)[0]:
        names = model_lib.path_to_names(path)
        if names and names[0] in {"wte", "lm_head", "value_embeds", "resid_lambdas", "x0_lambdas", "skip_lambdas"}:
            continue
        if getattr(leaf, "ndim", 0) == 2:
            groups[tuple(int(x) for x in leaf.shape)] = groups.get(tuple(int(x) for x in leaf.shape), 0) + 1
    for shape, count in sorted(groups.items()):
        logger.msg(f"Muon: Grouping {count} params of shape torch.Size({list(shape)})")


def run_evaluation(
    step: int,
    config: Config,
    params: Pytree,
    val_loader,
    precomputed_params: Pytree,
    token_bytes: jax.Array,
    logger: Logger,
    compiled_eval_fn: Callable,
):
    eval_start = time.perf_counter()
    logger.msg(f"Running validation for step {step}...")
    val_loader.reset()
    bpb_accum = 0.0
    val_steps = 0
    for batched_x, batched_y in val_loader:
        bpb = compiled_eval_fn(
            params,
            batched_x,
            batched_y,
            precomputed_params,
            token_bytes,
        )
        bpb_accum += float(bpb)
        val_steps += 1
    if val_steps == 0:
        logger.msg("Validation loader was empty, no validation was run.")
        return time.perf_counter() - eval_start
    final_bpb = bpb_accum / val_steps
    logger.log({"step": step, "val_bpb": final_bpb})
    wb.wandb_log({"step": step, "val_bpb": final_bpb})
    logger.msg(f"Step {step:05d} | Validation bpb: {final_bpb:.4f}")
    logger.msg(f"Validation finished for step {step}. BPB: {final_bpb:.4f}")
    return time.perf_counter() - eval_start


def validate_training_assets(config: Config) -> None:
    missing = []
    if not glob.glob(config.input_bin):
        missing.append(f"Training shard glob matched no files: {config.input_bin}")
    if not glob.glob(config.input_val_bin):
        missing.append(f"Validation shard glob matched no files: {config.input_val_bin}")
    if missing:
        details = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"Missing training assets:\n{details}")


def log_config_features(config: Config, logger: Logger):
    """Log all active modernization features for debugging."""
    features = []
    if config.lr_schedule != "linear":
        features.append(f"lr_schedule={config.lr_schedule}")
    if config.max_grad_norm > 0:
        features.append(f"max_grad_norm={config.max_grad_norm}")
    if config.z_loss_coeff > 0:
        features.append(f"z_loss={config.z_loss_coeff}")
    if config.weight_tying != "none":
        features.append(f"weight_tying={config.weight_tying}")
        if config.weight_tying == "delayed":
            features.append(f"untie_at={config.untie_at_step}")
    if config.init_style != "default":
        features.append(f"init={config.init_style}")
    if config.scale_embedding:
        features.append("scale_embedding")
    if config.normalize_logits:
        features.append("normalize_logits")
    if config.logit_cap_style != "sigmoid":
        features.append(f"logit_cap={config.logit_cap_style}")
    if config.n_kv_heads > 0 and config.n_kv_heads < config.n_heads:
        features.append(f"GQA(kv={config.n_kv_heads})")
    if config.use_long_short_attention:
        features.append("long_short_attn")
    if config.bigram_hash_embed:
        features.append("bigram_hash")
    if config.pko_enabled:
        features.append("PKO")
    if config.cross_document_mask:
        features.append("cross_doc_mask")
    if config.skip_connections:
        features.append(f"skip_conns={config.skip_connections}")
    if config.layer_drop_prob > 0:
        features.append(f"layer_drop={config.layer_drop_prob}")
    if config.recompute_layers != "none":
        features.append(f"recompute={config.recompute_layers}")
    if config.optimizer != "muon_adamw":
        features.append(f"optimizer={config.optimizer}")
    if config.joint_schedule_points or config.seq_schedule_points or config.batch_schedule_points:
        features.append("scheduled_shapes")
    if features:
        logger.msg("Active features: " + " | ".join(features))


def train_loop(
    config: Config,
    *,
    preset_name: str = "default",
    run_dir: str | None = None,
    resume: bool = False,
    smoke_iters: int | None = None,
):
    wall_start = time.perf_counter()
    validate_training_assets(config)
    logger = Logger(run_dir=run_dir)
    if config.tokenizer_json and not os.path.exists(config.tokenizer_json):
        logger.msg(
            f"Tokenizer JSON not found at {config.tokenizer_json}; "
            "falling back to approximate token-byte accounting."
        )
    mesh = get_mesh(config)

    if smoke_iters is not None and smoke_iters > 0:
        config = dataclasses.replace(
            config,
            n_train_iters=int(smoke_iters),
            n_warmup_iters=min(config.n_warmup_iters, max(int(smoke_iters) // 8, 1)),
            target_train_tokens=int(smoke_iters) * config.tokens_per_step,
        )
        logger.msg(f"Smoke override: n_train_iters={config.n_train_iters}")

    if logger.is_master:
        wb.wandb_init(
            stage="base",
            run_name=f"base-{preset_name}-{logger.run_id}",
            config_dict=wb.config_from_dataclass(config, {"preset": preset_name}),
        )

    resumed_state: dict | None = None
    start_step = 0
    if resume and logger.logdir is not None:
        try:
            resumed_state = ckpt_lib.load_latest(logger.logdir, stage="base")
        except FileNotFoundError:
            logger.msg("Resume requested but no base checkpoint found; starting from scratch.")
            resumed_state = None

    with mesh:
        logger.msg(f"Attention runtime: {backend_summary_for_config(config, mesh=mesh)}")
        params, precomputed_params = init_params(config, mesh)
        if resumed_state is not None:
            weight_sharding = model_lib.get_weight_sharding(config, mesh)
            params = tree_map(
                lambda leaf: jax.device_put(jnp.asarray(leaf), weight_sharding),
                resumed_state["params"],
            )
            start_step = int(resumed_state["step"]) + 1
            logger.msg(
                f"Resumed from step {resumed_state['step']} "
                f"({resumed_state.get('_checkpoint_path', '<unknown>')})"
            )
        expected_breakdown = expected_parameter_breakdown(config)
        actual_breakdown = parameter_breakdown_from_params(params)
        if actual_breakdown != expected_breakdown:
            logger.msg(
                "Parameter breakdown mismatch (non-fatal, likely due to active features).\n"
                f"expected={expected_breakdown}\nactual={actual_breakdown}"
            )

        token_bytes = precompute_token_bytes(config, mesh)
        optimizer, opt_state = init_optimizer(config, params, mesh)
        if resumed_state is not None and resumed_state.get("opt_state") is not None:
            opt_state = tree_map(jnp.asarray, resumed_state["opt_state"])
            logger.msg("Restored optimizer state from checkpoint.")
        param_count = count_parameters(params)
        train_shape_counts = get_train_shape_counts(config)
        val_shape = get_eval_shape(config)

        logger.msg(
            f"Preset: {preset_name} | depth: {config.depth} | params: {param_count:,} | "
            f"tokens_per_step: {config.tokens_per_step:,} | "
            f"optimizer: {config.optimizer}"
        )
        logger.msg(f"Param breakdown: {format_parameter_breakdown(actual_breakdown)}")
        log_nanochat_style_startup(config, actual_breakdown, logger)
        log_muon_group_summary(params, logger)
        _embed = actual_breakdown.get("wte", 0) + actual_breakdown.get("lm_head", 0) + actual_breakdown.get("value_embeds", 0)
        _nonembed = max(param_count - _embed, 1)
        _xfmr = max(actual_breakdown.get("transformer_matrices", _nonembed), 1)
        logger.msg(
            f"Param split: embedding={_embed:,} ({100.0*_embed/max(param_count,1):.0f}%) | "
            f"non-embedding={_nonembed:,} | transformer-matrices={_xfmr:,}"
        )
        logger.msg(
            f"Target train tokens: {config.target_train_tokens:,} | "
            f"Scheduled train tokens: {config.actual_train_tokens:,} | "
            f"steps: {config.n_train_iters:,} | "
            f"lr_schedule: {config.lr_schedule}"
        )
        logger.msg(
            f"Tokens/param: total={config.actual_train_tokens/max(param_count,1):.1f} | "
            f"non-embedding={config.actual_train_tokens/_nonembed:.1f} | "
            f"transformer-matrices={config.actual_train_tokens/_xfmr:.1f}"
        )
        logger.msg(
            "ETA range at 160k-240k tok/s: "
            f"{_hours_for_throughput(config.actual_train_tokens, 240_000):.2f}h - "
            f"{_hours_for_throughput(config.actual_train_tokens, 160_000):.2f}h"
        )
        logger.msg(f"Training shapes: {format_shape_summary(train_shape_counts)}")
        logger.msg(
            f"Validation shape: (seq_len={val_shape[0]}, batch={val_shape[1]}, "
            f"grad_accum={val_shape[2]}) | val_tokens: {config.val_tokens:,}"
        )

        # Log all active modern features
        log_config_features(config, logger)

        # Use activation_sharding for embedding output (matches data layout)
        # When micro_batch < num_devices, DP sharding would fail on the batch dim
        activation_pspec = P(*config.activation_sharding)
        embedding_out_sharding = NamedSharding(mesh, activation_pspec)
        token_bytes_out_sharding = get_data_parallel_sharding(config, mesh, ndim=1)
        jitted_train_step = jit(
            train_step,
            static_argnames=("config", "optimizer", "embedding_out_sharding"),
            donate_argnums=(1, 3),
        )
        jitted_eval_step = jit(
            eval_step,
            static_argnames=("config", "embedding_out_sharding", "token_bytes_out_sharding"),
        )

        logger.msg("Determining all unique training shapes...")
        train_shapes = set(train_shape_counts)
        train_shapes.add(val_shape)
        logger.msg(f"Total unique shapes to compile: {len(train_shapes)}")
        logger.msg("Starting Ahead-of-Time (AOT) compilation for all shapes...")
        compiled_train_steps = {}
        compiled_eval_fn = None
        activation_sharding = NamedSharding(mesh, P(*config.activation_sharding))
        compile_start = time.perf_counter()
        for seq_len, batch_size, n_grad_accum_steps in sorted(list(train_shapes)):
            shape_key = (seq_len, batch_size, n_grad_accum_steps)
            logger.msg(
                f"AOT compiling for seq_len={seq_len}, B={batch_size}, grad_accum={n_grad_accum_steps}..."
            )
            dummy_x_shape = (n_grad_accum_steps, config.micro_batch_size, seq_len)
            dummy_x = jnp.zeros(dummy_x_shape, dtype=jnp.int32)
            dummy_y = jnp.zeros_like(dummy_x)
            dummy_x = jax.device_put(dummy_x, activation_sharding)
            dummy_y = jax.device_put(dummy_y, activation_sharding)
            compiled_fn = jitted_train_step.lower(
                config,
                params,
                precomputed_params,
                opt_state,
                optimizer,
                embedding_out_sharding,
                dummy_x,
                dummy_y,
            ).compile()
            compiled_train_steps[shape_key] = compiled_fn
            if shape_key == val_shape:
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
        compile_s = time.perf_counter() - compile_start
        logger.msg("AOT compilation finished for all function variants.")
        logger.msg(f"compile_s: {compile_s:.3f}")

        data_open_start = time.perf_counter()
        train_loader = load_dataset(config, logger, mesh, is_training=True)
        val_loader = load_dataset(config, logger, mesh, is_training=False)
        data_open_s = time.perf_counter() - data_open_start
        logger.msg(f"data_open_s: {data_open_s:.3f}")

        eval_s_total = 0.0
        checkpoint_s_in_loop = 0.0
        if config.eval_at_start and start_step == 0:
            run_evaluation(
                0,
                config,
                params,
                val_loader,
                precomputed_params,
                token_bytes,
                logger,
                compiled_eval_fn,
            )
            logger.flush()
        logger.msg("Starting training...")
        train_loop_start = time.perf_counter()
        last_step_time = train_loop_start

        def _save_base(step_idx: int) -> str:
            return ckpt_lib.save(
                stage="base",
                step=step_idx,
                params=params,
                opt_state=opt_state,
                config=config,
                run_dir=logger.logdir or run_dir or ".",
                tokenizer_path=config.tokenizer_json,
                rng_seed=config.seed,
            )

        for step in range(start_step, config.n_train_iters):
            batched_x, batched_y = next(train_loader)

            # --- Resolve shape key for this step ---
            n_grad_accum, _, seq_len = batched_x.shape
            batch_size = n_grad_accum * config.micro_batch_size
            current_shape_key = (seq_len, batch_size, n_grad_accum)
            aot_train_fn = compiled_train_steps[current_shape_key]

            # --- Delayed weight tying: untie at the scheduled step ---
            if config.weight_tying == "delayed" and step == config.untie_at_step and "lm_head" in params:
                logger.msg(f"Untying weights at step {step}...")
                # Re-seed lm_head with an independent copy of wte^T (same shape as the
                # existing lm_head leaf, so the AOT-compiled train step still applies).
                # Optimizer state for lm_head is kept as-is.
                wte_val = params["wte"]
                lm_head_seed = jnp.transpose(wte_val, (1, 0))
                if lm_head_seed.shape != params["lm_head"].shape:
                    lm_head_seed = lm_head_seed.reshape(params["lm_head"].shape)
                lm_head_seed = jax.device_put(lm_head_seed, params["lm_head"].sharding)
                params = {k: (lm_head_seed if k == "lm_head" else v) for k, v in params.items()}
                logger.msg("Weights untied (lm_head re-seeded from wte, now independent).")

            params, opt_state, metrics = aot_train_fn(
                params,
                precomputed_params,
                opt_state,
                batched_x,
                batched_y,
            )

            now = time.perf_counter()
            dt = now - last_step_time
            last_step_time = now
            if config.log_every > 0 and (step % config.log_every == 0 or step == config.n_train_iters - 1):
                metrics_view = {k: float(v) for k, v in metrics.items()}
                lr_multiplier = float(get_lr_scale(
                    jnp.asarray(step, dtype=jnp.int32),
                    config.n_warmup_iters,
                    config.n_warmdown_iters,
                    config.n_train_iters,
                    config.lr_schedule,
                ))
                tok_s = float(config.tokens_per_step) / max(dt, 1e-9)
                mfu = estimate_mfu_proxy(param_count, config, dt) * 100.0
                elapsed_m = (now - train_loop_start) / 60.0
                percent = 100.0 * float(step) / max(float(config.n_train_iters), 1.0)
                logger.msg(
                    f"step {step:05d}/{config.n_train_iters} ({percent:.2f}%) | "
                    f"loss: {metrics_view.get('loss', float('nan')):.6f} | "
                    f"lrm: {lr_multiplier:.2f} | dt: {dt * 1000.0:.2f}ms | "
                    f"tok/sec: {tok_s:,.0f} | mfu: {mfu:.2f} | total time: {elapsed_m:.2f}m"
                )
                wb.wandb_log({"step": step, "lrm": lr_multiplier, "dt_ms": dt * 1000.0,
                              "tok_s": tok_s, "mfu": mfu, "elapsed_m": elapsed_m, **metrics_view})
            if step > 0 and step % config.val_loss_every == 0:
                eval_s_total += run_evaluation(
                    step,
                    config,
                    params,
                    val_loader,
                    precomputed_params,
                    token_bytes,
                    logger,
                    compiled_eval_fn,
                )
                logger.flush()
            if (
                config.save_every > 0
                and step > 0
                and step % config.save_every == 0
                and step < config.n_train_iters - 1
            ):
                checkpoint_start = time.perf_counter()
                save_path = _save_base(step)
                logger.msg(f"Saved checkpoint to {save_path}")
                checkpoint_s_in_loop += time.perf_counter() - checkpoint_start

        jax.block_until_ready(params)
        train_loop_s = time.perf_counter() - train_loop_start
        final_checkpoint_start = time.perf_counter()
        final_path = _save_base(config.n_train_iters - 1)
        logger.msg(f"Saved final checkpoint to {final_path}")
        final_checkpoint_s = time.perf_counter() - final_checkpoint_start
        checkpoint_s_total = checkpoint_s_in_loop + final_checkpoint_s
        total_wall_s = time.perf_counter() - wall_start
        effective_train_s = max(train_loop_s - eval_s_total - checkpoint_s_in_loop, 0.0)
        steady_state_step_s = effective_train_s / max(config.n_train_iters, 1)
        aggregate_tok_s = (
            config.actual_train_tokens / effective_train_s if effective_train_s > 0 else 0.0
        )
        mfu_proxy = estimate_mfu_proxy(param_count, config, steady_state_step_s)
        timing_metrics = {
            "compile_s": compile_s,
            "data_open_s": data_open_s,
            "train_loop_s": train_loop_s,
            "eval_s_total": eval_s_total,
            "checkpoint_s": checkpoint_s_total,
            "total_wall_s": total_wall_s,
            "steady_state_step_s": steady_state_step_s,
            "aggregate_tok_s": aggregate_tok_s,
            "mfu_proxy": mfu_proxy,
        }

        logger.msg("Training finished.")
        logger.log(timing_metrics)
        wb.wandb_log(timing_metrics)
        wb.wandb_finish()
        logger.msg(
            "Timing summary | "
            f"compile_s: {compile_s:.3f} | "
            f"data_open_s: {data_open_s:.3f} | "
            f"train_loop_s: {train_loop_s:.3f} | "
            f"eval_s_total: {eval_s_total:.3f} | "
            f"checkpoint_s: {checkpoint_s_total:.3f} | "
            f"total_wall_s: {total_wall_s:.3f} | "
            f"steady_state_step_s: {steady_state_step_s:.6f} | "
            f"aggregate_tok_s: {aggregate_tok_s:.1f} | "
            f"mfu_proxy: {mfu_proxy:.4f}"
        )
        logger.flush()


from jaxchat.presets import (
    DEFAULT_CONFIG,
    FINEWEB_32K_DIR,
    FINEWEB_TOKENIZER_JSON,
    FINEWEB_TRAIN_GLOB,
    FINEWEB_VAL_BIN,
    PRESET_1P384B_DEPTH24,
    PRESETS,
    SMOKE,
)


def build_config(args) -> tuple[Config, str]:
    config = PRESETS[args.preset]
    preset_name = args.preset
    replace_kwargs = {}
    if args.depth is not None:
        replace_kwargs["depth"] = args.depth
        preset_name = f"depth{args.depth}"
    if args.input_bin:
        replace_kwargs["input_bin"] = args.input_bin
    if args.input_val_bin:
        replace_kwargs["input_val_bin"] = args.input_val_bin
    if args.tokenizer_json is not None:
        replace_kwargs["tokenizer_json"] = args.tokenizer_json
    if args.optimizer is not None:
        replace_kwargs["optimizer"] = args.optimizer
    if args.lr_schedule is not None:
        replace_kwargs["lr_schedule"] = args.lr_schedule
    if args.weight_tying is not None:
        replace_kwargs["weight_tying"] = args.weight_tying
    if replace_kwargs:
        config = dataclasses.replace(config, **replace_kwargs)
    return config, preset_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the JAX depth-driven base model")
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="default",
        help="Training preset to run.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Depth override. d_model derives as depth*64 and n_heads as d_model/128.",
    )
    parser.add_argument("--input-bin", default="", help="Override training shard glob.")
    parser.add_argument("--input-val-bin", default="", help="Override validation shard glob.")
    parser.add_argument(
        "--tokenizer-json",
        default=None,
        help="Override tokenizer JSON used for token-byte accounting.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Explicit output directory for train.log and checkpoints.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If --run-dir already has a base checkpoint, resume from it.",
    )
    parser.add_argument(
        "--smoke-iters",
        type=int,
        default=None,
        help="Override n_train_iters for a quick wiring test.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("muon_adamw", "normuon", "soap"),
        default=None,
        help="Optimizer to use (default: muon_adamw).",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("linear", "cosine", "wsd"),
        default=None,
        help="LR schedule type.",
    )
    parser.add_argument(
        "--weight-tying",
        choices=("none", "full", "delayed"),
        default=None,
        help="Weight tying mode.",
    )
    args = parser.parse_args(argv)
    config, preset_name = build_config(args)
    train_loop(
        config,
        preset_name=preset_name,
        run_dir=args.run_dir,
        resume=args.resume,
        smoke_iters=args.smoke_iters,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
