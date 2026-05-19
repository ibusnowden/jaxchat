"""Group-Relative Policy Optimization (GRPO) entry point.

A minimal, single-update-per-batch flavor: importance-ratio clipping is
unnecessary because we take exactly one SGD step per sampled batch (so
``ratio == 1`` on-policy).  The objective reduces to::

    L = -E[A * log pi(a|s)]  +  beta * KL(pi || pi_ref)

where ``A`` is the group-relative advantage over ``G`` completions per prompt
and ``pi_ref`` is the SFT model frozen at start of training.

Sampling is autoregressive via :class:`jaxchat.engine.Engine` and is the
slowest part of each step.  Use small ``--m-prompts`` and ``--g-rollouts``
when smoke-testing.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime
import json
import os
import random
import sys
import time
from functools import partial

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jaxchat.model as model_lib  # noqa: E402

model_lib.configure_jax_runtime()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import jit, value_and_grad  # noqa: E402
from jax.tree_util import tree_map  # noqa: E402

from jaxchat import checkpoint as ckpt_lib  # noqa: E402
from jaxchat import wandb_log as wb  # noqa: E402
from jaxchat.engine import Engine  # noqa: E402
from jaxchat.model import (  # noqa: E402
    Logger,
    count_parameters,
    get_mesh,
    get_weight_sharding,
    gpt_forward,
    init_optimizer,
    init_params,
    precompute_rope,
)
from jaxchat.tokenizer import load_hf_tokenizer  # noqa: E402
from tasks import gsm8k  # noqa: E402


def rl_loss_fn(params, ref_params, batch, precomputed_params, config, embedding_out_sharding, kl_beta, clip_eps):
    idx, labels, mask, adv = batch
    logits = gpt_forward(params, idx, precomputed_params, config, embedding_out_sharding)
    logp_full = jax.nn.log_softmax(logits, axis=-1)
    logp_token = jnp.take_along_axis(logp_full, labels[..., None], axis=-1)[..., 0]

    ref_logits = gpt_forward(ref_params, idx, precomputed_params, config, embedding_out_sharding)
    ref_logp_full = jax.nn.log_softmax(ref_logits, axis=-1)
    ref_logp_token = jnp.take_along_axis(ref_logp_full, labels[..., None], axis=-1)[..., 0]

    mask_f = mask.astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(mask_f), 1.0)
    advantages = adv[:, None].astype(jnp.float32)

    # Importance ratio with clipping (standard PPO/GRPO).
    ratio = jnp.exp(logp_token - ref_logp_token)
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    pg = -(jnp.minimum(advantages * ratio, advantages * clipped_ratio) * mask_f).sum() / denom
    kl = ((logp_token - ref_logp_token) * mask_f).sum() / denom
    return pg + kl_beta * kl, {"pg": pg, "kl": kl, "approx_kl": float(kl), "ratio_mean": float(jnp.mean(ratio))}


@partial(jit, static_argnames=("optimizer", "config", "embedding_out_sharding", "kl_beta", "clip_eps"))
def rl_train_step(
    config,
    params,
    ref_params,
    precomputed_params,
    opt_state,
    optimizer,
    embedding_out_sharding,
    idx,
    labels,
    mask,
    adv,
    kl_beta,
    clip_eps,
):
    (loss, aux), grads = value_and_grad(rl_loss_fn, has_aux=True)(
        params, ref_params, (idx, labels, mask, adv), precomputed_params, config, embedding_out_sharding, kl_beta, clip_eps,
    )
    new_params, new_opt_state = optimizer.update(grads, params, opt_state)
    return new_params, new_opt_state, {"loss": loss, **aux}


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_padded_batch(
    *,
    prompt_ids_list: list[list[int]],
    gen_ids_list: list[list[int]],
    advantages: list[float],
    pad_id: int,
    seq_len: int,
):
    B = len(prompt_ids_list)
    idx = np.full((B, seq_len), pad_id, dtype=np.int32)
    labels = np.full((B, seq_len), pad_id, dtype=np.int32)
    mask = np.zeros((B, seq_len), dtype=np.int32)
    for i, (prompt, gen) in enumerate(zip(prompt_ids_list, gen_ids_list)):
        seq = list(prompt) + list(gen)
        if len(seq) > seq_len + 1:
            seq = seq[: seq_len + 1]
        n = len(seq) - 1
        if n <= 0:
            continue
        idx[i, :n] = np.asarray(seq[:-1], dtype=np.int32)[:n]
        labels[i, :n] = np.asarray(seq[1:], dtype=np.int32)[:n]
        # Supervise only the generated tokens.  Positions whose next-token
        # target is a generated token are [len(prompt)-1 .. len(prompt)-1+len(gen)-1].
        gen_start = max(len(prompt) - 1, 0)
        gen_end = min(gen_start + len(gen), n)
        if gen_end > gen_start:
            mask[i, gen_start:gen_end] = 1
    adv = np.asarray(advantages, dtype=np.float32)
    return idx, labels, mask, adv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GRPO on GSM8K against the SFT policy.")
    parser.add_argument("--sft-run-dir", required=True)
    parser.add_argument("--rl-data", required=True, help="JSONL with {question, answer}.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-iters", type=int, default=50)
    parser.add_argument("--m-prompts", type=int, default=4, help="Prompts per RL step.")
    parser.add_argument("--g-rollouts", type=int, default=4, help="Completions per prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--lr-scale", type=float, default=0.02)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--smoke-iters", type=int, default=None)
    parser.add_argument("--tokenizer-json", default=None)
    args = parser.parse_args(argv)

    if args.smoke_iters is not None and args.smoke_iters > 0:
        args.n_iters = int(args.smoke_iters)

    rng = random.Random(args.seed)
    rl_rows = _load_jsonl(args.rl_data)
    rl_rows = [r for r in rl_rows if "question" in r and "answer" in r]
    if not rl_rows:
        raise RuntimeError(f"No RL rows in {args.rl_data}")

    parent_state = ckpt_lib.load_latest(args.sft_run_dir, stage="sft")
    base_config = parent_state["config"]
    parent_meta = {
        "stage": "sft",
        "ckpt_path": parent_state.get("_checkpoint_path", ""),
        "step": int(parent_state.get("step", 0)),
    }

    rl_config = dataclasses.replace(
        base_config,
        embed_lr_base=base_config.embed_lr_base * args.lr_scale,
        lm_head_lr_base=base_config.lm_head_lr_base * args.lr_scale,
        muon_base_lr=base_config.muon_base_lr * args.lr_scale,
        scalar_resid_lr=base_config.scalar_resid_lr * args.lr_scale,
        scalar_x0_lr=base_config.scalar_x0_lr * args.lr_scale,
        n_train_iters=max(args.n_iters, 1),
        n_warmup_iters=max(1, args.n_iters // 20),
        f_warmdown_iters=0.5,
        target_train_tokens=base_config.tokens_per_step * max(args.n_iters, 1),
        weight_decay_base=0.0,
        save_every=args.save_every,
    )

    logger = Logger(run_dir=args.run_dir)
    if logger.is_master:
        wb.wandb_init(
            stage="rl",
            run_name=f"rl-{logger.run_id}",
            config_dict=wb.config_from_dataclass(
                rl_config,
                {
                    "n_iters": args.n_iters,
                    "m_prompts": args.m_prompts,
                    "g_rollouts": args.g_rollouts,
                    "kl_beta": args.kl_beta,
                    "clip_eps": args.clip_eps,
                    "lr_scale": args.lr_scale,
                    "parent": parent_meta,
                },
            ),
        )
    mesh = get_mesh(rl_config)

    tok_path = args.tokenizer_json or parent_state.get("tokenizer_path") or rl_config.tokenizer_json
    tokenizer = load_hf_tokenizer(tok_path)
    pad_id = int(tokenizer.get_bos_token_id())
    end_id = tokenizer.encode_special("<|assistant_end|>")
    stop_ids = {tid for tid in (end_id, pad_id) if tid is not None}

    with mesh:
        weight_sharding = get_weight_sharding(rl_config, mesh)
        params, _ = init_params(rl_config, mesh)
        params = tree_map(lambda leaf: jax.device_put(jnp.asarray(leaf), weight_sharding), parent_state["params"])
        ref_params = tree_map(lambda leaf: jax.device_put(jnp.asarray(leaf), weight_sharding), parent_state["params"])
        precomputed_params = precompute_rope(rl_config, mesh)
        optimizer, opt_state = init_optimizer(rl_config, params, mesh)
        param_count = count_parameters(params)
        logger.msg(f"RL preset | params: {param_count:,} | iters: {args.n_iters} | M={args.m_prompts} G={args.g_rollouts}")

        # Match train_base / engine: the embedding-output activations follow
        # ``config.activation_sharding`` (replicated batch on the 124m presets);
        # the engine reuses this for batch=1 autoregressive sampling. The RL
        # train batch (M*G prompts) often does divide dp evenly, but we play it
        # safe and route everything through the same replicated layout.
        activation_sharding_3d = NamedSharding(mesh, P(*rl_config.activation_sharding))
        embedding_out_sharding = activation_sharding_3d
        # ``token_sharding`` (idx/labels/mask) and ``adv_sharding`` (advantages)
        # also default to the replicated-batch layout for the same reason.
        token_sharding = NamedSharding(mesh, P(None, None))
        adv_sharding = NamedSharding(mesh, P(None))

        # Build a temporary engine for sampling that shares params (we'll refresh after each step).
        engine = Engine(
            params=params,
            precomputed_params=precomputed_params,
            config=rl_config,
            mesh=mesh,
            tokenizer=tokenizer,
            embedding_out_sharding=embedding_out_sharding,
            stage="rl",
            step=0,
        )

        seq_len = rl_config.max_seq_len
        train_start = time.perf_counter()
        for step in range(args.n_iters):
            # 1) Sample M prompts.
            prompts = rng.sample(rl_rows, k=min(args.m_prompts, len(rl_rows)))

            prompt_ids_list: list[list[int]] = []
            gen_ids_list: list[list[int]] = []
            rewards_per_group: list[list[float]] = []

            for p_idx, row in enumerate(prompts):
                gold = gsm8k._gold_from_solution(row["answer"]) or 0.0
                rendered = tokenizer.render_for_completion(
                    {"messages": gsm8k.build_prompt(row["question"]) + [{"role": "assistant", "content": ""}]}
                )
                group_rewards: list[float] = []
                for g_idx in range(args.g_rollouts):
                    gen_ids = engine.generate_ids(
                        rendered,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p if args.top_p < 1.0 else None,
                        seed=args.seed + step * 1000 + p_idx * 100 + g_idx,
                        stop_token_ids=stop_ids,
                    )
                    text = tokenizer.decode(gen_ids)
                    r = gsm8k.reward(text, gold)
                    group_rewards.append(r)
                    prompt_ids_list.append(rendered)
                    gen_ids_list.append(gen_ids)
                rewards_per_group.append(group_rewards)

            # 2) Group-relative advantages.
            advantages: list[float] = []
            for group_rewards in rewards_per_group:
                arr = np.asarray(group_rewards, dtype=np.float32)
                mean = arr.mean()
                std = arr.std() + 1e-6
                advantages.extend(((arr - mean) / std).tolist())

            mean_reward = float(np.mean([r for grp in rewards_per_group for r in grp]))

            # 3) Build the padded batch.
            idx_np, lbl_np, mask_np, adv_np = _build_padded_batch(
                prompt_ids_list=prompt_ids_list,
                gen_ids_list=gen_ids_list,
                advantages=advantages,
                pad_id=pad_id,
                seq_len=seq_len,
            )
            idx_d = jax.device_put(jnp.asarray(idx_np), token_sharding)
            lbl_d = jax.device_put(jnp.asarray(lbl_np), token_sharding)
            mask_d = jax.device_put(jnp.asarray(mask_np), token_sharding)
            adv_d = jax.device_put(jnp.asarray(adv_np), adv_sharding)

            # 4) Optimize.
            params, opt_state, metrics = rl_train_step(
                rl_config,
                params,
                ref_params,
                precomputed_params,
                opt_state,
                optimizer,
                embedding_out_sharding,
                idx_d,
                lbl_d,
                mask_d,
                adv_d,
                args.kl_beta,
                args.clip_eps,
            )

            engine.params = params  # refresh sampler.
            metrics_view = {k: float(v) for k, v in metrics.items()}
            logger.log({"step": step, "time": datetime.datetime.now(), "mean_reward": mean_reward} | metrics_view)
            wb.wandb_log({"step": step, "mean_reward": mean_reward, **metrics_view})
            if args.save_every > 0 and step > 0 and step % args.save_every == 0 and step < args.n_iters - 1:
                save_path = ckpt_lib.save(
                    stage="rl",
                    step=step,
                    params=params,
                    opt_state=opt_state,
                    config=rl_config,
                    run_dir=logger.logdir,
                    tokenizer_path=tok_path,
                    rng_seed=args.seed,
                    parent=parent_meta,
                )
                logger.msg(f"Saved checkpoint to {save_path}")

        jax.block_until_ready(params)
        logger.msg(f"rl train_loop_s: {time.perf_counter() - train_start:.3f}")
        final_path = ckpt_lib.save(
            stage="rl",
            step=args.n_iters - 1,
            params=params,
            opt_state=opt_state,
            config=rl_config,
            run_dir=logger.logdir,
            tokenizer_path=tok_path,
            rng_seed=args.seed,
            parent=parent_meta,
        )
        logger.msg(f"Saved final RL checkpoint to {final_path}")
        logger.flush()
        wb.wandb_finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
