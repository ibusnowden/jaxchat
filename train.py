
# TOKENISER & DATA

"""
Sections
  1. TinyStories tokenizer + dataset download
  2. Batch iterator
  3. Loss function (causal LM cross-entropy)
  4. Optax optimiser: AdamW + cosine LR schedule + warmup + grad clip
  5. JIT-compiled train step
  6. Evaluation loop + perplexity
  7. Training loop with logging
  8. Orbax checkpointing (save / restore)
 
Run:
    python train.py
"""
import argparse
import os
import math
import time
import urllib.request
from array import array
from pathlib import Path
from typing import Iterator
 
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
import tiktoken
 
try:
    import orbax.checkpoint as ocp
    _ORBAX = True
except ImportError:
    _ORBAX = False

from config import ModelConfig, TrainConfig, DEFAULT_MODEL, DEFAULT_TRAIN
from model import GPT


TINYSTORIES_TRAIN_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStoriesV2-GPT4-train.txt"
)
TINYSTORIES_VALID_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStoriesV2-GPT4-valid.txt"
)
TOKEN_CACHE_VERSION = "bos_eos_v1"


def download_file(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    print(f"  Downloading {url} → {path}")
    urllib.request.urlretrieve(url, path)
    print("  Done.")
    return path


class TinyStoriesTokenizer:
    """
    cl100k_base + explicit BOS/EOS/PAD special tokens for training/inference.
    """

    BASE_ENCODING = "cl100k_base"
    BOS = "<|bos|>"
    EOS = "<|eos|>"
    PAD = "<|pad|>"

    def __init__(self):
        base = tiktoken.get_encoding(self.BASE_ENCODING)

        start_id = base.max_token_value + 1

        extra_special_tokens = {
            self.BOS: start_id + 0,
            self.EOS: start_id + 1,
            self.PAD: start_id + 2,
        }

        self._enc = tiktoken.Encoding(
            name=f"{self.BASE_ENCODING}_tinystories_special",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens={
                **base._special_tokens,
                **extra_special_tokens,
            },
            
        )

        self.vocab_size = self._enc.max_token_value + 1
       
        self.bos_id = self._enc._special_tokens[self.BOS]
        self.eos_id = self._enc._special_tokens[self.EOS]
        self.pad_id = self._enc._special_tokens[self.PAD]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> np.ndarray:
        ids = self._enc.encode_ordinary(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return np.asarray(ids, dtype=np.int32)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        ids = list(map(int, ids))
        if skip_special_tokens:
            special_ids = {self.bos_id, self.eos_id, self.pad_id}
            ids = [i for i in ids if i not in special_ids]
        return self._enc.decode(ids)


def _token_cache_path(txt_path: Path) -> Path:
    return txt_path.with_name(f"{txt_path.stem}.{TOKEN_CACHE_VERSION}.tokens.npy")


def _load_or_build_tokens(
    txt_path: Path,
    tokenizer: TinyStoriesTokenizer,
    *,
    max_chars: int | None = None,
) -> np.ndarray:
    tok_path = _token_cache_path(txt_path)
    if tok_path.exists():
        print(f"  Using cached tokens → {tok_path}")
        return np.load(tok_path, mmap_mode="r")

    story_tokens = array("I")
    story_count = 0
    used_chars = 0

    with txt_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_story in fh:
            story = raw_story.strip()
            if not story:
                continue

            if max_chars is not None:
                remaining = max_chars - used_chars
                if remaining <= 0:
                    break
                if len(story) > remaining:
                    if story_count > 0:
                        break
                    story = story[:remaining].rstrip()
                    if not story:
                        break

            story_tokens.append(tokenizer.bos_id)
            story_tokens.extend(tokenizer.encode(story).tolist())
            story_tokens.append(tokenizer.eos_id)

            used_chars += len(story)
            story_count += 1

    if story_count == 0:
        raise ValueError(f"No TinyStories samples found in {txt_path}")

    ids = np.asarray(story_tokens, dtype=np.int32)
    np.save(tok_path, ids)
    print(
        f"  Saved token cache → {tok_path}  "
        f"({story_count:,} stories, {len(ids):,} tokens)"
    )
    return np.load(tok_path, mmap_mode="r")


def prepare_data(cfg: TrainConfig, model_cfg: ModelConfig):
    """
    Download TinyStories, build tokenizer, and return memory-mapped token arrays.

    Returns
    -------
    tokenizer  : TinyStoriesTokenizer
    train_ids  : np.ndarray or np.memmap  (N_train,)
    val_ids    : np.ndarray or np.memmap  (N_val,)
    """
    data_dir = Path(cfg.data_dir)
    train_txt = download_file(TINYSTORIES_TRAIN_URL, data_dir / "tinystories_train.txt")
    valid_txt = download_file(TINYSTORIES_VALID_URL, data_dir / "tinystories_valid.txt")

    tok = TinyStoriesTokenizer()
    print(
        f"  Tokenizer: {tok.BASE_ENCODING} + BOS/EOS/PAD | "
        f"vocab={tok.vocab_size:,} | bos={tok.bos_id}, eos={tok.eos_id}, pad={tok.pad_id}"
    )

    # For quick experiments, cap the train corpus if your TrainConfig has max_chars.
    max_chars = getattr(cfg, "max_chars", None)

    train_ids = _load_or_build_tokens(train_txt, tok, max_chars=max_chars)
    val_ids = _load_or_build_tokens(valid_txt, tok, max_chars=None)

    if tok.vocab_size > model_cfg.vocab_size:
        raise ValueError(
            f"Tokenizer vocab ({tok.vocab_size}) > model vocab ({model_cfg.vocab_size}). "
            "Increase ModelConfig.vocab_size."
        )

    print(f"  Train tokens: {len(train_ids):,}   Val tokens: {len(val_ids):,}")
    return tok, train_ids, val_ids

# batch iterator

def batch_iter(
    ids: np.ndarray,
    batch_size: int,
    seq_len:    int,
    key: jax.Array,
    device=None
):
    """
    Infinite generator of (inputs, targets) batches.

    inputs  : (B, T)
    targets : (B, T)   shifted by one token
    """
    n = len(ids) - seq_len - 1
    if n <= 0:
        raise ValueError(f"Dataset too short for seq_len={seq_len}")
    
    while True:
        key, subkey = jax.random.split(key)
        starts = np.asarray(jax.random.randint(subkey, (batch_size, ), 0, n))

        x = np.stack([ids[s:s + seq_len] for s in starts]).astype(np.int32)
        y = np.stack([ids[s + 1:s + seq_len + 1] for s in starts]).astype(np.int32)

        x = jnp.asarray(x)
        y = jnp.asarray(y)

        if device is not None:
            x = jax.device_put(x, device=device)
            y = jax.device_put(y, device=device)

        yield x, y

# loss function
def causal_lm_loss(
    model: GPT,
    inputs: jnp.ndarray, # (B, T)
    targets: jnp.ndarray # (B, T)
) -> jnp.ndarray:
    """
    causal language modeling cross-entropy loss

    for each position t, predict the token at positon t+1.
    equivalently: given inputs x[0...T-1], compute logits for each position,
    then measure cross-entropy against the shifted target sequence.

    Returns: scalar loss (mean ovel all batch x tiem positions)
    """
    logits = model(inputs)      # (B, T, V)

    # flatten for cross_entropy: (B*T, V) vs (B*T,)
    B, T, V = logits.shape
    logits_2d = logits.reshape(B * T, V)
    target_1d = targets.reshape(B * T)

    loss = optax.softmax_cross_entropy_with_integer_labels(logits_2d, target_1d)
    return loss.mean()

def perplexity(loss: float) -> float:
    # perplexity = exp(cross-entropy loss). lower is better
    return math.exp(min(loss, 20.0))          # cap to avoid overflow during warm-up

# Optimizer
def make_optimizer(cfg: TrainConfig) -> optax.GradientTransformation:
    """
    AdamW with:
      - linear warmup for cfg.warmup_steps steps
      - cosine decay from cfg.learning_rate to cfg.min_lr
      - global gradient norm clipping

    Using optax.chain to compose transfomrs in order:
      clip -> scale_by_adam -> add_decayed_weights -> scale_by_schedule
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.learning_rate,
        warmup_steps=cfg.warmup_step,
        decay_steps=cfg.total_steps,
        end_value=cfg.min_lr
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),          # gradient clipping
        optax.scale_by_adam(b1=cfg.beta1, b2=cfg.beta2),
        optax.add_decayed_weights(cfg.weight_decay),       # AdamW wight decay
        optax.scale_by_learning_rate(schedule),            # LR schedule
    )
    return optimizer

@nnx.jit
def train_step(
    model:     GPT,
    optimizer: nnx.Optimizer,
    inputs:    jnp.ndarray,
    targets:   jnp.ndarray,
) -> tuple[jnp.ndarray]:
   """
   One gradient update.

   Returns (loss, grad_norm) both as scalar jnp arrays.

   Note: @nnx.jit is Flax's version of @jax.jit, it handles the mutable
   model state (parameters, optimizer, state) automatically
   """
   loss, grads = nnx.value_and_grad(causal_lm_loss)(model, inputs, targets)

   # global gradient norm ( for monitoring: clipping already in optimizer)
   leaves = jax.tree.leaves(grads)
   grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))

   optimizer.update(grads)
   return loss, grad_norm

# evaluation
@nnx.jit
def eval_step(
    model: GPT,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
) -> jnp.ndarray:
   # compute loss on a single validation batch (no gradient)
   return causal_lm_loss(model, inputs, targets)

def evaluate(
    model: GPT,
    val_ds: np.ndarray,
    cfg: TrainConfig,
    key: jax.Array,
) -> tuple[float, float]:
    # compute mean loss and perplexity over cfg.eval_iters validation batches
    device = jax.devices()[0]
    losses = []
    iter  = batch_iter(val_ds, cfg.batch_size, cfg.seq_len, key, device=device)
    for _ in range(cfg.eval_iters):
       x, y = next(iter)
       loss = eval_step(model, x, y)
       losses.append(float(loss))
    mean_loss = float(np.mean(losses))
    return mean_loss, perplexity(mean_loss)

# checkpointing
def save_checkpoint(
    model: GPT,
    optimizer: nnx.Optimizer,
    step: int,
    ckpt_dir: str,
) -> None:
    # save model+ optimizer state using Orbax
    if not _ORBAX:
       print("  [checkpoint] orbax not available - skipping save.")
       return 
    
    ckpt_dir = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    mgr = ocp.CheckpointManager(ckpt_dir)

    # extract pytrees
    model_state = nnx.state(model)
    opt_state = nnx.state(optimizer)

    mgr.save(step, args=ocp.args.StandardSave({
        "model": model_state,
        "optimizer": opt_state,
        "step": step,
    }))
    mgr.wait_until_finished()


def _restore_standard_checkpoint_item(
    ckpt_dir: str,
    step: int,
    item,
):
    ckpt_path = os.path.join(os.path.abspath(ckpt_dir), str(step), "default")
    fallback_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    return checkpointer.restore(
        ckpt_path,
        args=ocp.args.StandardRestore(
            item=item,
            fallback_sharding=fallback_sharding,
        ),
    )

def restore_checkpoint(
    model: GPT,
    optimizer: nnx.ModelAndOptimizer,
    ckpt_dir: str,
) -> int:
    # restore latest checkpoint. Returns the step number.
    if not _ORBAX:
       print("  [checkpoint orbax not available - cannot restore]")
       return 0

    ckpt_dir = os.path.abspath(ckpt_dir)
    mgr = ocp.CheckpointManager(ckpt_dir)
    latest = mgr.latest_step()
    if latest is None:
        print("  [checkpoint] no checkpoint found.")
        return 0

    restored = _restore_standard_checkpoint_item(
        ckpt_dir,
        latest,
        {
            "model": nnx.state(model),
            "optimizer": nnx.state(optimizer),
            "step": 0,
        },
    )
    nnx.update(model,     restored["model"])
    nnx.update(optimizer, restored["optimizer"])
    print(f"   [checkpoint] restored step {latest} from {ckpt_dir}")
    return int(restored["step"])

# the main training loop
def train(
    model_cfg: ModelConfig = DEFAULT_MODEL,
    train_cfg: TrainConfig = DEFAULT_TRAIN,
    resume: bool           = False,
) -> GPT:
    """
    Full training loop.

    Parameters
    ----------
    model_cfg : architecture configuration
    train_cfg : training hyperparameters
    resume    : if True, load the latest checkpoint from train_cfg.checkpoint_dir

    Returns:
    -------
    Trained mini GPT model
    """
    print("=" * 64)
    print(" GPT - Training")

    # reproduce
    key = jax.random.PRNGKey(train_cfg.seed)

    # data
    print("\n[1] Preparing data...")
    tokenizer, train_ids, val_ids = prepare_data(train_cfg, model_cfg)

    # update vocab_size in model config to match corpus
    model_cfg = ModelConfig(
        vocab_size   = tokenizer.vocab_size,
        max_seq_len  = model_cfg.max_seq_len,
        d_model      = model_cfg.d_model,
        n_heads      = model_cfg.n_heads,
        n_layers     = model_cfg.n_layers,
        d_ff         = model_cfg.d_ff,
        dropout      = model_cfg.dropout,
        attn_dropout = model_cfg.attn_dropout,
        tie_weights  = model_cfg.tie_weights,
    )

    # model
    print("\n[2] Building model...")
    key, model_key = jax.random.split(key)
    rngs  = nnx.Rngs(params=int(model_key[0]), dropout=int(model_key[1]))
    model = GPT(model_cfg, rngs=rngs)
    n     = model.count_params()
    print(f"  Parameters: {n:,}  (~{n/1e6:.1f} M)")

    # optimizer
    print("\n[3] Setting up optimizer...")
    optimizer = nnx.ModelAndOptimizer(model, make_optimizer(train_cfg))
    print(f" AdamW lr={train_cfg.learning_rate}  wd={train_cfg.weight_decay}")
    print(f" Warmup={train_cfg.warmup_step} Total={train_cfg.total_steps}")
    print(f" Grad clip={train_cfg.grad_clip}")

    # optional resume
    start_step = 0
    if resume:
        start_step = restore_checkpoint(model, optimizer, train_cfg.checkpoint_dir)
    
    # batch iterator
    key, tk, vk = jax.random.split(key, 3)
    train_it = batch_iter(train_ids, train_cfg.batch_size, train_cfg.seq_len, tk)

    # training loop
    print(f"\n[4] Training (step {start_step} -> {train_cfg.total_steps})...")
    print(f" {'Step':>6} {'Loss':>8} {'Ppl':>8} {'GradNorm':>10} {'ms/step':>9}")
    print(" " + "-" * 50)

    t0 = time.perf_counter()
    for step in range(start_step, train_cfg.total_steps + 1):
        x, y = next(train_it)

        loss, grad_norm = train_step(model, optimizer, x, y)

        # logging
        if step % train_cfg.log_every == 0:
            ms = (time.perf_counter() - t0) / max(step - start_step, 1) * 1000
            ppl = perplexity(float(loss))
            print(f" {step:>6} {float(loss):>8.4f} {ppl:>8.2f} {float(grad_norm):>10.4f} {ms:>8.1f}ms")

        # validation
        if step > 0 and step % train_cfg.eval_every == 0:
            val_loss, val_ppl = evaluate(
                model,
                val_ids,
                train_cfg,
                jax.random.fold_in(vk, step),
            )
            print(f" {'val':>6} {val_loss:>8.4f} {val_ppl:>8.2f} {'-':>10} {'-':>9}")

        if step > 0 and step % train_cfg.save_every == 0:
            save_checkpoint(model, optimizer, step, train_cfg.checkpoint_dir)

    print("\n✓  Training complete.")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TinyStories GPT")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in DEFAULT_TRAIN.checkpoint_dir",
    )
    args = parser.parse_args()

    model = train(DEFAULT_MODEL, DEFAULT_TRAIN, resume=args.resume)
    print("\nTraining finished. Run inference.py against the saved checkpoint.")



