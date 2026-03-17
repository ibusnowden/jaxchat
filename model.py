
# architecture

"""
Sections:
  1. Flax NNX basics (Module, Param, Rngs)
  2. Enbeddings - token + positional encoding
  3. Cauusal multi-head self-attention
  4. Feed-forward block (GELU)
  5. Pre-norm transformer block
  6. Full minigpt model
  7. Tests + parameter count verification (~124M)


Run:
    python model.py
  
"""

import jax
import jax.numpy as jnp
from flax import nnx

from config import ModelConfig, DEFAULT_MODEL

# TOKEN + POSITIONAL EMBEDDINGS
class Embeddings(nnx.Module):
    # combines a learned token embedding table with  learned positional embeddings.

    # Every input token id is mapped to a d_model-dimensional vector, then a
    # positinal  vector (one per sequence position) is added. The result is 
    # passed through dropout and regularization.

    # Inputs : tokens_ids (batch, seq_len) int32
    # Outputs : embeddings (batch, seq_len, d_model) float32    

    def __init__(self, cfg: ModelConfig, rngs: nnx.Rngs):
        # Token lookup table: shape (vocab_size, d_model)
        self.token_emb = nnx.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.d_model,
            rngs=rngs,
        )
        # Positional lookup table: one vector per position
        self.pos_emb = nnx.Embed(
            num_embeddings=cfg.max_seq_len,
            features=cfg.d_model,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(rate=cfg.dropout, rngs=rngs)
    
    def __call__(self, tokens_ids: jnp.ndarray) -> jnp.ndarray:
        _, T = tokens_ids.shape                           # (B, T)
        tok = self.token_emb(tokens_ids)                  # (B, T, C)
        pos = self.pos_emb(jnp.arange(T))                 # (T, C) - broadcasted 
        return self.dropout(tok + pos)                    # (B, T, C)


# CAUSAL MULTI-HEAD SELF-ATTENTION
class CausalSelfAttention(nnx.Module):
    """
    Multi-head causal (decoder only) self-attention.
    
    # Design 
    - Fused QKV projection: one (d_model -> 3*d_model) matrix is faster
    and cleaner than three separate projections.
    - Causal mask: position i can only attend  to j <= i. Built with
    jnp.tril and jnp.where (jit-safe; no Python if needed).
    - Scaled dot-product: divide by sqrt(d_head) before softmax to keep
    gradients healthy regardless of d_head.

    # Shapes throughout  (B=batch, T=seq_len, C=d_model, H=n_heads, D=d_head)
    input  : (B, T, C)
    QKV    : (B, T, 3C) splits Q,K,V each (B, T, C)
    heads  : reshape + tanspose -> (B, H, T, D)
    scores : (B, H, T, T)
    output : (B, T, C)
    """
    def __init__(self, cfg: ModelConfig, rngs: nnx.Rngs):
        self.n_heads = cfg.n_heads    # model depth 8 linear MLP layer
        self.d_head  = cfg.d_head
        self.d_model = cfg.d_model    # model width 512

        self.qkv_proj   = nnx.Linear(cfg.d_model, 3 * cfg.d_model, use_bias=True, rngs=rngs)
        self.out_proj   = nnx.Linear(cfg.d_model, cfg.d_model, use_bias=True, rngs=rngs)
        self.attn_drop  = nnx.Dropout(rate=cfg.attn_dropout, rngs=rngs)
        self.resid_drop = nnx.Dropout(rate=cfg.dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, T, C = x.shape
        H, D = self.n_heads, self.d_head

        # Fused qkv projection
        qkv = self.qkv_proj(x)                     # (B, T, 3C)
        q, k, v = jnp.split(qkv, 3, axis=-1)       # each (B, T, C)

        # split into heads -> (B, H, T, D)
        def to_heads(t: jnp.ndarray) -> jnp.ndarray:
            return t.reshape(B, T, H, D).transpose(0, 2, 1, 3)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        # scaled dot-product attention
        scale = D ** -0.5
        scores = (q @ k.swapaxes(-2, -1)) * scale   # (B, H, T, T)

        # causal mask (lower-triangular)
        # mask   (T, T)
        mask = jnp.tril(jnp.ones((T, T))).astype(bool)
        scores = jnp.where(mask, scores, -jnp.inf)   # mask future tokens

        attn = jax.nn.softmax(scores, axis=-1)      # (B, H, T, T)
        attn = self.attn_drop(attn)                 

        # weight sum of values
        y = attn @ v                               # (B, H, T, D)

        # merge heads -> (B, T, C)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.resid_drop(self.out_proj(y))

class FFN(nnx.Module):
    """
    Position-wise two-layer MLP applied at every token prediction

    arch: Linear(C -> 4C) -> GELU -> Linear(4C -> C) -> Dropout

    gelu imporves training stability by having smoother gradient near zero
    compared to relu.
    """

    def __init__(self, cfg: ModelConfig, rngs: nnx.Rngs):
        self.fc1     = nnx.Linear(cfg.d_model, cfg.d_ff, rngs=rngs)
        self.fc2     = nnx.Linear(cfg.d_ff, cfg.d_model, rngs=rngs)
        self.dropout = nnx.Dropout(rate=cfg.dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # f1 = self.fc1(x)
        # g = jax.nn.gelu(f1)
        # d = self.dropout(g)
        # f2 = self.fc2(d)
        # return self.droput(f2)
        return self.dropout(self.fc2(self.dropout(jax.nn.gelu(self.fc1(x)))))
    
    # Transformer block, pre-layernorm
class TransformerBlock(nnx.Module):
    """
    One decoder block using Pre-LayerNorm:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """
    def __init__(self, cfg: ModelConfig, rngs: nnx.Rngs):
        self.ln_attn = nnx.LayerNorm(cfg.d_model, rngs=rngs)
        self.attn    = CausalSelfAttention(cfg, rngs)
        self.ln_ffn  = nnx.LayerNorm(cfg.d_model, rngs=rngs)
        self.ffn     = FFN(cfg, rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.attn(self.ln_attn(x))  # attention sub-layer
        x = x + self.ffn(self.ln_ffn(x))    # FFN sub-layer
        return x

# full gpt model
class GPT(nnx.Module):
    """
    a decoder-only transformer language model (~124 M parameters by default).
    FP:
    - token_ids (B, T) -> logits(B, T, vocab_size)

    to get the next-token distribution, take logits[:, -1. :].

    Training objectives
    - Causal LM:
        loss = cross_entropy(logits[:, -1, :], token_ids[:, 1:])
    (predict token t+1 given 0...t)

    """
    def __init__(self, cfg: ModelConfig = DEFAULT_MODEL, *, rngs: nnx.Rngs):
        self.cfg = cfg

        self.emb = Embeddings(cfg, rngs)
        self.blocks = nnx.List([TransformerBlock(cfg, rngs) for _ in range(cfg.n_layers)])
        self.ln_f = nnx.LayerNorm(cfg.d_model, rngs=rngs)

        # Only create a separate untied head when needed
        if not cfg.tie_weights:
            self.lm_head = nnx.Linear(
                cfg.d_model, cfg.vocab_size, use_bias=False, rngs=rngs
            )
    # when weights are tied, use the embedding layer’s attend() method in __call__.
    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        x = self.emb(token_ids)      # (B, T, C)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)             # (B, T, C)

        if self.cfg.tie_weights:
            return self.emb.token_emb.attend(x)   # (B, T, vocab_size)
        else:
            return self.lm_head(x)                # (B, T, vocab_size)

    # utils
    def count_params(self) -> int:
        # total number of trainabble parameters
        state = nnx.state(self)
        return int(sum(p.size for p in jax.tree.leaves(state)))

    def param_summary(self) -> None:
        # pretty print a parameter count table
        state = nnx.state(self)
        leaves = jax.tree_util.tree_leaves_with_path(state)
        total = 0
        print(f"\n {'Path':55s} {'Shape':20s} {'Params':>10s}")
        print(" " + "-" * 90)
        for path, leaf in leaves:
            path_str = "/".join(
               str(getattr(p, "key", p))
            for p in path
            )

            n = leaf.size
            total += n
            print(f" {path_str[:55]:55s} {str(leaf.shape):20s} {n:>10}")
        print(" " + "-" * 90)
        print(f" {'TOTAL':55s} {'':20s} {total:>10,} (~{total/1e6:.1f} M)")

