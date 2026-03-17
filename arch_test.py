from decimal import DefaultContext
import jax
import jax.numpy as jnp
from flax import nnx

from config import ModelConfig, DEFAULT_MODEL
from model import Embeddings, CausalSelfAttention, FFN, GPT, TransformerBlock

# Test optional
def _ok(msg: str) -> None:
    print(f"    ✓  {msg}")

def _fail(msg: str) -> None:
    raise AssertionError(f"✗  {msg}")


def test_embeddings(cfg: ModelConfig) -> None:
    print("\n  [Test] Embeddings")
    rngs = nnx.Rngs(params=0, dropout=1)
    emb  = Embeddings(cfg, rngs)

    ids = jnp.zeros((4, 32), dtype=jnp.int32)
    out = emb(ids)

    assert out.shape == (4, 32, cfg.d_model), \
        f"Expected (4,32,{cfg.d_model}), got {out.shape}"
    assert jnp.all(jnp.isfinite(out)), "Embeddings contain non-finite values"
    _ok(f"output shape {out.shape}")


def test_attention(cfg: ModelConfig) -> None:
    print("\n  [Test] CausalSelfAttention")
    rngs = nnx.Rngs(params=0, dropout=1)
    attn = CausalSelfAttention(cfg, rngs)
    attn.eval()   # disable dropout for deterministic test


    B, T, C = 2, 24, cfg.d_model
    x   = jax.random.normal(jax.random.PRNGKey(0), (B, T, C))
    out = attn(x)

    assert out.shape == x.shape, f"Shape changed: {x.shape} → {out.shape}"
    _ok(f"shape preserved {out.shape}")

    # Verify causality: the output at position 0 must NOT depend on position T-1.
    # We check this by zeroing out position T-1 and confirming position 0 is unchanged.
    x2 = x.at[:, T - 1, :].set(0.0)
    out2 = attn(x2)
    diff_at_pos0 = jnp.max(jnp.abs(out[:, 0, :] - out2[:, 0, :]))
    assert float(diff_at_pos0) < 1e-5, \
        f"Causality violated: pos-0 output changed by {diff_at_pos0}"
    _ok("causal masking verified (pos-0 independent of last token)")


def test_ffn(cfg: ModelConfig) -> None:
    print("\n  [Test] FeedForward")
    rngs = nnx.Rngs(params=0, dropout=1)
    ffn  = FFN(cfg, rngs)
    x    = jnp.ones((3, 16, cfg.d_model))
    out  = ffn(x)
    assert out.shape == x.shape
    _ok(f"shape {out.shape}")


def test_transformer_block(cfg: ModelConfig) -> None:
    print("\n  [Test] TransformerBlock")
    rngs  = nnx.Rngs(params=0, dropout=1)
    block = TransformerBlock(cfg, rngs)
    block.eval()

    x     = jnp.ones((2, 16, cfg.d_model))
    out   = block(x)
    assert out.shape == x.shape
    _ok(f"shape {out.shape}")


def test_minigpt(cfg: ModelConfig) -> GPT:
    print("\n  [Test] Full MiniGPT")
    rngs  = nnx.Rngs(params=42, dropout=0)
    model = GPT(cfg, rngs=rngs)

    # Forward pass
    B, T = 4, 32
    ids  = jax.random.randint(jax.random.PRNGKey(1), (B, T), 0, cfg.vocab_size)
    logits = model(ids)

    assert logits.shape == (B, T, cfg.vocab_size), \
        f"Bad logits shape: {logits.shape}"
    _ok(f"logits shape {logits.shape}")

    assert jnp.all(jnp.isfinite(logits)), "Logits contain NaN/Inf"
    _ok("all logits finite")

    n = model.count_params()
    assert 120_000_000 < n < 130_000_000, \
        f"Param count {n:,} outside expected range [120M, 130M]"
    _ok(f"parameter count {n:,}  (~{n/1e6:.1f} M)")

    return model


def demo_next_token(model: GPT, cfg: ModelConfig) -> None:
    print("\n  [Demo] Next-token prediction")
    prompt = jnp.array([[1, 5, 23, 99, 42]])          # 5 fake token IDs
    logits = model(prompt)                              # (1, 5, V)
    last   = logits[:, -1, :]                           # (1, V)
    probs  = jax.nn.softmax(last, axis=-1)
    top5   = jnp.argsort(probs[0])[::-1][:5]
    print(f"    prompt   : {prompt[0].tolist()}")
    print(f"    top-5 ids: {top5.tolist()}")
    print(f"    top-5 prob: {[f'{p:.4f}' for p in probs[0, top5].tolist()]}")


if __name__ == "__main__":
    print("GPT Architecture with Flax NNX")
    cfg = DEFAULT_MODEL
    print(f"\n  Config  →  vocab={cfg.vocab_size}  d_model={cfg.d_model}  "
        f"n_heads={cfg.n_heads}  n_layers={cfg.n_layers}  "
        f"d_ff={cfg.d_ff}  max_seq_len={cfg.max_seq_len}")
    print(f"  Estimated params: {cfg.approx_params:,}  (~{cfg.approx_params/1e6:.1f} M)\n")


    test_embeddings(cfg)
    test_attention(cfg)
    test_ffn(cfg)
    test_transformer_block(cfg)
    model = test_minigpt(cfg)
    demo_next_token(model, cfg)

    print("\n  Full parameter table:")
    model.param_summary()

   
    


   

    
