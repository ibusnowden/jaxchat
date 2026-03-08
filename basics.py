# Jax primitives + MLP assignment

"""
basics.py — JAX Core Primitives
=============================================
Module 1 of "Build a 20 M-param LLM from Scratch in JAX".

Sections
  1. Automatic differentiation  (grad, value_and_grad, higher-order)
  2. JIT compilation             (jax.jit, tracing semantics)
  3. Vectorised execution        (vmap, vmap+grad per-sample gradients)
  4. PRNG / random numbers       (keys, split, common distributions)
  5. Assignment — pure-JAX MLP   (no Flax, no Optax)

Run:
    python basics.py
"""

import time
import jax
import jax.numpy as jnp
import numpy as np

print(f"Jax version: {jax.__version__}")
print(f"Devices:     {jax.devices()}")

# Automatic differentiation


def section_autodiff():
    print("─" * 60)
    print("§1  Automatic Differentiation")
    print("─" * 60)
    
    # jax.grad
    # grad(f) returns a NEW function that computes ∂f/∂x.
    # f must be scalar-output (returns a float, not an array)
    def poly(x):
        return x**3 - 2*x**2 + x + 1.0
    
    dpoly = jax.grad(poly)  # ∂(x³ - 2x² + x + 1)/∂x = 3x² - 4x + 1

    for xv in [0.0, 1.0, 2.0]:
        analytic = 3*xv**2 - 4*xv + 1
        print(f"  grad(poly) at x={xv:4.1f}: JAX={dpoly(xv):+.4f}  "
            f"analytic={analytic:+.4f}")
    
    # value_and_grad
    # Returns (f(x), grad_f(x)) in a single forward-backward pass.
    # This is the form you'll use in every training loop.

    val, g = jax.value_and_grad(poly)(2.0)
    print(f"\n  value_and_grad(poly)(2.0) → value={val:.4f}, grad={g:.4f}")

    # Differentiating w.r.t. arrays 
    # grad works on any pytree input — arrays, dicts, nested structures.

    def mse(params, x, y_true):
        """Mean-squared error: L = mean((W·x + b - y)²)"""
        pred = params["W"] @ x + params["b"]
        return jnp.mean((pred - y_true)**2)
    
    key = jax.random.PRNGKey(0)
    params = {
        "W": jax.random.normal(key, (4, 3)),
        "b": jnp.zeros(4), 
    }

    x      = jnp.ones(3)
    y_true = jnp.zeros(4)

    grads = jax.grad(mse)(params, x, y_true)
    print(f"\n ∂MSE/∂W shape: ", {grads["W"].shape})
    print(f"\n ∂MSE/∂b shape: ", {grads["b"].shape})


    # Higher-order derivatives 
    # grad(grad(f)) computes the second derivative, etc.
    def sigmoid(x):
        return jax.nn.sigmoid(x)
    
    d1 = jax.grad(sigmoid)
    d2 = jax.grad(d1)
    d3 = jax.grad(d2)

    x0 = 0.0
    print(f"\n  Derivatives of sigmoid at x=0:")
    print(f"    σ'(0)   = {d1(x0):.6f}  (expected 0.25)")
    print(f"    σ''(0)  = {d2(x0):.6f}  (expected 0.0)")
    print(f"    σ'''(0) = {d3(x0):.6f}")


# §2  JIT COMPILATION

def section_jit():
    print("\n" + "─" * 60)
    print("§2  JIT Compilation")
    print("─" * 60)

    #  Basic jit 
    def slow_fn(x):
        return jnp.sum(jnp.sum(x) **2 + jnp.cos(x) **2)
    
    fast_fn = jax.jit(slow_fn)          # compile once, run many times

    x = jnp.ones(10_000)
    _ = fast_fn(x).block_until_ready() # # warm-up / compile

    N = 200
    t0 = time.perf_counter()
    for _ in range(N): slow_fn(x).block_until_ready()
    ms_raw = (time.perf_counter() - t0) / N * 1000

    t0 = time.perf_counter()
    for _ in range(N): fast_fn(x).block_until_ready()
    ms_jit = (time.perf_counter() - t0) / N * 1000

    print(f"\n  Benchmark (N={N}, array size=10 000):")
    print(f"    Without jit : {ms_raw:.3f} ms/call")
    print(f"    With    jit : {ms_jit:.3f} ms/call")
    speedup = ms_raw / max(ms_jit, 1e-9)
    print(f"    Speedup     : {speedup:.1f}×")

    # Tracing pitfall
    # Python control flow in a jitted function is traced ONCE at compile
    # time. Use jnp.where / jax.lax.cond for runtime-conditional logic.

    print("\n  ⚠  Tracing pitfall — Python 'if' inside jit:")

    @jax.jit
    def broken(x):
        # This branch is evaluated at TRACE time using the abstract value,
        # not the concrete value of x at runtime.
        if x > 0:          # noqa — intentionally broken for demo
            return x * 2
        return x * -1
    
    @jax.jit
    def correct(x):
        return jnp.where(x > 0, x * 2, x * -1)  # JAX-native conditional
    
    for xv in [-3.0, 5.0]:
        print(f"    correct({xv:+.0f}) = {correct(jnp.array(xv)):.1f}")

    #  static_argnums 
    # If you NEED Python-level conditionals, mark that argument as static.
    # The function is recompiled for each unique value of the static arg.

    @jax.jit
    def matmul_with_bias(W, x, use_bias: bool):
        # use_bias is traced as a concrete bool because it's in static_argnums
        y = W @ x
        return y + 1.0 if use_bias else y

# §3  VECTORISED EXECUTION — vmap

def section_vmap():
    print("\n" + "─" * 60)
    print("§3  vmap — Vectorised Execution")
    print("─" * 60)
    

    # Lift a single-sample function to a batch
    def linear(W, x):
        """Single-sample linear layer: (D_out, D_in) × (D_in,) → (D_out,)"""
        return W @ x
    
    # Vectorise over a batch of x's, keeping W fixed
    batched_linear = jax.vmap(linear, in_axes=(None, 0))

    W  = jnp.ones((4, 3))
    xs = jnp.ones((8, 3))          # batch of 8
    ys = batched_linear(W, xs)

    print(f"\n  Single-sample linear  W:{W.shape} × x:{(3,)} → {(4,)}")
    print(f"  Batched  (vmap)       W:{W.shape} × X:{xs.shape} → {ys.shape}")

    # Per-sample gradients  (vmap ∘ grad) 
    # Standard grad averages over the batch. vmap(grad) gives you
    # the gradient for EACH sample independently — useful for:
    #   • per-sample clipping (DP-SGD)
    #   • influence-function analysis
    #   • understanding which samples are hard

    def sample_loss(params, x, y):
        pred = params @ x
        return jnp.sum((pred - y)**2)
    
    per_sample_grad = jax.vmap(
        jax.grad(sample_loss),
        in_axes=(None, 0, 0),   # params fixed, x and y batched
    )

    params = jnp.ones((4, 3))
    xs     = jnp.ones((16, 3))
    ys     = jnp.zeros((16, 4))
    gs     = per_sample_grad(params, xs, ys)   # shape (16, 4, 3)

    print(f"\n  Per-sample gradients shape: {gs.shape}   "
          f"(batch × out_dim × in_dim)")
    print(f"  Per-sample grad norms: {jnp.linalg.norm(gs.reshape(16, -1), axis=1)}")

    # Composing transformations 
    fast_per_sample = jax.jit(per_sample_grad)
    _ = fast_per_sample(params, xs, ys)      # warm-up
    print("\n  jit(vmap(grad(...))) compiles cleanly ")


# §4  PRNG / RANDOM NUMBERS

def section_random():
    print("\n" + "─" * 60)
    print("§4  PRNG — Explicit Random Keys")
    print("─" * 60)

    # JAX's PRNG is *functional*: every random operation needs an explicit
    # key. You never modify a key in-place — you always split it.

    #  Creating & splitting keys 
    key = jax.random.PRNGKey(42)
    print(f"\n  PRNGKey(42) = {key}")

    key, subkey = jax.random.split(key)         # always split, never reuse
    print(f"  After split: key={key}  subkey={subkey}")

    keys = jax.random.split(key, num=6)          # get N keys at once
    print(f"  Batch split → {keys.shape}")

    # Common distributions

    k1, k2, k3, k4, key = jax.random.split(key, 5)

    normal = jax.random.normal(k1, (3, ))
    uniform = jax.random.uniform(k2, (3, ))
    randint  = jax.random.randint(k3, (5,), 0, 100)
    bernoulli = jax.random.bernoulli(k4, 0.3, (8,))

    print(f"\n  Normal(0,1) : {normal}")
    print(f"  Uniform[0,1)  : {uniform}")
    print(f"  RandInt[0,100): {randint}")
    print(f"  Bernoulli(0.3): {bernoulli.astype(int)}")

    # Weight initialisation
    def kaiming_linear(key, fan_in, fan_out):
        """He / Kaiming uniform initialisation."""
        k_w, k_b = jax.random.split(key)
        std = (2.0 / fan_in)** 0.5

        W = jax.random.normal(k_w, (fan_in, fan_out)) * std
        b   = jnp.zeros(fan_out)
        return {"W": W, "b": b}
    
    key, init_key = jax.random.split(key)
    layer = kaiming_linear(init_key, 512, 2048)
    print(f"\n  Kaiming init Linear(512→2048):")
    print(f"    W: {layer['W'].shape},  std={layer['W'].std():.4f}  "
          f"(expected ≈{(2/512)**0.5:.4f})")
    print(f"    b: {layer['b'].shape},  all zeros ✓")


# §5  ASSIGNMENT — Pure-JAX MLP (no Flax, no Optax)


def section_assignment():
    """
    Train a 2-layer MLP on XOR using ONLY raw JAX primitives.
    Goal: learn the XOR function (label 1 iff exactly one input is 1).

    Architecture: Linear(2→32) → Tanh → Linear(32→1) → Sigmoid
    Loss        : Binary cross-entropy
    Optimiser   : Vanilla SGD (manual parameter update)
    """
    print("\n" + "─" * 60)
    print("§5  Assignment — Pure-JAX MLP on XOR (no Flax, no Optax)")
    print("─" * 60)

    # Dataset
    X = jnp.array([[0., 0.],
                   [0., 1.],
                   [1., 0.],
                   [1., 1]])
    
    Y = jnp.array([[0.], [1.], [1.], [0.]])   # XOR labels

    # Parameter initialisation
    def init_params(key, hidden=32):
        k1, k2 = jax.random.split(key)
        return {
            "W1": jax.random.normal(k1, (2, hidden)) * 0.5,
            "b1": jnp.zeros(hidden),
            "W2": jax.random.normal(k2, (hidden, 1)) * 0.5,
            "b2": jnp.zeros(1),
        }
    
    # Forward pass
    def forward(p, x):
        h = jnp.tanh(x @ p["W1"] + p["b1"])       # hidden layer
        return jax.nn.sigmoid(h @ p["W2"] + p["b2"]) # output
    
    
    
    # Loss: Binary cross-entropy
    def bce_loss(p, x, y):
        pred = forward(p, X)
        eps = 1e-7
        return -jnp.mean(y * jnp.log(pred + eps) + (1 - y) * jnp.log(1 - pred + eps))
    
    # Compile the value+grad function once
    loss_and_grad = jax.jit(jax.value_and_grad(bce_loss)) 
    
    # ── Training loop ─────────────────────────────────────────────────────
    params = init_params(jax.random.PRNGKey(7))
    lr     = 1.0
    N      = 3_000

    print("\n Check shapes before training...")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("W1 shape:", params["W1"].shape)
    print("b1 shape:", params["b1"].shape)
    print("W2 shape:", params["W2"].shape)
    print("b2 shape:", params["b2"].shape)

   
    print(f"\n  Training {N} steps, lr={lr} ...")
    for step in range(N+1):
        loss, grads = loss_and_grad(params, X, Y)
        # Manual SGD: θ ← θ - lr·∇θ
        params = jax.tree.map(lambda p, g:p - lr * g, params, grads)

        if step % 10 == 0:
            pred = forward(params, X)
            acc = jnp.mean((pred > 0.5) == Y)
            print(f"   step {step:4d}  | Loss={float(loss):.4f}  |  acc={float(acc):.2f}")

    print("\n  Final predictions:")
    print(f"  {'Input':12s}  {'Pred':>6s}  {'Label':>6s}  {'Correct':>7s}")
    for xi, yi, pi in zip(X, Y, forward(params, X)):
        correct = "✓" if (float(pi[0]) > 0.5) == bool(yi[0]) else "✗"
        print(f"  {str(xi.tolist()):12s}  {float(pi[0]):6.3f}  {int(yi[0]):6d}  {correct:>7s}")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Module 1 — JAX Core Primitives                             ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    section_autodiff()
    section_jit()
    section_vmap()
    section_random()
    section_assignment()




    









    