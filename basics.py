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
    
    key = jnp.random.PRNGKey(0)
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
        return jax.nn.sigmoid(n)
    
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





    