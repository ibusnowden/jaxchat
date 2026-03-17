# sampling + TinyStories continuation UI
"""
Sections
  1. Sampling primitives (temperature,, top-k, top-p / nucleus)
  2. Autoregressive generation loop
  3. Loading a pretraining checkpoint
  4. Gradio TinyStories playground with streaming tokens

Run after training or downloading a checkpoint
  - Gradio UI (runs on http://0.0.0.0:7860):                                                                                                                   
    python inference.py --ui                                                                                                                                   
 
  - With a specific checkpoint:                                                                                                                                
    python inference.py --ui --ckpt ./checkpoints                                                                                                              

  - Print built-in TinyStories sample prompts:
    python inference.py --samples
                                                                                                                                                             
  - CLI mode (no UI, text generation in terminal):                                                                                                             
    python inference.py   
"""

import argparse
import functools
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np

from config import ModelConfig, GenerationConfig, DEFAULT_MODEL, DEFAULT_GEN, DEFAULT_TRAIN
from model import GPT
from train import TinyStoriesTokenizer, make_optimizer, _restore_standard_checkpoint_item

try:
    from flax import nnx
    import orbax.checkpoint as ocp
    _ORBAX = True
except ImportError:
    _ORBAX = False

try:
    import gradio as gr
    _GRADIO = True
except ImportError:
    _GRADIO = False
    print("x gradio not installed - UI disbaled. uv pip install gradio")


TINYSTORIES_SAMPLE_PROMPTS = [
    "Once upon a time, there was a little girl named Lily.",
    "One day, Tom found a shiny red ball in the grass.",
    "Ben was sad because he could not find his toy car.",
    "Mia wanted to bake a cake with her mom.",
    "The little dog was scared of the rain.",
    "Lucy found a tiny key under the old tree.",
    "At bedtime, Max asked his dad for one more story.",
]

GRADIO_CSS = """
.gradio-container { max-width: 860px; margin: auto; }
h1 { font-size: 1.6rem; font-weight: 800; }
"""


def _build_gradio_theme() -> "gr.Theme":
    if not _GRADIO:
        raise RuntimeError("gradio not installed, uv pip install gradio")
    return gr.themes.Base(
        primary_hue="amber",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("JetBrains Mono"),
    )


def _normalise_prompt_text(tokenizer: TinyStoriesTokenizer, prompt: str) -> str:
    text = prompt.strip()
    for special in (tokenizer.BOS, tokenizer.EOS, tokenizer.PAD):
        text = text.replace(special, "")
    return text.strip()


def _encode_prompt_ids(tokenizer: TinyStoriesTokenizer, prompt: str) -> list[int]:
    text = _normalise_prompt_text(tokenizer, prompt)
    return tokenizer.encode(text, add_bos=True, add_eos=False).tolist()


def _display_prompt(tokenizer: TinyStoriesTokenizer, prompt: str) -> str:
    return f"{tokenizer.BOS}{_normalise_prompt_text(tokenizer, prompt)}"


def _build_story_prompt(story_prefix: str, prompt: str) -> str:
    prefix = story_prefix.strip()
    message = prompt.strip()
    if prefix and message:
        return f"{prefix} {message}"
    return message or prefix


def _mask_non_generation_tokens(
    logits: jnp.ndarray,
    tokenizer: TinyStoriesTokenizer,
) -> jnp.ndarray:
    """
    Prevent BOS/PAD from being sampled as normal continuation tokens.
    """
    logits = logits.at[tokenizer.bos_id].set(-jnp.inf)
    logits = logits.at[tokenizer.pad_id].set(-jnp.inf)
    fallback = jnp.full_like(logits, -jnp.inf).at[tokenizer.eos_id].set(0.0)
    has_valid_token = jnp.any(jnp.isfinite(logits))
    return jnp.where(has_valid_token, logits, fallback)

# sampling primitives
def apply_temperature(logits: jnp.ndarray, temperature: float) -> jnp.ndarray:
    """
    divide logits by temperature before softmax.

    temp < 1.0  -> sharper distribution (more focused / repetitive)
    temp = 1.0  -> unchanged
    temp > 1.0  -> flatter distribution (more random/ creative)
    temp -> 0   -> greedy decoding (argmax)
    temp -> inf -> uniform random sampling
    """
    return logits / temperature

def apply_top_k(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    """
    zero out all logits except the top-k largest.

    focus the model to only sample from its k most-likely predictions,
    preventing very low-prob tokens from ever being generating.

    k = 0 -> disabled (all logits)
    """
    if k <= 0:
        return logits
    # find the k-th largest value
    threshold = jnp.sort(logits)[-k]
    return jnp.where(logits >= threshold, logits, -jnp.inf)

def apply_top_p(logits: jnp.ndarray, p: float) -> jnp.ndarray:
    """
    Nucleus sampling: keep the smallest set of tokens whose cumulative
    prob >= p, zero out the rest.

    Unlike top-k (fixed number of tokens), top-p adapts to the distribution:
    - confident prediction: only ~2-3 tokens needed to reach 90% prob
    - uncertain prediction: many tokens needed

    p = 1.0 -> disabled (all tokens eligible)
    p = 0.9 -> typical value: keeps ~50-200 tokens depending on entropy
    """
    if p >= 1.0:
        return logits
    
    # sort in descending order
    sorted_idx    = jnp.argsort(logits)[::-1]
    sorted_logits = logits[sorted_idx]
    cumprobs      = jnp.cumsum(jax.nn.softmax(sorted_logits))

    # find the first position where cumulative prob exceeds p
    # keep tokens at positions where cum_prob_shifted < p
    # (shifted by 1 so we always include the top token)
    cutoff_mask = jnp.concatenate([jnp.array([False]), cumprobs[:-1] >= p])
    sorted_logits = jnp.where(cutoff_mask, -jnp.inf, sorted_logits)

    # unsort back to original token ordering
    # jax doesn't have scatter_nd natively: use this workaroung
    result = jnp.full_like(logits, -jnp.inf)
    result = result.at[sorted_idx].set(sorted_logits)
    return result

def apply_repetition_penalty(
    logits: jnp.ndarray,
    context: jnp.ndarray,
    penalty: float,
) -> jnp.ndarray:
    """
    Reduce logits of token that already appeared in 'context'.

    For each token id t in context:
       if logit[t] > 0: logit[t] /= penalty
       else:            logit[t] *= penalty
   
    penalty > 1.0 -> discourages repetition
    penalty = 1.0 -> n= effect
    """
    if penalty == 1.0:
      return logits
    penalised = jnp.where(logits > 0, logits / penalty, logits * penalty)
    mask = jnp.zeros(logits.shape, dtype=bool).at[context].set(True)
    return jnp.where(mask, penalised, logits)

@functools.partial(jax.jit, static_argnames=('top_k', 'top_p'))
def sample_token(
    key:    jax.Array,
    logits: jnp.ndarray,   # vocab_size - logits for the next token
    temperature: float =  0.8,
    top_k:   int = 50,
    top_p:   float = 0.9,
) -> tuple[jax.Array, jnp.ndarray]:
    """
    sample the next token given logits

    Pipepline:
       raw logits -> temperature -> top-k filter -> softmax -> sample

    Returns (new_key, token_id_scalar)
    """
    logits = apply_temperature(logits, temperature)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)

    key, subkey = jax.random.split(key)
    token_id = jax.random.categorical(subkey, logits)
    return key, token_id

# autoregressive generation loop

def generate(
    model: GPT,
    tokenizer: TinyStoriesTokenizer,
    prompt: str,
    gen_cfg: GenerationConfig = DEFAULT_GEN,
    seed: int = 0,
) -> str:
    """
    autoregressively generate text given a string prompt.

    Algo
    ----
    1. Encode prompt -> token ids
    2. Loop max_new_tokens times:
       a. Forward pass -> logits (B=1, T, V)
       b. Take last-position logits -> (V,)
       c. Apply repetition penalty
       d. Sample next token (temperature + top-k + top-p)
       e. Append to context; stop if context exceeds max_seq_len
    3. Decode ids -> string

    Note: this is the simplest correct implementation.
    For faster inference, add a KV-cache so we only run one forward
    pass per new token instead of re-computing the entire context.
    """
    key     = jax.random.PRNGKey(seed)
    prompt_ids = _encode_prompt_ids(tokenizer, prompt)
    ids     = list(prompt_ids)
    max_len = model.cfg.max_seq_len

    for _ in range(gen_cfg.max_new_tokens):
        # truncate the context window
        ctx    = ids[-max_len:]
        x      = jnp.array([ctx], dtype=jnp.int32)       # (1, T)
        logits = model(x)                                # (1, T, V)
        last   = logits[0, -1, :]                        # (V,)

        # repetition penalty on the current context
        last = apply_repetition_penalty(
            last, jnp.array(ctx), gen_cfg.repetition_penalty
        )
        last = _mask_non_generation_tokens(last, tokenizer)

        key, tok = sample_token(
            key, last,
            temperature=gen_cfg.temperature,
            top_k=gen_cfg.top_k,
            top_p=gen_cfg.top_p
        )
        tok_id = int(tok)
        if tok_id == tokenizer.eos_id:
            break
        ids.append(tok_id)

    return tokenizer.decode(ids[len(prompt_ids):])

def generate_stream(
    model: GPT,
    tokenizer: TinyStoriesTokenizer,
    prompt: str,
    gen_cfg: GenerationConfig = DEFAULT_GEN,
    seed: int = 0,
) -> Iterator[str]:
    """
    Like generate(), but yields each token as it is produced.
    Used to stream tokens to the Gradio chat UI.
    """
    key     = jax.random.PRNGKey(seed)
    ids     = _encode_prompt_ids(tokenizer, prompt)
    max_len = model.cfg.max_seq_len
    output  = ""

    for _ in range(gen_cfg.max_new_tokens):
        ctx    = ids[-max_len:]
        x      = jnp.array([ctx], dtype=jnp.int32)
        logits = model(x)
        last   = logits[0, -1, :]

        last = apply_repetition_penalty(
            last, jnp.array(ctx), gen_cfg.repetition_penalty
        )
        last = _mask_non_generation_tokens(last, tokenizer)

        key, tok = sample_token(
            key, last,
            temperature=gen_cfg.temperature,
            top_k=gen_cfg.top_k,
            top_p=gen_cfg.top_p
        )
        tok_id = int(tok)
        if tok_id == tokenizer.eos_id:
            break
        ids.append(tok_id)
        chunk = tokenizer.decode([tok_id])
        if not chunk:
            continue
        output += chunk
        yield output


def run_sample_prompts(
    model: GPT,
    tokenizer: TinyStoriesTokenizer,
    gen_cfg: GenerationConfig | None = None,
) -> None:
    base_cfg = gen_cfg or GenerationConfig(
        max_new_tokens=96,
        temperature=0.8,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.1,
    )
    sample_cfg = GenerationConfig(
        max_new_tokens=min(base_cfg.max_new_tokens, 96),
        temperature=base_cfg.temperature,
        top_k=base_cfg.top_k,
        top_p=base_cfg.top_p,
        repetition_penalty=base_cfg.repetition_penalty,
    )

    print()
    for prompt in TINYSTORIES_SAMPLE_PROMPTS:
        print("-" * 80)
        completion = generate(model, tokenizer, prompt, sample_cfg)
        print(_display_prompt(tokenizer, prompt) + completion)
    print("-" * 80)
    print()

# loading a pretraining checkpoint

def load_model_from_checkpoint(
    ckpt_dir: str,
    model_cfg: ModelConfig = DEFAULT_MODEL,
) -> GPT:
    """
    Instantiate a fresh GPT and restore weights from an Orbax checkpoint.
    Assume thr checkpoint was saved by train.save_checkpoint().
    """
    if not _ORBAX:
        raise RuntimeError("orbax-checkpoint not installed. uv pip install orbax-checkpoint")

    print(f"  Loading checkpoint from {ckpt_dir}  ...")

    # build model (random params - will be overwritten)
    rngs = nnx.Rngs(params=0, dropout=0)
    model = GPT(model_cfg, rngs=rngs)

    # restore
    mgr = ocp.CheckpointManager(ckpt_dir)
    latest = mgr.latest_step()
    if latest is None:
        raise FileExistsError(f"No checkpoint found in {ckpt_dir}")

    # Orbax 0.11 needs an explicit target tree to remap saved sharding/topology.
    temp_optimizer = nnx.ModelAndOptimizer(model, make_optimizer(DEFAULT_TRAIN))
    restored = _restore_standard_checkpoint_item(
        ckpt_dir,
        latest,
        {
            "model": nnx.state(model),
            "optimizer": nnx.state(temp_optimizer),
            "step": 0,
        },
    )
    nnx.update(model, restored["model"])
    print(f" Loaded step {latest}")
    return model

# gradio chat interface
def build_gradio_app(
    model:     GPT,
    tokenizer: TinyStoriesTokenizer,
) -> "gr.Blocks":
    """
    Build a Gradio Blocks app for TinyStories-style continuations.

    Features:
      - Stream token output
      - Adjustable temperature, top-k, top-p sliders
      - Optional story prefix prepended to each prompt
      - Chat-style display, but each generation is conditioned on the
        current prompt only so it stays closer to the TinyStories training mix
    """
    if not _GRADIO:
        raise RuntimeError("gradio not installed, uv pip install gradio")
    
    # UI layout
    with gr.Blocks(title="MiniGPT TinyStories Playground") as demo:
        gr.Markdown(
            "# MiniGPT · TinyStories\n"
            "A small JAX language model for TinyStories-style continuations.\n"
            "> Prompt it with a story starter. This is continuation-first, not instruction-tuned chat."
        )
 
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Story continuations", height=420)
                msg_box = gr.Textbox(
                    placeholder="Type a story starter and press Enter…",
                    label="",
                    show_label=False,
                    lines=2,
                )
                with gr.Row():
                    send_btn  = gr.Button("Send ▶", variant="primary")
                    clear_btn = gr.Button("Clear 🗑")
 
            with gr.Column(scale=1):
                gr.Markdown("### Generation settings")
                story_prefix = gr.Textbox(
                    label="Story prefix",
                    value="",
                    lines=3,
                    placeholder="Optional text prepended before each prompt",
                )
                temperature  = gr.Slider(0.1, 2.0, value=0.8,  step=0.05,
                                         label="Temperature")
                top_k_slider = gr.Slider(0,   200,  value=50,   step=5,
                                         label="Top-k  (0 = off)")
                top_p_slider = gr.Slider(0.0, 1.0,  value=0.9,  step=0.05,
                                         label="Top-p  (nucleus)")

        # wire up events inside the Blocks context for Gradio 6.
        def user_submit(message, history):
            updated = list(history)
            updated.append({"role": "user", "content": message})
            return "", updated

        def bot_stream(history, sys_p, temp, top_k, top_p):
            updated = list(history)
            user_msg = str(updated[-1]["content"])
            updated.append({"role": "assistant", "content": ""})

            gen_cfg = GenerationConfig(
                max_new_tokens=256,
                temperature=temp,
                top_k=int(top_k),
                top_p=top_p,
            )

            prompt = _build_story_prompt(sys_p, user_msg)

            partial = ""
            for chunk in generate_stream(model, tokenizer, prompt, gen_cfg):
                partial = chunk
                updated[-1]["content"] = partial
                yield updated

        def clear_history():
            return [], ""

        msg_box.submit(user_submit, [msg_box, chatbot], [msg_box, chatbot],
                       queue=False).then(
            bot_stream,
            [chatbot, story_prefix, temperature, top_k_slider, top_p_slider],
            chatbot,
        )
        send_btn.click(user_submit, [msg_box, chatbot], [msg_box, chatbot],
                       queue=False).then(
            bot_stream,
            [chatbot, story_prefix, temperature, top_k_slider, top_p_slider],
            chatbot,
        )
        clear_btn.click(clear_history, None, [chatbot, msg_box])
    
    return demo

# cli demo
 
def cli_demo(model: GPT, tokenizer: TinyStoriesTokenizer) -> None:
    #Interactive command-line generation loop.
    gen_cfg = GenerationConfig(temperature=0.8, top_k=40, top_p=0.9)
 
    print("\n" + "=" * 60)
    print("  MiniGPT — TinyStories CLI  (Ctrl+C to exit)")
    print("=" * 60)
    print("  Commands:")
    print("    <story start>  →  generate TinyStories continuation")
    print("    :samples       →  run built-in TinyStories prompts")
    print("    :temp 0.5      →  set temperature")
    print("    :topk 40       →  set top-k")
    print("    :topp 0.9      →  set top-p")
    print("    :quit          →  exit")
    print()

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
 
        if not prompt:
            continue
 
        if prompt.startswith(":quit"):
            break
        elif prompt.startswith(":samples"):
            run_sample_prompts(model, tokenizer, gen_cfg)
            continue
        elif prompt.startswith(":temp "):
            gen_cfg.temperature = float(prompt.split()[1])
            print(f"  temperature → {gen_cfg.temperature}")
            continue
        elif prompt.startswith(":topk "):
            gen_cfg.top_k = int(prompt.split()[1])
            print(f"  top_k → {gen_cfg.top_k}")
            continue
        elif prompt.startswith(":topp "):
            gen_cfg.top_p = float(prompt.split()[1])
            print(f"  top_p → {gen_cfg.top_p}")
            continue

        print("\nGenerating...\n")
        t0 = time.perf_counter()
        display_prompt = _display_prompt(tokenizer, prompt)
        sys.stdout.write(display_prompt)
        sys.stdout.flush()
        previous = ""
        for partial in generate_stream(model, tokenizer, prompt, gen_cfg):
            delta = partial[len(previous):]
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
            previous = partial
        elapsed = time.perf_counter() - t0
        print(f"\n\n  [{elapsed:.1f}s  |  "
              f"temp={gen_cfg.temperature}  top_k={gen_cfg.top_k}  "
              f"top_p={gen_cfg.top_p}]\n")
        
# main
def _build_demo_model(ckpt_dir: str | None = None) -> tuple[GPT, TinyStoriesTokenizer]:
    """
    Build or load a model for demo purposes.
    If a checkpoint exists, load it.  Otherwise use random weights.
    """
    # Load tokenizer
    tok = TinyStoriesTokenizer()

    cfg  = ModelConfig(
        vocab_size   = tok.vocab_size,
        max_seq_len  = DEFAULT_MODEL.max_seq_len,
        d_model      = DEFAULT_MODEL.d_model,
        n_heads      = DEFAULT_MODEL.n_heads,
        n_layers     = DEFAULT_MODEL.n_layers,
        d_ff         = DEFAULT_MODEL.d_ff,
        dropout      = 0.0,
        attn_dropout = 0.0,
        tie_weights  = DEFAULT_MODEL.tie_weights,
    )

    ckpt_dir = os.path.abspath(ckpt_dir or DEFAULT_TRAIN.checkpoint_dir)
    if _ORBAX and os.path.isdir(ckpt_dir):
        try:
            model = load_model_from_checkpoint(ckpt_dir, cfg)
            return model, tok
        except Exception as e:
            print(f"  Could not load checkpoint: {e}")
 
    print("  Using randomly-initialised model (train first for coherent output).")
    rngs  = nnx.Rngs(params=0, dropout=0)
    model = GPT(cfg, rngs=rngs)
    return model, tok
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniGPT inference")
    parser.add_argument("--ui",   action="store_true", help="Launch Gradio UI")
    parser.add_argument("--samples", action="store_true",
                        help="Print built-in TinyStories sample prompts and exit")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint directory")
    args = parser.parse_args()
 
    print("Inference & TinyStories Playground")
    
    model, tokenizer = _build_demo_model(args.ckpt)
    n = model.count_params()
    print(f"  Model ready — {n:,} params  (~{n/1e6:.1f} M)\n")

    if args.samples:
        run_sample_prompts(model, tokenizer)
        if not args.ui:
            sys.exit(0)
 
    if args.ui:
        if not _GRADIO:
            print("Install gradio first:  uv pip install gradio")
            sys.exit(1)
        app = build_gradio_app(model, tokenizer)
        app.queue()
        app.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            theme=_build_gradio_theme(),
            css=GRADIO_CSS,
        )
    else:
        cli_demo(model, tokenizer)
