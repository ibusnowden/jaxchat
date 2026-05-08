# jaxchat

A from-scratch JAX implementation of a small GPT-style chatbot, trained end-to-end:
tokenizer → base pretraining → SFT → GRPO RL → eval → inference. The full pipeline
runs on a single H100 or RTX 6000 in a few hours and produces a checkpoint you can
chat with from a CLI or a small FastAPI web UI.

## Pipeline

`runs/speedrun_d4.sh` chains nine stages, each a separate entry point under
`scripts/`. Earlier stages are skipped on rerun if their outputs already exist
under `data/d4_speedrun/`.

| # | Stage | Script | Notes |
|---|-------|--------|-------|
| 1 | Tokenizer | `scripts.tok_train` | BPE, vocab=8192, trained on FineWeb-Edu |
| 2 | Pack corpus | `data.cached_fineweb` | 150M train / 1M val tokens |
| 3 | Base pretraining | `scripts.base_train` | preset `d4` (~10M params) |
| 4 | Base eval | `scripts.base_eval` | BPB + CORE-subset (n=200) |
| 5 | SFT data | `dev.synth_smoltalk` | 5k synthetic chat rows |
| 6 | SFT | `scripts.chat_sft` | 400 iters, seq_len=1024 |
| 7 | Chat eval | `scripts.chat_eval` | CORE-subset + GSM8K (n=50) |
| 8 | RL data + GRPO | `dev.synth_gsm8k`, `scripts.chat_rl` | 60 iters, M=4 G=4 |
| 9 | Final chat eval | `scripts.chat_eval` | with GSM8K (n=100) |

Loss curves and eval metrics are logged to Weights & Biases under the project
`jaxchat`.

## Layout

```
jaxchat/      model, attention, tokenizer, engine, checkpoint, presets
training/     pretraining loop, eval routines (used by scripts/)
scripts/      entry points: tok_train, base_train, base_eval,
              chat_sft, chat_eval, chat_rl, chat_cli, chat_server
data/         FineWeb caching + token packing
dev/          synthetic SFT (smoltalk) and RL (GSM8K) data generators
tasks/        eval harness (arc_easy, hellaswag, piqa, gsm8k, core)
runs/         SLURM launchers (h100_d4.sh, rtx_d4.sh) + speedrun_d4.sh
tests/        pytest suite (engine smoke, sft masking, training stack, ckpt)
```

`ablation_notes.md` tracks ideas under consideration and ones already ruled out.

## Replication

Prereqs: Python 3.12, [`uv`](https://github.com/astral-sh/uv), a CUDA GPU.

```bash
git clone git@github.com:ibusnowden/jaxchat.git
cd jaxchat
uv sync
```

Set wandb credentials once (either in `~/.netrc` via `wandb login`, or export
`WANDB_API_KEY`). The launchers default to `WANDB_PROJECT=jaxchat`.

Run the full pipeline:

```bash
# SLURM (single GPU)
sbatch runs/h100_d4.sh        # 1 x H100 80GB
sbatch runs/rtx_d4.sh         # 1 x RTX 6000

# Or directly on a node with a GPU visible
bash runs/speedrun_d4.sh
```

Outputs land in `data/d4_speedrun/`:

```
tokenizer/                # vocab + merges
fineweb8k/                # packed *.bin shards
runs/{base,sft,rl}/       # checkpoints, logs
```

## Inference

After the pipeline finishes (or after any of stages 3, 6, 8), point either
client at the matching run directory:

```bash
# Terminal REPL
python -m scripts.chat_cli --run-dir data/d4_speedrun/runs/rl

# FastAPI web UI on http://localhost:8000
python -m scripts.chat_server --run-dir data/d4_speedrun/runs/rl --port 8000
```

The web UI exposes `POST /chat` (`{messages, max_new_tokens?, temperature?, top_k?, top_p?, seed?}` → `{reply}`), plus `GET /health` and a minimal HTML chat page at `/`.

## Tests

```bash
uv run pytest tests/
```
