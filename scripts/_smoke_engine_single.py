"""CPU smoke test: load a checkpoint single-device, run chat() + chat_stream()."""
import os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

import jaxchat.model as model_lib  # noqa: E402
model_lib.configure_jax_runtime()

from jaxchat.engine import Engine  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "data/124m_sota_e2e/runs/rl"

t0 = time.time()
eng = Engine.from_run_dir(RUN, single_device=True)
print(f"[{time.time()-t0:6.1f}s] loaded. model_info: {eng.model_info()}", flush=True)

msgs = [{"role": "user", "content": "Say hello in one short sentence."}]

t1 = time.time()
reply = eng.chat(msgs, max_new_tokens=16, temperature=0.7, top_k=50, top_p=0.95, seed=1)
print(f"[{time.time()-t1:6.1f}s] chat() (cold, 16 tok): {reply!r}", flush=True)

t2 = time.time()
deltas = list(eng.chat_stream(msgs, max_new_tokens=12, temperature=0.7, top_k=50, top_p=0.95, seed=2))
print(f"[{time.time()-t2:6.1f}s] chat_stream() (warm, 12 tok): {(''.join(deltas))!r}", flush=True)
print("OK", flush=True)
