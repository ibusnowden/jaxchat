"""Smoke test for chat_server plumbing with a fake engine (no model load)."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from fastapi.testclient import TestClient  # noqa: E402
import scripts.chat_server as cs  # noqa: E402


class FakeEngine:
    stage = "rl"
    step = 79
    _n_params = 124000000

    def model_info(self):
        return {
            "stage": "rl", "step": 79, "n_params": 124000000, "n_params_human": "124.00M",
            "depth": 8, "n_layers": 8, "d_model": 512, "n_heads": 8, "n_kv_heads": 0,
            "n_recurrence": 1, "vocab_size": 32768, "max_seq_len": 1024,
            "tokenizer_name": "fineweb32k", "logit_softcap": 15.0, "dtype": "bfloat16",
        }

    def chat(self, messages, **kw):
        # Echo the last user message to prove wiring + kwargs arrive.
        last = messages[-1]["content"]
        return f"Hello from FakeEngine! You said: {last} (temp={kw.get('temperature')}, topk={kw.get('top_k')})"

    def chat_stream(self, messages, **kw):
        for w in ["Hello", " there", ".", " You", " said:", f" {messages[-1]['content']}"]:
            yield w

    def chat_with_tools(self, messages, **kw):
        return {"reply": "42", "events": [{"type": "python", "code": "2+2", "ok": True, "output": "4"}]}


app = cs.build_app(FakeEngine(), default_max_new_tokens=64, default_tools=False, run_dir="/fake/run")
c = TestClient(app)

fail = 0
def check(name, ok, detail=""):
    global fail
    print(("  OK  " if ok else "FAIL  ") + name + (f"  -- {detail}" if detail and not ok else ""))
    if not ok: fail += 1

# 1. UI HTML
r = c.get("/")
check("GET / serves HTML", r.status_code == 200 and "<html" in r.text and "jaxchat" in r.text)
check("UI has sidebar model panel", "Model" in r.text and "Sampling" in r.text)
check("UI has streaming endpoint ref", "/chat/stream" in r.text)
check("UI has tools toggle", 'id="tools-toggle"' in r.text)
check("UI has system prompt", 'id="sysprompt"' in r.text)
check("UI has regenerate/copy actions", "regenerate" in r.text and "copy" in r.text)

# 2. health
r = c.get("/health")
j = r.json()
check("GET /health", r.status_code == 200 and j["stage"] == "rl" and j["step"] == 79, str(j))
check("/health has model_info", "n_params_human" in j and "d_model" in j and "run_dir" in j, str(j))

# 3. api/model
r = c.get("/api/model")
j = r.json()
check("GET /api/model", j["n_params"] == 124000000 and j["vocab_size"] == 32768, str(j))

# 4. /chat non-streaming
r = c.post("/chat", json={"messages": [{"role": "user", "content": "hi there"}], "temperature": 0.3, "top_k": 10, "seed": 1})
j = r.json()
check("POST /chat", r.status_code == 200 and "Hello from FakeEngine" in j["reply"], str(j))
check("/chat passes sampling kwargs", "temp=0.3" in j["reply"] and "topk=10" in j["reply"], j["reply"])

# 5. /chat with tools
r = c.post("/chat", json={"messages": [{"role": "user", "content": "what is 2+2"}], "tools": True})
j = r.json()
check("POST /chat tools", j["reply"] == "42" and j["events"] and j["events"][0]["code"] == "2+2", str(j))

# 6. /chat/stream SSE
r = c.post("/chat/stream", json={"messages": [{"role": "user", "content": "stream test"}], "seed": 5})
check("POST /chat/stream status", r.status_code == 200, str(r.status_code))
check("SSE content-type", r.headers.get("content-type", "").startswith("text/event-stream"), r.headers.get("content-type"))
import json as _json
deltas = []
done = None
for block in r.text.split("\n\n"):
    line = block.strip()
    if not line.startswith("data:"):
        continue
    ev = _json.loads(line[5:].strip())
    if "delta" in ev: deltas.append(ev["delta"])
    if ev.get("done"): done = ev
check("SSE has deltas", "".join(deltas) == "Hello there. You said: stream test", repr(deltas))
check("SSE done event has full reply", done is not None and done.get("reply") == "Hello there. You said: stream test", str(done))

print()
print("FAILURES:", fail if fail else "none")
sys.exit(1 if fail else 0)
