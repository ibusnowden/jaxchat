"""FastAPI inference server: chat with a trained jaxchat checkpoint over HTTP."""

from __future__ import annotations

import argparse
import os
import sys
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jaxchat.model as model_lib  # noqa: E402

model_lib.configure_jax_runtime()

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from jaxchat.engine import Engine  # noqa: E402


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>jaxchat</title>
<style>
  :root {
    --bg: #fafaf9;
    --fg: #1f2937;
    --muted: #6b7280;
    --user-bg: #f3f4f6;
    --bot-bg: #ffffff;
    --border: #e5e7eb;
    --accent: #2563eb;
    --max: 760px;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", system-ui, sans-serif;
    background: var(--bg);
    color: var(--fg);
    display: flex;
    flex-direction: column;
    line-height: 1.5;
  }
  header {
    border-bottom: 1px solid var(--border);
    background: white;
    padding: .75rem 1rem;
    display: flex;
    align-items: center;
    gap: .75rem;
  }
  header h1 { font-size: 1rem; font-weight: 600; margin: 0; }
  header .badge {
    font-size: .75rem;
    color: var(--muted);
    padding: 2px 8px;
    border: 1px solid var(--border);
    border-radius: 999px;
  }
  main { flex: 1; overflow-y: auto; padding: 1.5rem 1rem 2rem; }
  .stream {
    max-width: var(--max);
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  .msg { display: flex; flex-direction: column; gap: .25rem; }
  .msg .role { font-size: .75rem; color: var(--muted); font-weight: 600; }
  .msg .body {
    padding: .75rem 1rem;
    border-radius: 12px;
    border: 1px solid var(--border);
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .msg.user .body { background: var(--user-bg); }
  .msg.assistant .body { background: var(--bot-bg); }
  .empty { color: var(--muted); text-align: center; padding: 4rem 0; }
  .typing { display: inline-flex; gap: 4px; padding: 4px 0; }
  .typing span {
    width: 6px; height: 6px;
    background: var(--muted);
    border-radius: 50%;
    animation: bounce 1.2s infinite ease-in-out;
  }
  .typing span:nth-child(2) { animation-delay: .15s; }
  .typing span:nth-child(3) { animation-delay: .3s; }
  @keyframes bounce {
    0%, 80%, 100% { transform: scale(0.4); opacity: .4; }
    40% { transform: scale(1); opacity: 1; }
  }
  footer { border-top: 1px solid var(--border); background: white; padding: 1rem; }
  form {
    max-width: var(--max);
    margin: 0 auto;
    display: flex;
    gap: .5rem;
    align-items: flex-end;
  }
  textarea {
    flex: 1;
    resize: none;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: .65rem .9rem;
    font: inherit;
    line-height: 1.4;
    min-height: 44px;
    max-height: 160px;
    outline: none;
    background: white;
  }
  textarea:focus { border-color: var(--accent); }
  button {
    background: var(--accent);
    color: white;
    border: 0;
    border-radius: 12px;
    padding: 0 1rem;
    font: inherit;
    font-weight: 600;
    height: 44px;
    cursor: pointer;
  }
  button:disabled { opacity: .5; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <h1>jaxchat</h1>
  <span class="badge" id="badge">loading…</span>
</header>
<main>
  <div class="stream" id="stream">
    <div class="empty" id="empty">Start a conversation.</div>
  </div>
</main>
<footer>
  <form id="f">
    <textarea id="m" rows="1" placeholder="Send a message" autofocus></textarea>
    <button id="send" type="submit">Send</button>
  </form>
</footer>
<script>
  const stream = document.getElementById('stream');
  const empty = document.getElementById('empty');
  const form = document.getElementById('f');
  const ta = document.getElementById('m');
  const send = document.getElementById('send');
  const badge = document.getElementById('badge');
  const hist = [];

  fetch('/health').then(r => r.json()).then(j => {
    badge.textContent = j.stage + ' · step ' + j.step;
  }).catch(() => { badge.textContent = 'offline'; });

  function autoresize() {
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
  }
  ta.addEventListener('input', autoresize);
  ta.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  function addMsg(role, text) {
    const e = document.getElementById('empty');
    if (e) { e.remove(); }
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + role;
    const r = document.createElement('div'); r.className = 'role';
    r.textContent = role === 'user' ? 'You' : 'jaxchat';
    const b = document.createElement('div'); b.className = 'body';
    b.textContent = text;
    wrap.appendChild(r); wrap.appendChild(b);
    stream.appendChild(wrap);
    wrap.scrollIntoView({block: 'end', behavior: 'smooth'});
    return b;
  }

  function addTyping() {
    const body = addMsg('assistant', '');
    body.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
    return body;
  }

  form.addEventListener('submit', async e => {
    e.preventDefault();
    const t = ta.value.trim();
    if (!t) return;
    hist.push({role: 'user', content: t});
    addMsg('user', t);
    ta.value = ''; autoresize();
    ta.disabled = true; send.disabled = true;
    const placeholder = addTyping();
    try {
      const r = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({messages: hist, seed: Math.floor(Math.random() * 2147483647)}),
      });
      const j = await r.json();
      placeholder.textContent = j.reply || '(empty reply)';
      hist.push({role: 'assistant', content: j.reply});
    } catch (err) {
      placeholder.textContent = '[error: ' + err + ']';
    } finally {
      ta.disabled = false; send.disabled = false;
      ta.focus();
    }
  });
</script>
</body>
</html>"""


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    max_new_tokens: int | None = None
    temperature: float = 0.7
    top_k: int | None = 50
    top_p: float | None = 0.95
    seed: int = 0
    tools: bool = False


class ChatResponse(BaseModel):
    reply: str
    events: list[dict] = Field(default_factory=list)


def build_app(engine: Engine, default_max_new_tokens: int, default_tools: bool = False) -> FastAPI:
    app = FastAPI(title="jaxchat", version="0.1")
    lock = threading.Lock()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    @app.get("/health")
    def health() -> dict:
        return {"stage": engine.stage, "step": int(engine.step)}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        with lock:
            if req.tools or default_tools:
                result = engine.chat_with_tools(
                    msgs,
                    max_new_tokens=req.max_new_tokens or default_max_new_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    seed=req.seed,
                )
                return ChatResponse(reply=result["reply"], events=result["events"])
            reply = engine.chat(
                    msgs,
                    max_new_tokens=req.max_new_tokens or default_max_new_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    seed=req.seed,
                )
        return ChatResponse(reply=reply, events=[])

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HTTP chat server backed by a jaxchat checkpoint.")
    parser.add_argument("--run-dir", required=True, help="Directory produced by base_train / chat_sft / chat_rl.")
    parser.add_argument("--stage", default=None, choices=(None, "base", "sft", "rl"))
    parser.add_argument("--tokenizer-json", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--tools", action="store_true", help="Enable local Python tool execution by default.")
    args = parser.parse_args(argv)

    engine = Engine.from_run_dir(args.run_dir, stage=args.stage, tokenizer_path=args.tokenizer_json)
    print(f"Loaded {engine.stage} stage @ step {engine.step}; serving on http://{args.host}:{args.port}")
    app = build_app(engine, default_max_new_tokens=args.max_new_tokens, default_tools=args.tools)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
