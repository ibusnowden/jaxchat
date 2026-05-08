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
from pydantic import BaseModel  # noqa: E402

from jaxchat.engine import Engine  # noqa: E402


_INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>jaxchat</title>
<style>
body{font-family:system-ui,sans-serif;max-width:720px;margin:2em auto;padding:0 1em}
#log{border:1px solid #ccc;padding:1em;height:60vh;overflow-y:auto;white-space:pre-wrap}
.user{color:#024}.bot{color:#240}
form{display:flex;gap:.5em;margin-top:1em}
input{flex:1;padding:.5em;font-size:1em}
button{padding:.5em 1em}
</style></head><body>
<h2>jaxchat</h2>
<div id="log"></div>
<form id="f"><input id="m" autocomplete="off" placeholder="say something"><button>send</button></form>
<script>
const log=document.getElementById('log'),f=document.getElementById('f'),m=document.getElementById('m');
const hist=[];
function add(role,text){const d=document.createElement('div');d.className=role;d.textContent=role+'> '+text;log.appendChild(d);log.scrollTop=log.scrollHeight}
f.onsubmit=async(e)=>{e.preventDefault();const t=m.value.trim();if(!t)return;hist.push({role:'user',content:t});add('user',t);m.value='';m.disabled=true;
  const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:hist})});
  const j=await r.json();hist.push({role:'assistant',content:j.reply});add('bot',j.reply);m.disabled=false;m.focus()};
</script></body></html>"""


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


class ChatResponse(BaseModel):
    reply: str


def build_app(engine: Engine, default_max_new_tokens: int) -> FastAPI:
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
            reply = engine.chat(
                msgs,
                max_new_tokens=req.max_new_tokens or default_max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                seed=req.seed,
            )
        return ChatResponse(reply=reply)

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HTTP chat server backed by a jaxchat checkpoint.")
    parser.add_argument("--run-dir", required=True, help="Directory produced by base_train / chat_sft / chat_rl.")
    parser.add_argument("--stage", default=None, choices=(None, "base", "sft", "rl"))
    parser.add_argument("--tokenizer-json", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args(argv)

    engine = Engine.from_run_dir(args.run_dir, stage=args.stage, tokenizer_path=args.tokenizer_json)
    print(f"Loaded {engine.stage} stage @ step {engine.step}; serving on http://{args.host}:{args.port}")
    app = build_app(engine, default_max_new_tokens=args.max_new_tokens)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
