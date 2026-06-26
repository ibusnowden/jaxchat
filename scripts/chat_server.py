"""FastAPI inference server: chat with a trained jaxchat checkpoint over HTTP.

Endpoints
---------
- GET  /            → a self-contained chat web UI (no external dependencies).
- GET  /health      → {stage, step, ...model_info, run_dir, defaults}.
- GET  /api/model   → model_info + run_dir + serving defaults.
- POST /chat        → non-streaming chat (supports the local Python tool loop).
- POST /chat/stream → Server-Sent-Events stream of decoded text deltas.

The UI talks to /chat/stream for live token-by-token rendering and falls back
to /chat when the tool loop is enabled.
"""

from __future__ import annotations

import argparse
import json
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
from fastapi.responses import HTMLResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from jaxchat.engine import Engine  # noqa: E402


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>jaxchat · inference console</title>
<style>
  :root {
    --bg: #0b0d12;
    --bg-soft: #12151c;
    --panel: #161a22;
    --panel-2: #1b2029;
    --fg: #e6e9ef;
    --fg-dim: #a9b0bd;
    --muted: #6b7280;
    --border: #242a36;
    --border-soft: #1e2430;
    --accent: #7c9cff;
    --accent-2: #6366f1;
    --user-bg: #2a2f3a;
    --user-fg: #e6e9ef;
    --bot-bg: #161a22;
    --code-bg: #0a0c10;
    --code-fg: #c8d3e0;
    --ok: #34d399;
    --bad: #f87171;
    --max: 880px;
    --sidebar: 300px;
    --mono: ui-monospace, "SF Mono", "JetBrains Mono", "Fira Code", Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", system-ui, sans-serif;
    background: var(--bg);
    color: var(--fg);
    display: flex;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
  }
  /* ---------- sidebar ---------- */
  aside {
    width: var(--sidebar);
    flex: 0 0 var(--sidebar);
    background: var(--bg-soft);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    transition: transform .2s ease;
  }
  .brand {
    padding: 16px 18px;
    border-bottom: 1px solid var(--border-soft);
    display: flex; align-items: center; gap: 10px;
  }
  .logo {
    width: 30px; height: 30px; border-radius: 8px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    display: grid; place-items: center; color: #0b0d12; font-weight: 800; font-size: 15px;
  }
  .brand h1 { font-size: 15px; margin: 0; font-weight: 700; letter-spacing: .2px; }
  .brand .sub { font-size: 11px; color: var(--muted); }
  .panel { padding: 14px 16px; border-bottom: 1px solid var(--border-soft); }
  .panel h3 {
    font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
    color: var(--muted); margin: 0 0 10px; font-weight: 600;
  }
  .info-grid { display: grid; grid-template-columns: auto 1fr; gap: 6px 10px; font-size: 12px; }
  .info-grid .k { color: var(--muted); }
  .info-grid .v { color: var(--fg); font-family: var(--mono); text-align: right; }
  .badge-row { display: flex; gap: 6px; flex-wrap: wrap; }
  .badge {
    font-size: 11px; padding: 2px 8px; border-radius: 999px;
    border: 1px solid var(--border); background: var(--panel); color: var(--fg-dim);
  }
  .badge.stage { color: var(--accent); border-color: color-mix(in srgb, var(--accent) 40%, transparent); }
  .field { margin-bottom: 12px; }
  .field:last-child { margin-bottom: 0; }
  .field label {
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: 12px; color: var(--fg-dim); margin-bottom: 4px;
  }
  .field label .val { color: var(--accent); font-family: var(--mono); font-size: 11px; }
  input[type=range] {
    width: 100%; -webkit-appearance: none; appearance: none; height: 4px;
    background: var(--border); border-radius: 4px; outline: none;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%;
    background: var(--accent); cursor: pointer; border: 2px solid var(--bg);
  }
  input[type=range]::-moz-range-thumb {
    width: 14px; height: 14px; border-radius: 50%; background: var(--accent);
    cursor: pointer; border: 2px solid var(--bg);
  }
  input[type=number], textarea {
    width: 100%; background: var(--panel); color: var(--fg);
    border: 1px solid var(--border); border-radius: 8px; padding: 6px 8px;
    font: inherit; font-size: 12px; outline: none;
  }
  input[type=number]:focus, textarea:focus { border-color: var(--accent); }
  textarea { resize: vertical; min-height: 44px; }
  .switch { display: flex; align-items: center; justify-content: space-between; font-size: 12px; color: var(--fg-dim); }
  .toggle { position: relative; width: 34px; height: 18px; background: var(--border); border-radius: 999px; cursor: pointer; transition: background .15s; }
  .toggle.on { background: var(--accent-2); }
  .toggle::after {
    content: ""; position: absolute; top: 2px; left: 2px; width: 14px; height: 14px;
    background: #fff; border-radius: 50%; transition: transform .15s;
  }
  .toggle.on::after { transform: translateX(16px); }
  .btn {
    width: 100%; padding: 8px 10px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--panel); color: var(--fg); font: inherit; font-size: 12px;
    cursor: pointer; transition: background .12s, border-color .12s;
  }
  .btn:hover { background: var(--panel-2); border-color: var(--accent); }
  .btn.danger:hover { border-color: var(--bad); color: var(--bad); }
  .btn-row { display: flex; gap: 8px; }
  .btn-row .btn { width: auto; flex: 1; }
  .status { display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--muted); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
  .dot.live { background: var(--ok); box-shadow: 0 0 8px var(--ok); }
  .dot.busy { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
  .dot.off { background: var(--bad); }
  /* ---------- main ---------- */
  main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  header.topbar {
    height: 52px; flex: 0 0 52px; border-bottom: 1px solid var(--border);
    background: var(--bg-soft); display: flex; align-items: center; gap: 12px;
    padding: 0 16px; position: sticky; top: 0; z-index: 5;
  }
  .hamburger { display: none; background: none; border: 0; color: var(--fg); cursor: pointer; font-size: 18px; }
  .topbar .title { font-weight: 700; font-size: 14px; }
  .topbar .spacer { flex: 1; }
  .topbar .pill { font-size: 11px; padding: 3px 9px; border-radius: 999px; border: 1px solid var(--border); color: var(--fg-dim); font-family: var(--mono); }
  .stream-ind { display: none; align-items: center; gap: 6px; font-size: 11px; color: var(--accent); }
  .stream-ind.on { display: inline-flex; }
  .stream-ind .dot { background: var(--accent); box-shadow: 0 0 8px var(--accent); animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  .thread { flex: 1; overflow-y: auto; padding: 24px 16px 8px; scroll-behavior: smooth; }
  .stream-wrap { max-width: var(--max); margin: 0 auto; display: flex; flex-direction: column; gap: 18px; }
  .msg { display: flex; gap: 12px; max-width: var(--max); }
  .msg.user { flex-direction: row-reverse; }
  .avatar {
    width: 32px; height: 32px; border-radius: 8px; flex: 0 0 32px;
    display: grid; place-items: center; font-size: 13px; font-weight: 700;
  }
  .msg.assistant .avatar { background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: #0b0d12; }
  .msg.user .avatar { background: var(--user-bg); color: var(--user-fg); border: 1px solid var(--border); }
  .bubble-wrap { display: flex; flex-direction: column; gap: 4px; min-width: 0; flex: 1; }
  .msg.user .bubble-wrap { align-items: flex-end; }
  .bubble {
    padding: 11px 14px; border-radius: 14px; font-size: 14.5px; line-height: 1.6;
    max-width: 100%; word-wrap: break-word; overflow-wrap: anywhere;
  }
  .msg.assistant .bubble { background: var(--bot-bg); border: 1px solid var(--border-soft); border-top-left-radius: 4px; }
  .msg.user .bubble { background: var(--user-bg); border: 1px solid var(--border); border-top-right-radius: 4px; }
  .bubble pre {
    background: var(--code-bg); border: 1px solid var(--border-soft); border-radius: 8px;
    padding: 10px 12px; overflow-x: auto; margin: 6px 0; font-family: var(--mono);
    font-size: 12.5px; color: var(--code-fg); white-space: pre;
  }
  .bubble code { font-family: var(--mono); font-size: 12.5px; background: var(--code-bg);
    border: 1px solid var(--border-soft); padding: 1px 5px; border-radius: 4px; }
  .bubble pre code { background: none; border: 0; padding: 0; }
  .msg-tools { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px; }
  .tool-chip { font-size: 11px; padding: 2px 8px; border-radius: 8px; background: var(--panel); border: 1px solid var(--border); color: var(--fg-dim); font-family: var(--mono); }
  .msg-actions { display: none; gap: 6px; }
  .msg.assistant:hover .msg-actions { display: flex; }
  .act-btn { font-size: 11px; padding: 2px 8px; border-radius: 6px; background: var(--panel); border: 1px solid var(--border); color: var(--muted); cursor: pointer; }
  .act-btn:hover { color: var(--fg); border-color: var(--accent); }
  .empty { text-align: center; padding: 60px 20px; color: var(--muted); }
  .empty .big { font-size: 28px; font-weight: 700; color: var(--fg); margin-bottom: 8px; }
  .empty .sub { font-size: 14px; margin-bottom: 22px; }
  .examples { display: flex; flex-direction: column; gap: 8px; max-width: 460px; margin: 0 auto; }
  .example {
    text-align: left; padding: 11px 14px; border-radius: 12px; background: var(--panel);
    border: 1px solid var(--border); color: var(--fg-dim); cursor: pointer; font-size: 13.5px;
    transition: border-color .12s, color .12s;
  }
  .example:hover { border-color: var(--accent); color: var(--fg); }
  .typing { display: inline-flex; gap: 4px; padding: 2px 0; }
  .typing span { width: 6px; height: 6px; background: var(--muted); border-radius: 50%; animation: bounce 1.2s infinite ease-in-out; }
  .typing span:nth-child(2) { animation-delay: .15s; }
  .typing span:nth-child(3) { animation-delay: .3s; }
  @keyframes bounce { 0%,80%,100%{transform:scale(.4);opacity:.4} 40%{transform:scale(1);opacity:1} }
  /* ---------- composer ---------- */
  .composer { flex: 0 0 auto; border-top: 1px solid var(--border); background: var(--bg-soft); padding: 12px 16px 14px; }
  .composer-inner { max-width: var(--max); margin: 0 auto; }
  .composer textarea {
    width: 100%; resize: none; border: 1px solid var(--border); border-radius: 14px;
    padding: 12px 14px; font: inherit; font-size: 14.5px; line-height: 1.5; min-height: 48px;
    max-height: 200px; outline: none; background: var(--panel); color: var(--fg);
  }
  .composer textarea:focus { border-color: var(--accent); }
  .composer-bar { display: flex; align-items: center; gap: 10px; margin-top: 8px; }
  .composer-bar .hint { font-size: 11px; color: var(--muted); flex: 1; }
  .send, .stop {
    border: 0; border-radius: 10px; padding: 0 16px; height: 36px; font: inherit; font-weight: 600;
    cursor: pointer; display: inline-flex; align-items: center; gap: 6px;
  }
  .send { background: var(--accent-2); color: white; }
  .send:hover { background: var(--accent); }
  .send:disabled { opacity: .45; cursor: not-allowed; }
  .stop { background: var(--bad); color: white; }
  .stop:disabled { opacity: .45; cursor: not-allowed; }
  .err-banner { background: color-mix(in srgb, var(--bad) 12%, transparent); border: 1px solid var(--bad);
    color: #fecaca; padding: 8px 12px; border-radius: 10px; font-size: 12.5px; margin: 0 auto 10px; max-width: var(--max); display: none; }
  .err-banner.on { display: block; }
  /* ---------- responsive ---------- */
  @media (max-width: 860px) {
    aside { position: fixed; top: 0; left: 0; height: 100%; z-index: 50; transform: translateX(-100%); box-shadow: 0 0 40px rgba(0,0,0,.5); }
    aside.open { transform: translateX(0); }
    .hamburger { display: inline-block; }
    .scrim { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.5); z-index: 40; }
    .scrim.on { display: block; }
  }
</style>
</head>
<body>
<aside id="sidebar">
  <div class="brand">
    <div class="logo">jc</div>
    <div>
      <h1>jaxchat</h1>
      <div class="sub">inference console</div>
    </div>
  </div>

  <div class="panel">
    <h3>Model</h3>
    <div class="badge-row" id="stage-badges">
      <span class="badge stage" id="stage-badge">…</span>
      <span class="badge" id="params-badge">…</span>
    </div>
    <div class="info-grid" id="model-grid" style="margin-top:10px"></div>
  </div>

  <div class="panel">
    <h3>Sampling</h3>
    <div class="field">
      <label>temperature <span class="val" id="temp-val">0.70</span></label>
      <input type="range" id="temp" min="0" max="2" step="0.05" value="0.7">
    </div>
    <div class="field">
      <label>top-k <span class="val" id="topk-val">50</span></label>
      <input type="range" id="topk" min="0" max="200" step="1" value="50">
    </div>
    <div class="field">
      <label>top-p <span class="val" id="topp-val">0.95</span></label>
      <input type="range" id="topp" min="0" max="1" step="0.01" value="0.95">
    </div>
    <div class="field">
      <label>max new tokens <span class="val" id="maxn-val">256</span></label>
      <input type="range" id="maxn" min="16" max="1024" step="16" value="256">
    </div>
  </div>

  <div class="panel">
    <h3>Options</h3>
    <div class="field">
      <div class="switch">
        <span>local Python tools</span>
        <div class="toggle" id="tools-toggle" role="switch" aria-checked="false" tabindex="0"></div>
      </div>
    </div>
    <div class="field">
      <label>system prompt <span class="val" style="font-size:10px;color:var(--muted)">optional</span></label>
      <textarea id="sysprompt" rows="2" placeholder="e.g. You are a helpful, concise assistant."></textarea>
    </div>
  </div>

  <div class="panel" style="margin-top:auto">
    <div class="btn-row">
      <button class="btn" id="new-chat">New chat</button>
      <button class="btn danger" id="clear-btn">Clear</button>
    </div>
    <div class="status" style="margin-top:12px">
      <span class="dot" id="conn-dot"></span>
      <span id="conn-text">connecting…</span>
    </div>
  </div>
</aside>
<div class="scrim" id="scrim"></div>

<main>
  <header class="topbar">
    <button class="hamburger" id="hamburger" aria-label="menu">&#9776;</button>
    <span class="title">jaxchat</span>
    <span class="pill" id="model-pill">…</span>
    <span class="spacer"></span>
    <span class="stream-ind" id="stream-ind"><span class="dot"></span> streaming</span>
  </header>

  <div class="thread" id="thread">
    <div class="stream-wrap" id="stream"></div>
  </div>

  <div class="err-banner" id="err-banner"></div>

  <div class="composer">
    <div class="composer-inner">
      <textarea id="input" rows="1" placeholder="Send a message  (Enter to send, Shift+Enter for newline)" autofocus></textarea>
      <div class="composer-bar">
        <span class="hint" id="hint">Ready</span>
        <button class="stop" id="stop" style="display:none">Stop</button>
        <button class="send" id="send">Send</button>
      </div>
    </div>
  </div>
</main>

<script>
"use strict";
const $ = (id) => document.getElementById(id);
const stream = $("stream");
const thread = $("thread");
const input = $("input");
const send = $("send");
const stopBtn = $("stop");
const errBanner = $("err-banner");
const connDot = $("conn-dot");
const connText = $("conn-text");
const streamInd = $("stream-ind");
const modelPill = $("model-pill");
const modelGrid = $("model-grid");
const stageBadge = $("stage-badge");
const paramsBadge = $("params-badge");

let hist = [];
let generating = false;
let abortCtrl = null;
const DEFAULTS = { temperature: 0.7, top_k: 50, top_p: 0.95, max_new_tokens: 256, tools: false, sysprompt: "" };
const settings = Object.assign({}, DEFAULTS);

// ---- persistence ----
function load() {
  try {
    const s = JSON.parse(localStorage.getItem("jaxchat.settings") || "null");
    if (s) Object.assign(settings, s);
    const h = JSON.parse(localStorage.getItem("jaxchat.history") || "null");
    if (Array.isArray(h)) hist = h;
  } catch (e) {}
}
function saveSettings() { localStorage.setItem("jaxchat.settings", JSON.stringify(settings)); }
function saveHistory() { try { localStorage.setItem("jaxchat.history", JSON.stringify(hist.slice(-50))); } catch (e) {} }

// ---- settings wiring ----
function wireSlider(id, key, fmt) {
  const el = $(id);
  el.value = settings[key];
  const valEl = $(id + "-val");
  if (valEl) valEl.textContent = fmt ? fmt(settings[key]) : settings[key];
  el.addEventListener("input", () => {
    settings[key] = (el.type === "number" || el.type === "range") ? parseFloat(el.value) : el.value;
    if (valEl) valEl.textContent = fmt ? fmt(settings[key]) : settings[key];
    saveSettings();
  });
}
function wireUI() {
  wireSlider("temp", "temperature", (v) => Number(v).toFixed(2));
  wireSlider("topk", "top_k", (v) => String(v|0));
  wireSlider("topp", "top_p", (v) => Number(v).toFixed(2));
  wireSlider("maxn", "max_new_tokens", (v) => String(v|0));
  const toolsToggle = $("tools-toggle");
  toolsToggle.classList.toggle("on", settings.tools);
  toolsToggle.setAttribute("aria-checked", settings.tools ? "true" : "false");
  toolsToggle.addEventListener("click", () => {
    settings.tools = !settings.tools;
    toolsToggle.classList.toggle("on", settings.tools);
    toolsToggle.setAttribute("aria-checked", settings.tools ? "true" : "false");
    saveSettings();
  });
  const sysEl = $("sysprompt");
  sysEl.value = settings.sysprompt || "";
  sysEl.addEventListener("input", () => { settings.sysprompt = sysEl.value; saveSettings(); });
  $("new-chat").addEventListener("click", newChat);
  $("clear-btn").addEventListener("click", () => { if (confirm("Clear this conversation?")) newChat(); });
  $("hamburger").addEventListener("click", () => { $("sidebar").classList.toggle("open"); $("scrim").classList.toggle("on"); });
  $("scrim").addEventListener("click", () => { $("sidebar").classList.remove("open"); $("scrim").classList.remove("on"); });
}

// ---- model info ----
function fmtParams(n) {
  if (n >= 1e9) return (n/1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n/1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n/1e3).toFixed(0) + "K";
  return String(n);
}
function setConn(state) {
  connDot.className = "dot " + state;
  connText.textContent = state === "live" ? "online" : state === "busy" ? "generating…" : state === "off" ? "offline" : "connecting…";
}
async function loadModel() {
  try {
    const r = await fetch("/api/model");
    const m = await r.json();
    stageBadge.textContent = (m.stage || "model") + " · step " + (m.step ?? 0);
    paramsBadge.textContent = m.n_params_human || (m.n_params ? fmtParams(m.n_params) : "—");
    modelPill.textContent = (m.stage || "model") + " · " + (m.n_params_human || "—") + " · d" + (m.depth ?? "?");
    const rows = [
      ["stage", m.stage], ["step", m.step],
      ["params", m.n_params_human || (m.n_params ? fmtParams(m.n_params) : "—")],
      ["depth", m.depth], ["layers", m.n_layers], ["d_model", m.d_model],
      ["heads", m.n_heads], ["kv_heads", m.n_kv_heads || "—"], ["recur", m.n_recurrence || 1],
      ["vocab", m.vocab_size], ["seq_len", m.max_seq_len], ["tokenizer", m.tokenizer_name],
      ["softcap", m.logit_softcap], ["dtype", (m.dtype || "").replace("jax.numpy.", "").replace("<", "").replace(">", "")],
    ];
    modelGrid.innerHTML = rows.map(([k, v]) => `<span class="k">${k}</span><span class="v">${v == null ? "—" : v}</span>`).join("");
    setConn("live");
  } catch (e) {
    setConn("off");
    modelPill.textContent = "model offline";
  }
}

// ---- rendering ----
function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function renderMarkdown(text) {
  // minimal, safe markdown: fenced code, inline code, bold, line breaks
  let s = escapeHtml(text);
  // fenced code blocks
  s = s.replace(/```([a-zA-Z0-9]*)\\n([\\s\\S]*?)```/g, (_, lang, code) => `<pre><code>${code.replace(/\\n$/, "")}</code></pre>`);
  // inline code
  s = s.replace(/`([^`\\n]+)`/g, "<code>$1</code>");
  // bold
  s = s.replace(/\\*\\*(.+?)\\*\\*/g, "<strong>$1</strong>");
  return s;
}
function emptyState() {
  stream.innerHTML = "";
  const ex = [
    "Hello! What can you do?",
    "Write a haiku about gradient descent.",
    "Explain transformers in two sentences.",
    "What is 37 * 23? Think step by step.",
  ];
  stream.innerHTML = `<div class="empty"><div class="big">jaxchat</div><div class="sub">A from-scratch JAX GPT chatbot. Ask anything, or try:</div>` +
    `<div class="examples">` + ex.map((t) => `<button class="example">${escapeHtml(t)}</button>`).join("") + `</div></div>`;
  stream.querySelectorAll(".example").forEach((b) => b.addEventListener("click", () => { input.value = b.textContent; autoresize(); input.focus(); }));
}
function addMsg(role, text) {
  const empty = $(".empty", stream); if (empty) stream.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const av = document.createElement("div");
  av.className = "avatar";
  av.textContent = role === "user" ? "you" : "jc";
  const bw = document.createElement("div");
  bw.className = "bubble-wrap";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = role === "user" ? escapeHtml(text) : renderMarkdown(text);
  bw.appendChild(bubble);
  if (role === "assistant") {
    const actions = document.createElement("div");
    actions.className = "msg-actions";
    const copy = document.createElement("button");
    copy.className = "act-btn"; copy.textContent = "copy";
    copy.addEventListener("click", () => navigator.clipboard.writeText(bubble._raw || text));
    const regen = document.createElement("button");
    regen.className = "act-btn"; regen.textContent = "regenerate";
    regen.addEventListener("click", () => regenerate(wrap, bubble));
    actions.appendChild(copy); actions.appendChild(regen);
    bw.appendChild(actions);
  }
  wrap.appendChild(av); wrap.appendChild(bw);
  stream.appendChild(wrap);
  bubble._raw = text;
  thread.scrollTop = thread.scrollHeight;
  return bubble;
}
function addTyping() {
  const bubble = addMsg("assistant", "");
  bubble.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
  return bubble;
}
function toolsChip(events) {
  if (!events || !events.length) return;
  const wrap = stream.lastChild.querySelector(".bubble-wrap");
  const tools = document.createElement("div");
  tools.className = "msg-tools";
  events.forEach((ev) => {
    const chip = document.createElement("span");
    chip.className = "tool-chip";
    const ok = ev.ok ? "ok" : "err";
    chip.textContent = "python " + ok + ": " + (ev.output || "").slice(0, 80);
    chip.title = ev.code || "";
    tools.appendChild(chip);
  });
  wrap.appendChild(tools);
}

// ---- composer ----
function autoresize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
}
function setGenerating(on) {
  generating = on;
  send.style.display = on ? "none" : "";
  stopBtn.style.display = on ? "" : "none";
  input.disabled = false;
  setConn(on ? "busy" : "live");
  streamInd.classList.toggle("on", on);
  $("hint").textContent = on ? "Generating…" : "Ready";
}
function buildMessages(text) {
  const msgs = [];
  if (settings.sysprompt && settings.sysprompt.trim()) msgs.push({ role: "system", content: settings.sysprompt.trim() });
  for (const m of hist) msgs.push(m);
  msgs.push({ role: "user", content: text });
  return msgs;
}
function showError(msg) {
  errBanner.textContent = msg;
  errBanner.classList.add("on");
  setTimeout(() => errBanner.classList.remove("on"), 6000);
}
function clearError() { errBanner.classList.remove("on"); errBanner.textContent = ""; }

async function streamReply(messages, bubble) {
  abortCtrl = new AbortController();
  const seed = Math.floor(Math.random() * 2147483647);
  const payload = {
    messages,
    max_new_tokens: Math.round(settings.max_new_tokens),
    temperature: parseFloat(settings.temperature),
    top_k: Math.round(settings.top_k),
    top_p: parseFloat(settings.top_p),
    seed,
  };
  // non-streaming path when tools are enabled
  if (settings.tools) {
    payload.tools = true;
    const r = await fetch("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal: abortCtrl.signal });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || ("HTTP " + r.status));
    bubble.innerHTML = renderMarkdown(j.reply || "(empty reply)");
    bubble._raw = j.reply || "";
    if (j.events && j.events.length) toolsChip(j.events);
    return j.reply || "";
  }
  const res = await fetch("/chat/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal: abortCtrl.signal });
  if (!res.ok) {
    const j = await res.json().catch(() => ({}));
    throw new Error(j.detail || ("HTTP " + res.status));
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let full = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\\n\\n")) !== -1) {
      const line = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (!data) continue;
      let ev;
      try { ev = JSON.parse(data); } catch (e) { continue; }
      if (ev.delta) { full += ev.delta; bubble.innerHTML = renderMarkdown(full); bubble._raw = full; thread.scrollTop = thread.scrollHeight; }
      if (ev.done) { full = ev.reply || full; bubble._raw = full; bubble.innerHTML = renderMarkdown(full); if (ev.events && ev.events.length) toolsChip(ev.events); }
      if (ev.error) throw new Error(ev.error);
    }
  }
  return full;
}

async function send(text) {
  text = (text == null ? input.value : text).trim();
  if (!text || generating) return;
  clearError();
  input.value = ""; autoresize(); input.focus();
  hist.push({ role: "user", content: text });
  addMsg("user", text);
  saveHistory();
  const bubble = addTyping();
  setGenerating(true);
  const messages = buildMessages(text);
  try {
    const reply = await streamReply(messages, bubble);
    if (!reply) { bubble.innerHTML = "<span style='color:var(--muted)'>(empty reply)</span>"; }
    hist.push({ role: "assistant", content: reply });
    saveHistory();
  } catch (e) {
    if (e.name === "AbortError") {
      bubble.innerHTML = "<span style='color:var(--muted)'>[stopped]</span>";
      if (bubble._raw) hist.push({ role: "assistant", content: bubble._raw });
    } else {
      bubble.innerHTML = "<span style='color:var(--bad)'>[error: " + escapeHtml(String(e.message || e)) + "]</span>";
      showError(String(e.message || e));
    }
    saveHistory();
  } finally {
    setGenerating(false);
    abortCtrl = null;
  }
}
async function regenerate(wrap, bubble) {
  // drop the last assistant message and re-run from the preceding user turn
  const lastAssistant = hist.length ? hist[hist.length - 1] : null;
  if (!lastAssistant || lastAssistant.role !== "assistant") return;
  hist.pop();
  const lastUser = hist[hist.length - 1];
  bubble.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
  setGenerating(true);
  try {
    const reply = await streamReply(buildMessages(lastUser ? lastUser.content : ""), bubble);
    if (!reply) bubble.innerHTML = "<span style='color:var(--muted)'>(empty reply)</span>";
    hist.push({ role: "assistant", content: reply });
    saveHistory();
  } catch (e) {
    bubble.innerHTML = "<span style='color:var(--bad)'>[error: " + escapeHtml(String(e.message || e)) + "]</span>";
  } finally {
    setGenerating(false);
  }
}
function newChat() {
  hist = []; saveHistory(); emptyState();
  $("sidebar").classList.remove("open"); $("scrim").classList.remove("on");
}

// ---- bootstrap ----
input.addEventListener("input", autoresize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
send.addEventListener("click", () => send());
stopBtn.addEventListener("click", () => { if (abortCtrl) abortCtrl.abort(); });
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && generating && abortCtrl) abortCtrl.abort();
});
// restore history
load();
wireUI();
if (hist.length) {
  hist.forEach((m) => { const b = addMsg(m.role, m.content); });
} else {
  emptyState();
}
loadModel();
autoresize();
input.focus();
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


def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


def build_app(
    engine: Engine,
    default_max_new_tokens: int,
    default_tools: bool = False,
    run_dir: str = "",
) -> FastAPI:
    app = FastAPI(title="jaxchat", version="0.2")
    lock = threading.Lock()

    def resolve_defaults(req: ChatRequest) -> dict:
        return {
            "max_new_tokens": req.max_new_tokens or default_max_new_tokens,
            "temperature": req.temperature,
            "top_k": req.top_k,
            "top_p": req.top_p,
            "seed": req.seed,
        }

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    @app.get("/health")
    def health() -> dict:
        info = engine.model_info()
        info["run_dir"] = run_dir
        info["max_new_tokens"] = default_max_new_tokens
        info["tools"] = default_tools
        return info

    @app.get("/api/model")
    def model() -> dict:
        info = engine.model_info()
        info["run_dir"] = run_dir
        info["max_new_tokens"] = default_max_new_tokens
        info["tools"] = default_tools
        return info

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        kw = resolve_defaults(req)
        with lock:
            if req.tools or default_tools:
                result = engine.chat_with_tools(msgs, **kw)
                return ChatResponse(reply=result["reply"], events=result["events"])
            reply = engine.chat(msgs, **kw)
        return ChatResponse(reply=reply, events=[])

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest):
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        kw = resolve_defaults(req)

        def gen():
            # Tool loop is not streamed; emit a single delta then done.
            if req.tools or default_tools:
                with lock:
                    result = engine.chat_with_tools(msgs, **kw)
                yield _sse({"delta": result["reply"]})
                yield _sse({"done": True, "reply": result["reply"], "events": result.get("events", [])})
                return
            with lock:
                try:
                    full = ""
                    for delta in engine.chat_stream(msgs, **kw):
                        full += delta
                        yield _sse({"delta": delta})
                    yield _sse({"done": True, "reply": full, "events": []})
                except Exception as exc:  # noqa: BLE001
                    yield _sse({"error": str(exc)})

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

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
    parser.add_argument(
        "--single-device",
        action="store_true",
        help="Collapse any data-parallel mesh to a single replicated device. Use for interactive inference on one GPU or on CPU when the checkpoint was trained across multiple devices.",
    )
    args = parser.parse_args(argv)

    engine = Engine.from_run_dir(
        args.run_dir,
        stage=args.stage,
        tokenizer_path=args.tokenizer_json,
        single_device=args.single_device,
    )
    print(
        f"Loaded {engine.stage} stage @ step {engine.step}; serving on http://{args.host}:{args.port}"
    )
    app = build_app(
        engine,
        default_max_new_tokens=args.max_new_tokens,
        default_tools=args.tools,
        run_dir=os.path.abspath(args.run_dir),
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
