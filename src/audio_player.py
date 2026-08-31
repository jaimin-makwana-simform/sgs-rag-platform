"""Gapless audio-queue component for Streamlit.

Streamlit's ``st.audio`` plays a single finished clip and can't play a growing
stream, so this renders a tiny HTML/JS component that takes an ordered list of
base64 MP3 clips (one per synthesized sentence) and plays them **back-to-back with
no gaps** — each clip starts the next on its ``ended`` event. Autoplay is allowed
because playback follows the user's button/record gesture; if a browser still
blocks it, a "Play answer" button appears.
"""

from __future__ import annotations

import base64
import json
import urllib.parse

import streamlit.components.v1 as components


def render_audio_queue(clips: list[bytes], *, height: int = 56) -> None:
    """Render + autoplay an ordered list of MP3 clips gaplessly."""
    if not clips:
        return
    b64 = [base64.b64encode(c).decode("ascii") for c in clips]
    clips_json = json.dumps(b64)

    html = """
<div style="font:13px/1.4 system-ui,sans-serif;color:#444;display:flex;
            align-items:center;gap:8px">
  <button id="play" style="display:none;padding:4px 10px;border-radius:6px;
          border:1px solid #ccc;background:#f6f6f6;cursor:pointer">▶ Play answer</button>
  <span id="status">🔊 preparing…</span>
</div>
<script>
const clips = __CLIPS__;
let i = 0;
const status = document.getElementById("status");
const playBtn = document.getElementById("play");

function playNext() {
  if (i >= clips.length) { status.textContent = "✓ finished"; return; }
  const audio = new Audio("data:audio/mp3;base64," + clips[i]);
  status.textContent = "🔊 playing " + (i + 1) + " / " + clips.length;
  audio.onended = () => { i++; playNext(); };
  audio.onerror = () => { i++; playNext(); };   // skip a bad clip, keep going
  audio.play().catch(() => {                     // autoplay blocked → offer a button
    status.textContent = "🔇 autoplay blocked";
    playBtn.style.display = "inline-block";
    playBtn.onclick = () => { playBtn.style.display = "none"; audio.play(); };
  });
}
playNext();
</script>
""".replace("__CLIPS__", clips_json)

    components.html(html, height=height)


def render_voice_stream(
    question: str,
    backend_url: str,
    *,
    overrides: dict | None = None,
    height: int = 460,
) -> None:
    """Concurrent streaming voice via the FastAPI SSE backend.

    Opens an EventSource to ``{backend_url}/voice/stream`` and, from that single
    connection, renders streaming text, plays per-sentence audio gaplessly as it
    arrives (so sentence 1 speaks while later sentences are still synthesized), and
    lists citations. Keys never reach the browser — the backend holds them.
    """
    params = {"q": question}
    for key in ("top_k", "reranker_threshold"):
        if overrides and overrides.get(key) is not None:
            params[key] = overrides[key]
    url = f"{backend_url.rstrip('/')}/voice/stream?" + urllib.parse.urlencode(params)
    url_json = json.dumps(url)

    html = r"""
<div style="font:14px/1.5 system-ui,sans-serif;color:#1a1a1a">
  <div id="status" style="font-size:13px;color:#666;margin-bottom:6px">🔊 connecting…</div>
  <button id="play" style="display:none;margin-bottom:8px;padding:4px 10px;
          border-radius:6px;border:1px solid #ccc;background:#f6f6f6;cursor:pointer">
    ▶ Play answer</button>
  <div id="answer" style="white-space:pre-wrap;margin-bottom:12px"></div>
</div>
<script>
const url = __URL__;
const statusEl = document.getElementById("status");
const answerEl = document.getElementById("answer");
const playBtn = document.getElementById("play");

// --- gapless audio queue ---
const queue = [];
let playing = false;
function enqueue(b64) { queue.push(b64); if (!playing) playNext(); }
function playNext() {
  if (queue.length === 0) { playing = false; return; }
  playing = true;
  const audio = new Audio("data:audio/mp3;base64," + queue.shift());
  audio.onended = playNext;
  audio.onerror = playNext;
  audio.play().catch(() => {                 // autoplay blocked → offer a button
    playBtn.style.display = "inline-block";
    playBtn.onclick = () => { playBtn.style.display = "none"; audio.play(); };
  });
}

// --- minimal Markdown → HTML (bold/italic/code/headings/bullets) ---
function mdToHtml(md) {
  let s = md.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  const lines = s.split(/\n/);
  let html = "", inList = false;
  for (const line of lines) {
    const h = line.match(/^\s{0,3}#{1,6}\s*(.+)$/);
    const b = line.match(/^\s*[-*+]\s+(.*)$/);
    if (b) {
      if (!inList) { html += "<ul style='margin:4px 0 4px 18px'>"; inList = true; }
      html += "<li>" + b[1] + "</li>";
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      if (h) html += "<p style='margin:8px 0 4px'><strong>" + h[1] + "</strong></p>";
      else if (line.trim()) html += "<p style='margin:6px 0'>" + line + "</p>";
    }
  }
  if (inList) html += "</ul>";
  return html;
}

let rawMd = "";
let renderScheduled = false;
function scheduleRender() {
  if (renderScheduled) return;
  renderScheduled = true;
  requestAnimationFrame(() => { answerEl.innerHTML = mdToHtml(rawMd); renderScheduled = false; });
}

const es = new EventSource(url);
es.addEventListener("delta", (e) => { rawMd += e.data; scheduleRender(); });
es.addEventListener("audio", (e) => {
  statusEl.textContent = "🔊 speaking…";
  enqueue(e.data);
});
// Citations are rendered natively by Streamlit (always visible, not clipped by
// this fixed-height iframe), so the backend's "sources" event is ignored here.
es.addEventListener("done", () => {
  answerEl.innerHTML = mdToHtml(rawMd);   // final render
  statusEl.textContent = "✓ done";
  es.close();
});
es.addEventListener("error", (e) => {
  statusEl.textContent = "⚠️ " + (e.data || "stream error");
  es.close();
});
es.onerror = () => { statusEl.textContent = "⚠️ connection lost (is the backend running?)"; es.close(); };
</script>
""".replace("__URL__", url_json)

    components.html(html, height=height)
