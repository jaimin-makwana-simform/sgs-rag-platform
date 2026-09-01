/*
 * SGS agent avatar client (Azure real-time TTS Avatar).
 *
 * Served from the FastAPI backend's own origin and embedded in an iframe that is
 * granted `allow="autoplay"`, so it **auto-connects on load** (no button) and the
 * WebRTC video autoplays. Flow:
 *   1. On load -> fetch /avatar/token (ICE relay + short-lived Speech auth token).
 *   2. AvatarSynthesizer.startAvatarAsync(pc); video/audio tracks attach in pc.ontrack.
 *      Video is muted so it always autoplays; audio plays via a separate element and,
 *      only if a browser blocks audio autoplay, a one-tap "Enable sound" button appears.
 *   3. If a question (?q=) is present, open /voice/stream?...&audio=off, accumulate text
 *      deltas, split into sentences, and speak each via avatarSynthesizer.speakTextAsync().
 *      With no question the avatar just idles (visible from the start).
 *   4. Page unload / Disconnect -> avatarSynthesizer.close() (stops billing).
 *
 * The raw Speech key never reaches the browser — only relay creds + auth token.
 */
"use strict";

const stage = document.getElementById("stage");
const placeholder = document.getElementById("placeholder");
const statusEl = document.getElementById("status");
const answerEl = document.getElementById("answer");
const enableSoundBtn = document.getElementById("enable-sound");
const reconnectBtn = document.getElementById("reconnect");
const disconnectBtn = document.getElementById("disconnect");

let avatarSynthesizer = null;
let peerConnection = null;
let eventSource = null;

// Serialize speak() calls: the avatar speaks one utterance at a time.
const speakQueue = [];
let speaking = false;

function setStatus(text) { statusEl.textContent = text; }

/** Strip Markdown so the avatar speaks clean prose (mirrors src/speech.clean_for_speech). */
function cleanForSpeech(text) {
  return text
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")   // [text](url) -> text
    .replace(/\[[^\]]*\]/g, "")                // [source p.N] citations -> drop
    .replace(/\*\*([^*]+)\*\*/g, "$1")         // **bold**
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1$2") // *italic*
    .replace(/__([^_]+)__/g, "$1")
    .replace(/`([^`]+)`/g, "$1")               // `code`
    .replace(/^\s{0,3}#{1,6}\s*/gm, "")        // # headings
    .replace(/^\s*[-*+]\s+/gm, "")             // - bullets
    .replace(/[*_`#]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

async function drainSpeakQueue() {
  if (speaking) return;
  speaking = true;
  while (speakQueue.length > 0) {
    const clean = cleanForSpeech(speakQueue.shift());
    if (!clean) continue;
    setStatus("🗣️ speaking…");
    try {
      const result = await new Promise((resolve, reject) => {
        avatarSynthesizer.speakTextAsync(clean, resolve, reject);
      });
      // Surface a non-success reason instead of failing silently.
      if (result && result.reason === window.SpeechSDK.ResultReason.Canceled) {
        const details = window.SpeechSDK.CancellationDetails.fromResult(result);
        console.error("avatar speak canceled:", details.errorDetails || details.reason);
      }
    } catch (err) {
      console.error("speakTextAsync failed:", err);
    }
  }
  speaking = false;
  if (eventSource === null) setStatus("✓ done");
}

function enqueueSpeak(text) {
  speakQueue.push(text);
  drainSpeakQueue();
}

let pendingSource = null;
let lastQuestionId = 0;

/** Subscribe once to the question channel; speak the answer for each new question.
 *  Keeping this channel separate from the page URL is what lets the WebRTC avatar
 *  session persist across Streamlit reruns (no reconnect between questions). */
function subscribeQuestions() {
  setStatus("connected — ask a question to hear the avatar");
  pendingSource = new EventSource("/avatar/pending");
  pendingSource.addEventListener("question", (e) => {
    let item;
    try { item = JSON.parse(e.data); } catch (_) { return; }
    if (!item.q || item.id === lastQuestionId) return;   // ignore replays
    lastQuestionId = item.id;
    runAnswer(item);
  });
  pendingSource.onerror = () => { /* auto-reconnects; keep the avatar up */ };
}

/** Minimal Markdown -> HTML (bold/italic/code/headings/bullets), matching the
 *  streaming voice component so the on-screen answer is formatted, not raw '**' etc. */
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

/** Open the text-only RAG stream, show text live, and speak the whole answer once.
 *  (Speaking the full answer in a single utterance is more robust than chaining a
 *  speakTextAsync per sentence — the avatar SDK chunks/lip-syncs the long text itself.) */
function runAnswer(item) {
  if (eventSource) { eventSource.close(); eventSource = null; }
  answerEl.innerHTML = "";

  const streamParams = new URLSearchParams({ q: item.q, audio: "off" });
  if (item.top_k != null) streamParams.set("top_k", item.top_k);
  if (item.reranker_threshold != null) streamParams.set("reranker_threshold", item.reranker_threshold);
  eventSource = new EventSource("/voice/stream?" + streamParams.toString());

  let buffer = "";
  let renderScheduled = false;
  const scheduleRender = () => {
    if (renderScheduled) return;
    renderScheduled = true;
    requestAnimationFrame(() => { answerEl.innerHTML = mdToHtml(buffer); renderScheduled = false; });
  };

  eventSource.addEventListener("delta", (e) => {
    buffer += e.data;
    scheduleRender();                 // render formatted text live for responsiveness
    setStatus("✍️ generating answer…");
  });
  eventSource.addEventListener("done", () => {
    if (eventSource) { eventSource.close(); eventSource = null; }
    answerEl.innerHTML = mdToHtml(buffer);   // final formatted render
    if (buffer.trim()) enqueueSpeak(buffer);   // speak the complete answer in one go
    else setStatus("✓ done");
  });
  eventSource.addEventListener("error", (e) => {
    setStatus("⚠️ " + (e.data || "stream error"));
    if (eventSource) { eventSource.close(); eventSource = null; }
  });
  eventSource.onerror = () => {
    setStatus("⚠️ connection lost (is the backend running?)");
    if (eventSource) { eventSource.close(); eventSource = null; }
  };
}

function attachTrack(event) {
  const kind = event.track.kind;                 // "video" or "audio"
  const existing = document.getElementById("avatar-" + kind);
  if (existing) existing.remove();
  const el = document.createElement(kind);
  el.id = "avatar-" + kind;
  el.srcObject = event.streams[0];
  el.autoplay = true;
  if (kind === "video") {
    el.playsInline = true;
    el.muted = true;                             // muted video always autoplays
    placeholder.style.display = "none";
    stage.appendChild(el);
  } else {
    el.style.display = "none";                   // audio-only element
    document.body.appendChild(el);
    el.play().catch(() => {                      // autoplay blocked -> offer one tap
      enableSoundBtn.style.display = "inline-block";
      enableSoundBtn.onclick = () => {
        el.play().then(() => { enableSoundBtn.style.display = "none"; }).catch(() => {});
      };
    });
  }
}

async function connect() {
  reconnectBtn.style.display = "none";
  setStatus("connecting…");
  try {
    const resp = await fetch("/avatar/token");
    if (!resp.ok) throw new Error("token endpoint " + resp.status + ": " + (await resp.text()));
    const cfg = await resp.json();

    const SDK = window.SpeechSDK;
    const speechConfig = SDK.SpeechConfig.fromAuthorizationToken(cfg.speechToken, cfg.region);
    speechConfig.speechSynthesisVoiceName = cfg.voice;

    const avatarConfig = new SDK.AvatarConfig(cfg.character, cfg.style);
    avatarSynthesizer = new SDK.AvatarSynthesizer(speechConfig, avatarConfig);

    peerConnection = new RTCPeerConnection({ iceServers: cfg.iceServers });
    peerConnection.ontrack = attachTrack;        // fires once per media kind
    peerConnection.addTransceiver("video", { direction: "sendrecv" });
    peerConnection.addTransceiver("audio", { direction: "sendrecv" });

    await avatarSynthesizer.startAvatarAsync(peerConnection);

    disconnectBtn.style.display = "inline-block";
    setStatus("connected");
    subscribeQuestions();
  } catch (err) {
    console.error(err);
    setStatus("⚠️ " + (err && err.message ? err.message : err));
    reconnectBtn.style.display = "inline-block";
  }
}

function disconnect() {
  if (eventSource) { eventSource.close(); eventSource = null; }
  if (pendingSource) { pendingSource.close(); pendingSource = null; }
  speakQueue.length = 0;
  if (avatarSynthesizer) { try { avatarSynthesizer.close(); } catch (_) {} avatarSynthesizer = null; }
  if (peerConnection) { try { peerConnection.close(); } catch (_) {} peerConnection = null; }
  document.querySelectorAll("#avatar-video, #avatar-audio").forEach((n) => n.remove());
  placeholder.style.display = "";
  placeholder.textContent = "Disconnected.";
  disconnectBtn.style.display = "none";
  reconnectBtn.style.display = "inline-block";
  setStatus("disconnected");
}

reconnectBtn.addEventListener("click", connect);
disconnectBtn.addEventListener("click", disconnect);
window.addEventListener("pagehide", disconnect);
window.addEventListener("beforeunload", disconnect);

// Auto-connect on load (no manual "Connect" click).
if (window.SpeechSDK) {
  connect();
} else {
  window.addEventListener("load", () => {
    if (window.SpeechSDK) connect();
    else setStatus("⚠️ Speech SDK failed to load (check network/CDN access).");
  });
}
