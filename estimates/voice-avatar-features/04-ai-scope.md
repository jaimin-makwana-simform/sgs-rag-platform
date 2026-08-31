# Architecture — Voice I/O + Talking Avatar

Both features wrap the **existing** RAG core (`src/pipelines/*` → `gpt-5-1`), which is unchanged.

## Turn-based data flow

```
 ┌─────────────────────────── Streamlit app (browser) ───────────────────────────┐
 │  🎙 mic (st.audio_input)                          🧑 avatar panel (JS component) │
 │        │ audio bytes                                     ▲ WebRTC video          │
 └────────┼────────────────────────────────────────────────┼──────────────────────┘
          ▼                                                  │ speak(text) + token
   Azure AI Speech STT ──► transcript ──► EXISTING RAG pipeline (Custom | Foundry IQ)
                                                │ answer text (+ citations)
                              ┌─────────────────┴───────────────────┐
                     avatar mode?                              voice-only mode?
                              ▼                                       ▼
                 Azure TTS Avatar (real-time)              Azure Speech neural TTS
                 lip-synced video via WebRTC                  audio → st.audio
                              ▲
                 ephemeral relay token  ◄──  backend token endpoint (FastAPI/Streamlit route)
                                              (keeps the Speech key server-side)
```

## Component notes

**STT (C1).** Azure Speech SDK server-side; `st.audio_input` captures the mic in-browser
(built into Streamlit — no custom recorder needed). Short utterances; batch recognize is fine
for turn-based (streaming STT deferred).

**TTS (C2).** Azure neural voice; return audio to `st.audio`. Only used when the avatar is off.

**Avatar (C3) — the crux.** Azure real-time TTS Avatar runs as a **JavaScript SDK in the
browser** over WebRTC. Streamlit hosts it via a **custom component** (`components.v1` / embedded
HTML+JS). The browser opens a WebRTC peer connection to the Azure avatar relay using a
**short-lived token** minted by a small backend endpoint (so the Speech key never reaches the
client). Answer text is passed to the avatar's `speak()` call; a `stop`/interrupt path handles
new questions.

**Auth/secrets.** Backend holds the Speech resource key; issues ephemeral tokens. Reuse the
project's existing Azure auth model (`DefaultAzureCredential` / keys via `.env`).

**Coexistence.** A mode toggle (voice-only vs avatar) decides whether C2 or C3 speaks. Answer
text + citations still render on screen in both modes (accessibility + verifiability).

## Infrastructure
- New: 1 **Azure AI Speech** resource (STT + TTS + Avatar are features of Speech). Confirm the
  real-time avatar feature is available in the target region (`eastus`) and its concurrency cap.
- Reused unchanged: Azure AI Search, Azure OpenAI `gpt-5-1`, Foundry agent.
- Small backend surface for the token endpoint (can live alongside the Streamlit app or as a tiny
  FastAPI sidecar).

## Model / service families
- Speech-to-Text: Azure AI Speech (managed). Text-to-Speech: Azure AI Speech neural voices.
- Avatar: Azure AI Speech **Text-to-Speech Avatar (real-time)**, prebuilt avatar.
- LLM/retrieval: unchanged (existing `gpt-5-1` + hybrid search / Foundry IQ).

## Key risk to de-risk first
Embedding a **real-time WebRTC avatar SDK inside Streamlit's component sandbox** (iframe/CSP,
token relay, video element lifecycle) is the novel part. **Recommend a 1–2 day spike** before
committing Feature 2 hours. Fallbacks if the sandbox fights it: (a) batch avatar video per answer
(simpler, higher per-answer latency), or (b) a minimal standalone React view for the avatar.
