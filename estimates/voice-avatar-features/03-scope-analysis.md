# AI Component Analysis — Voice I/O + Talking Avatar (SGS Chatbot)

Add-on to an existing **AI-Enhanced** system (Streamlit + Azure AI Search + Azure OpenAI
`gpt-5-1` + Azure AI Foundry agent). The two new features are **I/O layers** around the
*existing* RAG pipeline — the retrieval/answer core does not change.

**Locked scoping decisions (from discovery):** POC/demo first · stay in Streamlit ·
turn-based push-to-talk (not real-time streaming) · Azure AI TTS Avatar (real-time).

## Classification Summary

| Feature element | Class | AI? | Build category |
|---|---|---|---|
| Speech-to-Text (transcribe user) | AI/ML (managed) | Yes | Integrate |
| Text-to-Speech (speak answer) | AI/ML (managed) | Yes | Integrate |
| Talking avatar (real-time video) | AI/ML (managed) | Yes | Integrate / Wire |
| Mic capture + audio playback UI | Frontend | No | Software |
| Push-to-talk state machine | Frontend/Backend | No | Software |
| Avatar embed (JS SDK, WebRTC) in Streamlit | Frontend | No | Software (JS) |
| Ephemeral token / relay endpoint | Backend | No | Software |
| Wiring to existing RAG answer path | Backend | No | Software |

**Takeaway:** all three AI pieces are **managed Azure services (Integrate/Wire)** — no model
training, no fine-tuning. ~70% of the effort is ordinary software (UI, state, WebRTC embed,
token plumbing), which is exactly where the risk and hours concentrate.

## AI Components

### C1 — Speech-to-Text (Azure AI Speech STT)
- Build Category: Integrate · Difficulty: Read · Tier: Managed Service
- Inputs: recorded mic audio (WAV/webm, short utterances). Outputs: transcript text.
- Data deps: none (no training). Integration: transcript → existing pipeline input.
- Owner: Backend/AI.

### C2 — Text-to-Speech (Azure AI Speech neural voice)
- Build Category: Integrate · Difficulty: Read · Tier: Managed Service
- Inputs: answer text. Outputs: neural-voice audio.
- Note: **subsumed by the avatar when avatar mode is on** (the avatar speaks the text). Kept
  as a separate path for voice-only mode.
- Owner: Backend/AI.

### C3 — Real-time Talking Avatar (Azure AI TTS Avatar)
- Build Category: Integrate / Wire · Difficulty: Derive (integration) · Tier: Managed Service
- Inputs: answer text + ephemeral session token. Outputs: live lip-synced avatar video (WebRTC).
- Integration: Azure avatar **JavaScript SDK embedded in a Streamlit custom component**;
  backend issues a short-lived relay token; answer text is pushed to the avatar to speak.
- Owner: Shared (frontend JS + backend token endpoint). **Highest effort + risk component.**

## Non-AI Components (standard software estimation)
Mic capture/playback UI · push-to-talk flow + session state · avatar panel + reconnect UX ·
token/relay backend endpoint · glue to existing Custom/Foundry answer path · voice-vs-avatar
mode toggle · docs/IaC.

## Hidden Work Items (POC treatment)
- [x] Cost controls / session budgeting — **in** (avatar billed per minute; add session timeout + opt-in).
- [x] Latency handling (STT→RAG→TTS/avatar round-trip) — **in** (loading states, streaming where cheap).
- [x] Error handling — **in** (mic permission, network drop, avatar reconnect, cross-browser).
- [x] Secure token handling — **in** (no Speech key in the browser; ephemeral tokens only).
- [~] Observability — **light** (basic session logging; full tracing deferred to production).
- [ ] PII/redaction of voice transcripts — **excluded for POC** (flag for production).
- [ ] Multilingual STT/TTS — **excluded for POC** (English only; add locales later).
- [ ] Custom avatar likeness (a specific person's face/voice) — **excluded** (prebuilt Azure
  avatar only; custom likeness adds significant cost, lead time, and consent/approval).

## Architecture Decisions (Locked)
1. Reuse the existing RAG pipeline unchanged; voice + avatar are pure I/O layers.
2. Turn-based flow: record → STT → RAG answer → speak (TTS or avatar). No real-time barge-in.
3. Avatar = Azure real-time TTS Avatar, embedded via a Streamlit custom (HTML/JS) component.
4. Avatar mode handles its own speech; plain TTS only used in voice-only mode.
5. Prebuilt avatar; English; low concurrency (1–5 sessions).

## Assumptions (Phase 3)
See `01-executive-summary.md` → Top Assumptions.
