# Executive Summary — Voice I/O + Talking Avatar (SGS Chatbot)

## Overview
Two features added to the existing SGS RAG chatbot: (1) **voice input/output** — talk to the bot
and hear spoken answers; (2) a **lifelike talking avatar** that speaks the answers. Both are **I/O
layers around the existing RAG core, which is unchanged.** Scoped as a **POC/demo** on the current
Streamlit app + Azure stack: turn-based push-to-talk, English, low concurrency, Azure AI TTS Avatar.

## Recommended architecture
Record mic (Streamlit) → **Azure Speech STT** → existing RAG pipeline (`gpt-5-1`) → answer text →
either **Azure neural TTS** (voice-only) or the **Azure real-time TTS Avatar** (a JavaScript SDK
embedded in a Streamlit custom component over WebRTC, fed by a short-lived server-issued token).
All three speech features are **managed Azure services** — no model training. ~70% of the work is
ordinary software (UI, state, the WebRTC avatar embed, token plumbing).

## Effort summary

| Scenario | Hours | Duration |
|---|---|---|
| Lean (best case) | ~106 | ~2.5–3 wks (1 dev) |
| **Realistic (recommended)** | **~218** | **~3–3.5 wks (2 devs)** |

Split: Voice I/O ~88h · Avatar ~112h · Foundation ~18h (Realistic).

## Team composition
1 full-stack dev (Streamlit + JS/WebRTC avatar embed) · 1 backend/AI dev (Azure Speech, token
endpoint, pipeline glue) · ~0.2 FTE QA/review. ≈ 1.5–2 FTE over ~3–4 weeks.

## Operating cost (illustrative — verify)
Avatar minutes dominate (>90%). ~$65/mo (pilot) · ~$645/mo (expected) · ~$3,380/mo (high).
Biggest lever: **make the avatar opt-in, default to voice-only.** See `06-cost-projection.md`.

## Top 5 risks
1. **Avatar-in-Streamlit embed (WebRTC/CSP/sandbox)** — H. Novel; could double Feature 2. → spike first.
2. **Avatar cost + concurrency cap** — M/H. Per-minute billing + per-resource session limits; POC ok, production needs controls.
3. **Unverified pricing** — M. Verify avatar per-minute rate before quoting.
4. **Turn-based latency + existing `gpt-5-1` 10K-TPM quota** — M. Adds round-trip latency / 429 risk under load.
5. **Browser mic + cross-browser WebRTC** — M. Needs a test matrix.

## Top 5 assumptions
1. POC scope · English · 1–5 concurrent sessions.
2. Stays in Streamlit (avatar via custom component); if the sandbox proves too limiting, a minimal React view is the fallback (moderate impact).
3. Turn-based, not real-time streaming/barge-in.
4. Azure real-time TTS Avatar, **prebuilt** avatar (custom likeness excluded — cost/lead-time/consent).
5. Existing RAG pipeline + answers reused unchanged; voice/avatar are pure I/O.

## Recommended next steps
1. **1–2 day spike:** embed Azure real-time avatar in a Streamlit component (de-risk risk #1).
2. Verify Speech + **avatar per-minute pricing** and **regional avatar availability + concurrency cap** (`eastus`).
3. Build **Feature 1 (voice I/O) first** — lower risk, immediately demoable; layer the avatar on top.
4. Decide the default mode (voice-only vs avatar-on) for cost control.
5. (If it graduates past POC) plan the production delta: multilingual, real-time streaming, PII redaction, observability, concurrency scaling.

## Files produced
- 01-executive-summary.md · 03-scope-analysis.md · 04-ai-scope.md · 05-effort-estimation.md · 06-cost-projection.md
