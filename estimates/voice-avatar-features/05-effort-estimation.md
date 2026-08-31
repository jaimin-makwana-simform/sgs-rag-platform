# Effort Estimation — Voice I/O + Talking Avatar (POC)

Hours. **Lean** = minimum viable happy path. **Realistic** = delivery-safe for a credible demo
(edge cases, hardening, review). Productivity tiers: STT/TTS = HIGH (mature APIs); avatar-in-
Streamlit = MEDIUM–LOW (novel embed). Scope = POC, English, low concurrency, Streamlit.

## Summary

| Area | Lean | Realistic | Owner |
|---|---|---|---|
| Feature 1 — Voice I/O (STT + TTS) | 44 | 88 | Backend/AI + FE |
| Feature 2 — Talking Avatar | 54 | 112 | Shared (FE-JS + BE) |
| Foundation / shared | 8 | 18 | Shared |
| **Grand total** | **106** | **218** | |

≈ **13 dev-days (Lean)** / **27 dev-days (Realistic)**.

## Feature 1 — Voice Input/Output (turn-based, Azure Speech STT+TTS)

| Sub-task | Complexity | Lean | Realistic |
|---|---|---|---|
| STT integration + mic capture (`st.audio_input` → Speech STT) | Small–Med | 10 | 18 |
| TTS (neural voice) + playback | Small | 6 | 12 |
| Voice UX / push-to-talk state, transcript, loading/errors | Medium | 10 | 20 |
| Wire to existing RAG pipeline (keep citations on screen) | Small | 6 | 12 |
| Cross-cutting: mic-permission/network errors, latency, session logging | Medium | 6 | 14 |
| Unit + integration tests + review buffer | Small–Med | 6 | 12 |
| **Subtotal** | | **44** | **88** |

Productivity: HIGH · Mandatory: Yes

## Feature 2 — Talking Avatar (Azure real-time TTS Avatar in Streamlit)

| Sub-task | Complexity | Lean | Realistic |
|---|---|---|---|
| Avatar JS SDK embed: Streamlit custom component, WebRTC, video lifecycle | Complex | 16 | 32 |
| Ephemeral token / relay backend endpoint (key stays server-side) | Small–Med | 6 | 12 |
| Drive avatar from answer text (speak/stop, mode toggle) | Medium | 8 | 16 |
| UX integration: avatar panel, sync with answer/citations, reconnect states | Medium | 8 | 18 |
| Cross-cutting: per-minute cost/session controls, concurrency guard, browser compat | Medium | 8 | 18 |
| Integration tests + demo hardening + review buffer | Medium | 8 | 16 |
| **Subtotal** | | **54** | **112** |

Productivity: MEDIUM–LOW (novel embed) · Mandatory: Yes

## Foundation / shared

| Sub-task | Lean | Realistic |
|---|---|---|
| Azure Speech resource + config + secrets/IaC + region/avatar-availability check | 4 | 8 |
| Voice + avatar coexistence (mode toggle) + docs | 4 | 10 |
| **Subtotal** | **8** | **18** |

## Sensitivity Analysis (where Realistic could double)
- **Avatar-in-Streamlit embed (C3).** If Streamlit's component sandbox fights the Azure avatar
  WebRTC SDK (iframe/CSP, token relay, media autoplay), the 32h embed can balloon to **60h+**.
  → **Resolve with a 1–2 day spike first.** Fallback: batch avatar video, or a minimal React view.
- **Region availability.** If real-time avatar isn't offered in `eastus`, add a resource in a
  supported region + cross-region wiring (+8–12h).

## Reconciliation (dual-axis)
- Bottom-up (Realistic): **218h**.
- Timeline-derived: 2 devs (1 full-stack/JS + 1 backend/AI) × ~3.5 weeks × ~30 productive h/wk
  ≈ **210h**. Divergence <5% — consistent.

## Duration & team
- **Lean:** ~106h → ~2.5–3 weeks (1 dev).
- **Realistic:** ~218h → ~3–3.5 weeks with **2 devs**, or ~5.5–6 weeks solo.
- Team: 1 full-stack (Streamlit + JS/WebRTC embed), 1 backend/AI (Speech, token endpoint, glue),
  ~0.2 FTE QA/review.

## Assumptions (Phase 5)
POC scope; English; 1–5 concurrent sessions; Streamlit retained; prebuilt avatar; existing RAG
reused unchanged; Azure avatar available in-region. Pricing for cost sizing per `06-cost-projection.md`.
