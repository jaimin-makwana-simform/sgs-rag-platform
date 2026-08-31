# Cost Projection — Voice I/O + Talking Avatar (monthly operating cost)

> ⚠️ **Pricing below is ILLUSTRATIVE and MUST be verified** on the Azure Pricing Calculator for
> your region (`eastus`) before quoting a client. The **avatar per-minute rate dominates** total
> cost, so verify that number first. LLM/search costs are unchanged from the existing system and
> are excluded here.

## Assumed unit prices (verify)

| Service | Assumed price | Confidence |
|---|---|---|
| Azure Speech STT (standard) | ~$1.00 / audio-hour | Verify |
| Azure Speech neural TTS | ~$15 / 1M characters | Verify |
| **Azure real-time TTS Avatar** | **~$0.40 / avatar-minute** | **Verify — dominant driver** |

## Scenarios (per month)

Assumes each "session" ≈ a few Q&A turns; avatar minutes = time the avatar is actively speaking.

| Scenario | Sessions/mo | Avatar min | Avatar $ | STT $ | TTS $ | **Total/mo** |
|---|---|---|---|---|---|---|
| Low (pilot) | 50 (~3 min) | 150 | ~$60 | ~$3 | ~$2 | **~$65** |
| Expected | 500 (~3 min) | 1,500 | ~$600 | ~$25 | ~$20 | **~$645** |
| High | 2,000 (~4 min) | 8,000 | ~$3,200 | ~$100 | ~$80 | **~$3,380** |

**Avatar minutes are >90% of the bill in every scenario.** STT/TTS are rounding error by comparison.

## Cost levers
1. **Default to voice-only; make the avatar opt-in.** The single biggest lever — avatar minutes
   only accrue when a user turns the avatar on.
2. **Session timeouts + idle disconnect** so the avatar stream isn't billed while idle.
3. **Cap avatar speaking length** (summarize long answers for the spoken track; full text on screen).
4. Voice-only path (STT+TTS) is cheap enough to leave always-on.

## Other cost notes
- **Concurrency:** real-time avatar has a per-resource concurrent-session cap. Fine for a POC
  (1–5), but a production rollout may need a quota increase or a queue — verify the regional cap.
- **No material fixed cost:** Azure Speech has no large standing fee; cost is usage-driven.
- **Existing LLM constraint unchanged:** the `gpt-5-1` 10K-TPM quota still governs answer
  generation; voice/avatar don't add LLM tokens but do add latency pressure under load.

## Recommended verification step
Model Low/Expected/High in the **Azure Pricing Calculator** (Speech + Avatar line items) and
confirm the avatar per-minute rate + regional concurrency cap before committing to a client price.
