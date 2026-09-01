"""FastAPI SSE backend for concurrent streaming voice.

Single source of the answer generation: ``/voice/stream`` runs the Custom RAG
pipeline once and streams **sources → text deltas → per-sentence audio → done** over
one Server-Sent Events connection, so the browser can speak sentence 1 while the
model is still generating sentence 3. Keys stay server-side (the browser only ever
receives text + audio, never secrets).

Run:  uvicorn server:app --host localhost --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src import speech
from src.config import get_settings
from src.pipelines import get_pipeline

app = FastAPI(title="Document Query Assistant Voice Backend")
settings = get_settings()

_STATIC_DIR = Path(__file__).parent / "static"

# Agent-avatar question channel (single-user POC pub/sub). The persistent avatar page
# subscribes to `/avatar/pending`; the Streamlit app posts each new question to
# `/avatar/ask`. This decouples the question from the iframe URL so the avatar's
# WebRTC session survives Streamlit reruns (no reconnect between questions).
_avatar_state: dict = {"id": 0, "q": None, "top_k": None, "reranker_threshold": None}
_avatar_subscribers: set[asyncio.Queue] = set()

# Local POC: the Streamlit component (another origin/port) calls this cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static assets for the avatar client (served at its own origin so WebRTC/autoplay
# behave like a normal page inside the Streamlit iframe).
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

_SENTINEL = object()


def _source_dict(src) -> dict:
    return {
        "source_file": src.source_file,
        "page": src.page,
        "score": round(src.reranker_score, 2) if src.reranker_score else 0.0,
        "content": src.content,
    }


def _friendly_error(exc: Exception) -> str:
    text = str(exc)
    if "429" in text or "rate_limit" in text.lower():
        return ("The gpt-5-1 deployment hit its rate limit — wait a moment and retry, "
                "or raise its capacity.")
    return f"{type(exc).__name__}: {text}"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/voice/stream")
async def voice_stream(
    request: Request,
    q: str,
    top_k: int | None = None,
    reranker_threshold: float | None = None,
    audio: bool = True,
):
    """Stream a spoken answer for question ``q`` as SSE events.

    ``audio=true`` (default) emits per-sentence ``audio`` events (browser TTS
    playback). ``audio=false`` streams **text only** (``sources``/``delta``/``done``)
    — used by the avatar client, which synthesizes its own speech from the text and
    would otherwise make the backend do redundant TTS work.
    """
    overrides: dict = {}
    if top_k is not None:
        overrides["top_k"] = top_k
    if reranker_threshold is not None:
        overrides["reranker_threshold"] = reranker_threshold

    async def event_gen():
        loop = asyncio.get_event_loop()

        # Retrieval + generation kickoff (blocking) off the event loop.
        try:
            pipeline = get_pipeline(settings, "custom", **overrides)
            sources, deltas = await loop.run_in_executor(
                None, pipeline.answer_stream, q
            )
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "data": _friendly_error(exc)}
            return

        yield {"event": "sources",
               "data": json.dumps([_source_dict(s) for s in sources])}

        acc = speech.SentenceAccumulator()
        deltas_it = iter(deltas)

        async def speak(sentence: str):
            clip = await loop.run_in_executor(
                None, speech.synthesize_sentence, settings, sentence
            )
            if not clip:  # nothing speakable after markdown cleaning
                return None
            return {"event": "audio", "data": base64.b64encode(clip).decode("ascii")}

        try:
            while True:
                if await request.is_disconnected():
                    return
                delta = await loop.run_in_executor(None, next, deltas_it, _SENTINEL)
                if delta is _SENTINEL:
                    break
                yield {"event": "delta", "data": delta}
                if audio:
                    for sentence in acc.push(delta):
                        event = await speak(sentence)
                        if event:
                            yield event
            if audio:
                for sentence in acc.flush():
                    event = await speak(sentence)
                    if event:
                        yield event
            yield {"event": "done", "data": ""}
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "data": _friendly_error(exc)}

    return EventSourceResponse(event_gen())


# --------------------------------------------------------------------------- #
# Agent avatar (Azure real-time TTS Avatar)
#
# The avatar client is a real page served from this backend's origin (so WebRTC +
# media autoplay behave like a normal page inside the Streamlit iframe). It fetches
# an ICE relay token + a short-lived Speech authorization token from `/avatar/token`
# — the raw Speech key never reaches the browser.
# --------------------------------------------------------------------------- #


@app.get("/avatar")
def avatar_page() -> FileResponse:
    """Serve the avatar client page (its URL stays constant across questions)."""
    return FileResponse(_STATIC_DIR / "avatar" / "index.html")


class AvatarAsk(BaseModel):
    q: str
    top_k: int | None = None
    reranker_threshold: float | None = None


@app.post("/avatar/ask")
async def avatar_ask(body: AvatarAsk) -> dict:
    """Publish a new question to the connected avatar page(s)."""
    _avatar_state["id"] += 1
    _avatar_state.update(
        q=body.q, top_k=body.top_k, reranker_threshold=body.reranker_threshold
    )
    snapshot = dict(_avatar_state)
    for queue in list(_avatar_subscribers):
        queue.put_nowait(snapshot)
    return {"id": snapshot["id"]}


@app.get("/avatar/pending")
async def avatar_pending(request: Request):
    """SSE stream of questions for the avatar to speak (opened once per session)."""

    async def gen():
        queue: asyncio.Queue = asyncio.Queue()
        _avatar_subscribers.add(queue)
        try:
            # A late-joining subscriber still gets the current pending question.
            if _avatar_state["q"] is not None:
                queue.put_nowait(dict(_avatar_state))
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    yield {"event": "ping", "data": ""}  # keep the connection warm
                    continue
                yield {"event": "question", "data": json.dumps(item)}
        finally:
            _avatar_subscribers.discard(queue)

    return EventSourceResponse(gen())


@app.get("/avatar/token")
async def avatar_token() -> dict:
    """Mint the credentials the browser needs for a real-time avatar session.

    Returns WebRTC ICE relay servers plus a short-lived Speech authorization token
    (valid ~10 min) so the client can build ``SpeechConfig.fromAuthorizationToken``
    without ever seeing the subscription key.
    """
    region = settings.speech_region
    key = settings.effective_speech_key
    if not key:
        raise HTTPException(
            status_code=500,
            detail="No Speech key available (set SPEECH_API_KEY or AZURE_OPENAI_API_KEY).",
        )

    relay_url = (
        f"https://{region}.tts.speech.microsoft.com"
        "/cognitiveservices/avatar/relay/token/v1"
    )
    token_url = f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    headers = {"Ocp-Apim-Subscription-Key": key}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            relay_resp, token_resp = await asyncio.gather(
                client.get(relay_url, headers=headers),
                client.post(token_url, headers=headers),
            )
        relay_resp.raise_for_status()
        token_resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Failed to get avatar relay/auth token from Azure Speech "
                f"(region '{region}'): {type(exc).__name__}: {exc}. "
                "Check the key and that the region supports real-time avatar."
            ),
        ) from exc

    relay = relay_resp.json()
    return {
        "iceServers": [
            {
                "urls": relay["Urls"],
                "username": relay["Username"],
                "credential": relay["Password"],
            }
        ],
        "speechToken": token_resp.text,
        "region": region,
        "character": settings.avatar_character,
        "style": settings.avatar_style,
        "voice": settings.speech_voice,
    }
