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

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from src import speech
from src.config import get_settings
from src.pipelines import get_pipeline

app = FastAPI(title="SGS Voice Stream")
settings = get_settings()

# Local POC: the Streamlit component (another origin/port) calls this cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
):
    """Stream a spoken answer for question ``q`` as SSE events."""
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
                for sentence in acc.push(delta):
                    event = await speak(sentence)
                    if event:
                        yield event
            for sentence in acc.flush():
                event = await speak(sentence)
                if event:
                    yield event
            yield {"event": "done", "data": ""}
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "data": _friendly_error(exc)}

    return EventSourceResponse(event_gen())
