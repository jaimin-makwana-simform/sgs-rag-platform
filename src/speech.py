"""Azure AI Speech wrapper: STT (transcribe a recorded clip) + per-sentence TTS.

Reuses the shared AIServices resource (region + key) — leave ``SPEECH_API_KEY`` blank
and it falls back to ``AZURE_OPENAI_API_KEY`` (see ``Settings.effective_speech_key``).
The synthesizer is built once per (key, region, voice) and pre-connected to cut
first-byte latency, because the "speak-as-it-generates" flow synthesizes one short
sentence at a time.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor

import azure.cognitiveservices.speech as speechsdk

from .config import Settings

# Parallel synthesis fan-out — a few concurrent sentence syntheses shrink the
# post-generation "preparing audio" gap without stressing the Speech quota.
_TTS_MAX_WORKERS = 4

# Compact MP3 keeps the base64 payload small for the browser audio-queue component.
_MP3_FORMAT = speechsdk.SpeechSynthesisOutputFormat.Audio24Khz48KBitRateMonoMp3

# Flush a spoken chunk on sentence terminators or a newline.
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)|\n")

# Markdown → plain speech: the answer is Markdown, but TTS must not read out "**",
# bullet dashes, heading hashes, code backticks, or [source p.N] citation tags.
_MD_SUBS = [
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),  # [text](url) → keep the text
    (re.compile(r"\[[^\]]*\]"), r""),               # [source p.N] citations → drop
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),        # **bold**
    (re.compile(r"(?<!\*)\*(?!\*)([^*\n]+)\*"), r"\1"),  # *italic*
    (re.compile(r"__([^_]+)__"), r"\1"),
    (re.compile(r"`([^`]+)`"), r"\1"),              # `code`
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), r""),  # # headings
    (re.compile(r"^\s*[-*+]\s+", re.M), r""),       # - bullet markers
]


def clean_for_speech(text: str) -> str:
    """Strip Markdown so TTS speaks clean prose (no '**', dashes, citations)."""
    for pattern, repl in _MD_SUBS:
        text = pattern.sub(repl, text)
    text = re.sub(r"[*_`#]+", " ", text)  # any stray markers left over
    return re.sub(r"\s+", " ", text).strip()


def _key_region(settings: Settings) -> tuple[str, str]:
    key = settings.effective_speech_key
    if not key:
        raise RuntimeError(
            "No Speech key available. Set SPEECH_API_KEY (or AZURE_OPENAI_API_KEY) "
            "in .env — the app reuses the AIServices resource for Speech."
        )
    return key, settings.speech_region


# Per-thread synthesizer: a SpeechSynthesizer isn't safe to drive from multiple
# threads at once, so each worker thread reuses its own pre-connected instance.
_tls = threading.local()


def _thread_synthesizer(key: str, region: str, voice: str) -> speechsdk.SpeechSynthesizer:
    """Return this thread's pre-connected synthesizer (built once per thread)."""
    sig = (key, region, voice)
    if getattr(_tls, "sig", None) != sig or getattr(_tls, "synth", None) is None:
        cfg = speechsdk.SpeechConfig(subscription=key, region=region)
        cfg.speech_synthesis_voice_name = voice
        cfg.set_speech_synthesis_output_format(_MP3_FORMAT)
        synth = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=None)
        try:  # pre-establish the websocket so the first sentence isn't slowed by handshake
            speechsdk.Connection.from_speech_synthesizer(synth).open(True)
        except Exception:  # noqa: BLE001 - pre-connect is a latency optimization, not required
            pass
        _tls.synth = synth
        _tls.sig = sig
    return _tls.synth


def synthesize_sentence(settings: Settings, text: str) -> bytes:
    """Synthesize one sentence/phrase to MP3 bytes (no server-side playback).

    Markdown is stripped first so the audio never reads out formatting. Returns
    empty bytes if nothing speakable remains (e.g. a lone heading/bullet marker).
    """
    text = clean_for_speech(text)
    if not text:
        return b""
    key, region = _key_region(settings)
    synth = _thread_synthesizer(key, region, settings.speech_voice)
    result = synth.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return result.audio_data
    details = result.cancellation_details
    raise RuntimeError(
        f"TTS failed: {getattr(details, 'reason', None)} "
        f"{getattr(details, 'error_details', '')}"
    )


def synthesize_many(settings: Settings, sentences: list[str]) -> list[bytes]:
    """Synthesize sentences concurrently, preserving order.

    Fans out across a small thread pool (each thread has its own synthesizer) so the
    total wait is ~= slowest sentence rather than the sum of all sentences.
    """
    if not sentences:
        return []
    workers = min(_TTS_MAX_WORKERS, len(sentences))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        clips = list(pool.map(lambda s: synthesize_sentence(settings, s), sentences))
    return [c for c in clips if c]  # drop clips that were empty after cleaning


def transcribe(settings: Settings, audio_bytes: bytes) -> str:
    """Transcribe a recorded audio clip (WAV bytes, e.g. from ``st.audio_input``).

    Writes to a temp WAV file so the SDK reads the container header (sample rate,
    channels) rather than us guessing the raw format. Returns "" on no-match.
    """
    key, region = _key_region(settings)
    cfg = speechsdk.SpeechConfig(subscription=key, region=region)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    try:
        audio_cfg = speechsdk.audio.AudioConfig(filename=path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio_cfg)
        result = recognizer.recognize_once_async().get()
    finally:
        os.unlink(path)

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    if result.reason == speechsdk.ResultReason.NoMatch:
        return ""
    details = result.cancellation_details
    raise RuntimeError(
        f"STT failed: {getattr(details, 'reason', None)} "
        f"{getattr(details, 'error_details', '')}"
    )


class SentenceAccumulator:
    """Incrementally buffer streamed text deltas and release complete sentences.

    Same flush rule as ``sentence_chunks`` (sentence terminator/newline, or a
    ``max_chars`` cap), but stateful — the backend can emit each raw delta *and*
    synthesize sentences as they complete from the same token stream.
    """

    def __init__(self, max_chars: int = 200) -> None:
        self._buf = ""
        self._max = max_chars

    def push(self, delta: str) -> list[str]:
        """Add a delta; return any sentences it completed."""
        out: list[str] = []
        self._buf += delta
        while True:
            match = _SENTENCE_END.search(self._buf)
            if not match:
                break
            end = match.end()
            chunk = self._buf[:end].strip()
            self._buf = self._buf[end:]
            if chunk:
                out.append(chunk)
        if len(self._buf) >= self._max:
            chunk = self._buf.strip()
            self._buf = ""
            if chunk:
                out.append(chunk)
        return out

    def flush(self) -> list[str]:
        """Return any trailing remainder (call once the stream ends)."""
        chunk = self._buf.strip()
        self._buf = ""
        return [chunk] if chunk else []


def sentence_chunks(deltas: Iterable[str], max_chars: int = 200) -> Iterator[str]:
    """Group streamed text deltas into speakable sentences.

    Yields a chunk as soon as a sentence terminator (. ! ? or newline) is seen, or
    when the buffer grows past ``max_chars`` (so a long run without punctuation still
    starts speaking). Flushes any remainder at the end.
    """
    buf = ""
    for delta in deltas:
        buf += delta
        while True:
            match = _SENTENCE_END.search(buf)
            if not match:
                break
            end = match.end()
            chunk = buf[:end].strip()
            buf = buf[end:]
            if chunk:
                yield chunk
        if len(buf) >= max_chars:
            chunk = buf.strip()
            buf = ""
            if chunk:
                yield chunk
    if buf.strip():
        yield buf.strip()
