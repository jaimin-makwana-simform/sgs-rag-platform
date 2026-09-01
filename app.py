"""Streamlit UI for the SGS RAG platform.

Two interchangeable strategies behind one UI:
  - Default (Foundry IQ): a Foundry Agent + Knowledge Base, managed retrieval.
  - Custom RAG: the local hybrid pipeline with tunable chunk/overlap/top-k/threshold.

Both generate on the guardrailed gpt-5-1 model (Microsoft.DefaultV2), and both feed
a common evaluation pipeline that compares any Custom config against the cached
Default Foundry IQ baseline.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src import speech
from src.audio_player import render_audio_queue, render_voice_stream
from src.config import get_settings
from src.embeddings import build_client
from src.evaluation import (
    ALL_METRICS,
    EvalReport,
    compare,
    get_or_compute_baseline,
    load_cached_baseline,
    run_evaluation,
    save_report,
)
from src.pdf_loader import discover_pdfs
from src.pipelines import MODE_LABELS, get_pipeline
from src.rag import ingest_files
from src.search_index import document_count

ROOT = Path(__file__).parent
LABEL_TO_MODE = {v: k for k, v in MODE_LABELS.items()}

st.set_page_config(page_title="Document Query Assistant", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def _settings():
    return get_settings()


@st.cache_resource(show_spinner=False)
def _embed_client(_settings_obj):
    return build_client(_settings_obj)


def _save_uploaded_files(files, custom_dir: Path) -> list[Path]:
    custom_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for f in files:
        dest = custom_dir / f.name
        dest.write_bytes(f.getbuffer())
        saved.append(dest)
    return saved


def _reindex_all(settings) -> int:
    """Re-chunk, re-embed and re-upload all seed + custom PDFs. Returns chunk count.

    Used when chunk size/overlap change — those are ingest-time parameters, so the
    index must be rebuilt with new chunks + embeddings for the change to take effect.
    """
    seed_pdfs = discover_pdfs(
        [d for d in settings.docs_dirs_list if d != settings.custom_docs_dir], root=ROOT
    )
    custom_pdfs = discover_pdfs([settings.custom_docs_dir], root=ROOT)
    total = 0
    for group, source in ((seed_pdfs, "seed"), (custom_pdfs, "custom")):
        if group:
            total += ingest_files(
                settings, group, doc_source=source, client=_embed_client(settings)
            ).chunks_uploaded
    return total


def _sidebar(settings) -> tuple[str, dict]:
    """Render the sidebar; return (mode, custom_overrides)."""
    with st.sidebar:
        st.header("⚙️ Mode")
        label = st.radio(
            "Retrieval strategy",
            options=list(MODE_LABELS.values()),
            help=(
                "Default (Foundry IQ): Microsoft-managed retrieval via a Foundry "
                "Agent + Knowledge Base.\n\nCustom RAG: local hybrid search you tune."
            ),
        )
        mode = LABEL_TO_MODE[label]

        overrides: dict = {}
        if mode == "custom":
            # Applied params (not the raw sliders) drive answering + evaluation. The
            # sliders live inside a form, so dragging them does NOT rerun the app or
            # hit the backend — nothing changes until "Submit" is pressed.
            if "applied_params" not in st.session_state:
                st.session_state.applied_params = {
                    "top_k": settings.top_k,
                    "reranker_threshold": float(settings.reranker_threshold),
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                }
            if "indexed_chunking" not in st.session_state:
                # The chunk config the current index was actually built with.
                st.session_state.indexed_chunking = {
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                }
            applied = st.session_state.applied_params

            with st.form("custom_params"):
                st.subheader("Custom parameters")
                top_k = st.slider("Top-K retrieved chunks", 1, 15, applied["top_k"])
                threshold = st.slider(
                    "Reranker threshold (0-4)", 0.0, 4.0,
                    float(applied["reranker_threshold"]), 0.1,
                    help="Min semantic reranker score for a chunk to be used / answered.",
                )
                chunk_size = st.slider(
                    "Chunk size (tokens)", 128, 1024, applied["chunk_size"], 32
                )
                chunk_overlap = st.slider(
                    "Chunk overlap (tokens)", 0, 256, applied["chunk_overlap"], 16
                )
                st.caption(
                    "Nothing is applied until you press Submit. Top-K / threshold "
                    "apply with no re-index; changing chunk size / overlap triggers a "
                    "re-index (re-chunk + re-embed) on Submit."
                )
                submitted = st.form_submit_button("Submit", type="primary")

            if submitted:
                need_reindex = (
                    chunk_size != st.session_state.indexed_chunking["chunk_size"]
                    or chunk_overlap != st.session_state.indexed_chunking["chunk_overlap"]
                )
                st.session_state.applied_params = {
                    "top_k": top_k,
                    "reranker_threshold": threshold,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                }
                if need_reindex:
                    settings.chunk_size = chunk_size
                    settings.chunk_overlap = chunk_overlap
                    with st.spinner(
                        "Chunking changed — re-indexing (re-chunk + re-embed)..."
                    ):
                        total = _reindex_all(settings)
                    st.session_state.indexed_chunking = {
                        "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                    }
                    st.success(
                        f"Applied. Re-indexed {total} chunk(s) at chunk_size="
                        f"{chunk_size}, overlap={chunk_overlap}. "
                        f"Top-K={top_k}, threshold={threshold}."
                    )
                else:
                    st.success(
                        f"Applied Top-K={top_k}, threshold={threshold} "
                        "(no re-index needed — embeddings unchanged)."
                    )
            overrides = st.session_state.applied_params
        else:
            st.caption(
                f"Agent: `{settings.foundry_agent_name}`  \n"
                f"Knowledge base: `{settings.foundry_knowledge_base_name}`  \n"
                "Retrieval config is managed by Foundry IQ (no knobs)."
            )

        st.divider()
        st.header("📁 Documents")
        try:
            st.metric("Chunks in Custom index", document_count(settings))
        except Exception:  # noqa: BLE001
            st.info("Custom index not created yet. Ingest documents to get started.")

        uploaded = st.file_uploader(
            "Add PDFs (stored locally under custom_docs/)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded and st.button("Save & index uploads", type="primary"):
            saved = _save_uploaded_files(uploaded, ROOT / settings.custom_docs_dir)
            with st.spinner(f"Indexing {len(saved)} file(s)..."):
                result = ingest_files(
                    settings, saved, doc_source="custom", client=_embed_client(settings)
                )
            st.success(
                f"Indexed {result.files_processed} file(s), "
                f"{result.chunks_uploaded} chunk(s)."
            )

        st.divider()
        st.caption("🛡️ Guardrails: **Microsoft.DefaultV2** (enforced on gpt-5-1 — "
                   "applies to both modes and cannot be tuned away).")

    return mode, overrides


def _render_sources(sources) -> None:
    if not sources:
        return
    st.markdown("### Sources")
    for i, src in enumerate(sources, 1):
        label = (f"{i}. {src.source_file}"
                 + (f" — p.{src.page}" if src.page else "")
                 + (f" (reranker {src.reranker_score:.2f})" if src.reranker_score else ""))
        with st.expander(label):
            st.write(src.content)


def _render_avatar(url: str, *, height: int = 620) -> None:
    """Embed the backend avatar page in an iframe granted autoplay permission.

    Uses ``components.html`` (rather than ``components.iframe``) so we can set the
    ``allow`` attribute — delegating autoplay to the cross-origin avatar page so its
    WebRTC video/audio start without a manual click. The ``url`` is kept **constant**
    (no per-question query string) so Streamlit reruns don't reload the iframe and the
    avatar's WebRTC session persists — questions are delivered out-of-band (see below).
    """
    safe = url.replace('"', "&quot;")
    components.html(
        f'<iframe src="{safe}" allow="autoplay; camera; microphone; fullscreen" '
        f'referrerpolicy="no-referrer" '
        f'style="width:100%;height:{height - 8}px;border:0;border-radius:12px"></iframe>',
        height=height,
    )


def _post_avatar_question(settings, question: str, overrides: dict) -> None:
    """Publish a question to the connected avatar via the backend channel.

    Guarded by session state so it fires once per new question (not on every rerun),
    which keeps the persistent avatar from re-speaking the same answer.
    """
    payload = {
        "q": question,
        "top_k": overrides.get("top_k"),
        "reranker_threshold": overrides.get("reranker_threshold"),
    }
    if st.session_state.get("_avatar_last_ask") == payload:
        return
    try:
        httpx.post(
            f"{settings.voice_backend_url.rstrip('/')}/avatar/ask",
            json=payload,
            timeout=5,
        ).raise_for_status()
        st.session_state["_avatar_last_ask"] = payload
    except Exception as exc:  # noqa: BLE001 - non-fatal; avatar just won't speak this turn
        st.caption(f"(couldn't send the question to the avatar: {type(exc).__name__})")


def _get_question(settings, input_mode: str) -> tuple[str | None, bool]:
    """Return (question, ready_to_answer) for the selected input mode."""
    if input_mode == "🎙️ Speak":
        clip = st.audio_input("Record your question")
        if clip is None:
            return None, False
        try:
            with st.spinner("Transcribing…"):
                question = speech.transcribe(settings, clip.getvalue())
        except Exception as exc:  # noqa: BLE001
            st.error(f"Transcription failed: {type(exc).__name__}: {exc}")
            return None, False
        if not question.strip():
            st.warning("Didn't catch that — please record again.")
            return None, False
        st.info(f"📝 Transcript: {question}")
        return question, True

    question = st.text_input(
        "Your question",
        placeholder="e.g. What does the SGS anti-corruption policy prohibit?",
    )
    return question, bool(st.button("Ask", type="primary") and question.strip())


def _ask_tab(settings, mode: str, overrides: dict) -> None:
    st.subheader(f"Ask — {MODE_LABELS[mode]}")

    c1, c2 = st.columns([2, 1])
    with c1:
        input_mode = st.radio("Input", ["⌨️ Type", "🎙️ Speak"], horizontal=True)
    with c2:
        # "Streaming (live)" and "🧑 Avatar" need the FastAPI backend + Custom pipeline;
        # the other options work in any mode. "Text only" produces no audio. The avatar
        # is opt-in (billed per minute) and only offered when AVATAR_ENABLED is set.
        voice_options = (
            ["Streaming (live)", "After generation", "Text only"]
            if mode == "custom"
            else ["After generation", "Text only"]
        )
        if mode == "custom" and settings.avatar_enabled:
            voice_options.insert(1, "🧑 Avatar")
        voice_mode = st.selectbox("Voice output", voice_options)

    # --- Agent avatar: the FastAPI-served avatar page (own origin) speaks the answer
    #     via Azure real-time TTS Avatar over WebRTC. Reuses the same STT input + RAG.
    #     The iframe renders in a FIXED slot with a CONSTANT url and auto-connects once;
    #     each question is pushed over the backend channel (POST /avatar/ask), so the
    #     WebRTC session persists across reruns — no reconnect between questions. ---
    if voice_mode == "🧑 Avatar":
        # Reserve the INPUT area first (question box / recorder stay on top), then place
        # the avatar in a slot below it. Because both are fixed containers, the transcript
        # (st.info) grows *inside* the input slot without shifting the avatar's position —
        # so the iframe isn't remounted on rerun and the WebRTC session persists.
        input_slot = st.container()
        st.markdown("### Agent avatar")
        avatar_slot = st.container()
        with avatar_slot:
            _render_avatar(f"{settings.voice_backend_url.rstrip('/')}/avatar")
            st.caption(
                "The avatar connects automatically and speaks each answer. "
                "Needs the voice backend running (use `run.sh`)."
            )

        with input_slot:
            question, ready = _get_question(settings, input_mode)
        if ready and question and question.strip():
            q = question.strip()
            _post_avatar_question(settings, q, overrides)
            try:
                sources = get_pipeline(settings, "custom", **overrides).retrieve(q)
                _render_sources(sources)
            except Exception as exc:  # noqa: BLE001 - citations are best-effort here
                st.caption(f"(couldn't load citations: {type(exc).__name__})")
        return

    question, ready = _get_question(settings, input_mode)
    if not ready or not question:
        return
    question = question.strip()

    # --- Concurrent streaming voice (c1): backend drives text + audio in the component;
    #     citations are rendered natively (always visible, not clipped by the iframe). ---
    if mode == "custom" and voice_mode == "Streaming (live)":
        st.markdown("### Answer")
        render_voice_stream(question, settings.voice_backend_url, overrides=overrides)
        st.caption(
            "Streaming from the voice backend — audio plays as the answer generates. "
            "If it says the backend isn't running, start it (or use `run.sh`)."
        )
        try:
            sources = get_pipeline(settings, "custom", **overrides).retrieve(question)
            _render_sources(sources)
        except Exception as exc:  # noqa: BLE001 - citations are best-effort here
            st.caption(f"(couldn't load citations: {type(exc).__name__})")
        return

    # --- Non-streaming path: stream text; optional spoken answer after generation (b) ---
    try:
        pipeline = get_pipeline(settings, mode, **overrides)
        sources, deltas = pipeline.answer_stream(question)
        st.markdown("### Answer")
        full_text = st.write_stream(deltas)
    except Exception as exc:  # noqa: BLE001
        msg = f"Error: {type(exc).__name__}: {exc}"
        if "429" in str(exc) or "rate_limit" in str(exc).lower():
            msg += ("  \n\n_The gpt-5-1 deployment hit its rate limit — wait a moment "
                    "and retry, or raise its capacity._")
        st.error(msg)
        return

    # Voice output: synthesize each sentence and play them gaplessly ("Text only" skips this).
    if voice_mode == "After generation" and isinstance(full_text, str) and full_text.strip():
        try:
            with st.spinner("Preparing audio…"):
                sentences = list(speech.sentence_chunks([full_text]))
                clips = speech.synthesize_many(settings, sentences)  # parallel
            render_audio_queue(clips)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Voice output unavailable: {type(exc).__name__}: {exc}")

    _render_sources(sources)


def _metrics_table(reports: dict[str, EvalReport]) -> pd.DataFrame:
    data = {label: {m: rep.metrics.get(m) for m in ALL_METRICS}
            for label, rep in reports.items()}
    return pd.DataFrame(data)


def _evaluate_tab(settings) -> None:
    st.subheader("Evaluate & compare vs the Foundry IQ baseline")
    uploaded = st.file_uploader(
        "Upload ground-truth dataset (JSONL)",
        type=["jsonl"],
        help=(
            "One JSON object per line, e.g.: "
            '{"question": "...", "ground_truth_answer": "...", '
            '"relevant_docs": ["file.pdf"], "answerable": true}'
        ),
    )
    if uploaded is None:
        st.info("Upload a ground-truth `.jsonl` dataset to run an evaluation.")
        return

    # Persist the upload so the (path-based) eval harness can read it. The baseline
    # cache is keyed by a fingerprint of the file *contents*, so re-uploading the
    # same dataset reuses the cached baseline; a different dataset recomputes it.
    dataset_path = Path(settings.eval_results_dir) / "uploaded_dataset.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_bytes(uploaded.getvalue())
    n_rows = sum(1 for line in uploaded.getvalue().decode("utf-8").splitlines() if line.strip())
    st.caption(f"Loaded **{uploaded.name}** — {n_rows} question(s).")

    # All eval inputs live in a FORM: changing metrics / max questions / recompute does
    # NOT rerun the page or hit the backend — nothing runs until Submit. On submit we
    # compute the Foundry IQ baseline (cached) and the Custom RAG run, then compare.
    with st.form("eval_params"):
        col1, col2 = st.columns(2)
        with col1:
            metrics = st.multiselect("Metrics", list(ALL_METRICS), default=list(ALL_METRICS))
        with col2:
            limit = st.number_input(
                "Max questions (0 = all)", min_value=0, value=5,
                help="Cap questions to stay within gpt-5.1 quota during demos.",
            )
        recompute = st.checkbox(
            "Recompute Foundry IQ baseline",
            value=False,
            help="Force-refresh the cached baseline on this Submit (otherwise it's reused).",
        )
        submitted = st.form_submit_button("Submit — run evaluation", type="primary")

    lim = None if limit == 0 else int(limit)
    overrides = st.session_state.get(
        "applied_params",
        {
            "top_k": settings.top_k,
            "reranker_threshold": settings.reranker_threshold,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
    )

    # Backend work happens ONLY on Submit — Custom uses the sidebar's applied params.
    if submitted:
        if not metrics:
            st.warning("Select at least one metric before submitting.")
        else:
            try:
                cached = None if recompute else load_cached_baseline(settings, dataset_path)
                if cached is not None:
                    baseline = cached  # reuse — no backend call
                else:
                    bar_b = st.progress(0.0, text="Running Foundry IQ baseline...")
                    baseline = get_or_compute_baseline(
                        settings, dataset_path, force=recompute, limit=lim,
                        progress=lambda d, t: bar_b.progress(d / t, text=f"Baseline {d}/{t}"),
                    )
                    bar_b.empty()
                bar = st.progress(0.0, text="Running Custom evaluation...")
                pipeline = get_pipeline(settings, "custom", **overrides)
                custom = run_evaluation(
                    pipeline, dataset_path, settings,
                    metrics=tuple(metrics), limit=lim,
                    progress=lambda d, t: bar.progress(d / t, text=f"Custom {d}/{t}"),
                )
                bar.empty()
                save_report(settings, custom, f"custom_{int(custom.timestamp)}.json")
                st.session_state["custom"] = custom.to_json()
                st.success("Evaluation complete.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Evaluation failed: {type(exc).__name__}: {exc}")

    # --- Results: render whatever has been computed (persists across reruns) ---
    baseline = load_cached_baseline(settings, dataset_path)
    if baseline is None:
        st.info(
            "Configure metrics / max questions above, then click **Submit** to run the "
            "evaluation. Changing those inputs won't call the backend until you submit."
        )
        return

    st.markdown("#### Default Foundry IQ baseline")
    st.caption(f"Baseline config: `{baseline.config}`")
    st.dataframe(_metrics_table({"Foundry IQ (baseline)": baseline}))

    if "custom" in st.session_state:
        custom = EvalReport.from_json(st.session_state["custom"])
        if custom.dataset_fingerprint == baseline.dataset_fingerprint:
            cmp = compare(baseline, custom)
            st.markdown("#### Comparison — Custom RAG vs Foundry IQ")
            st.info(f"**Verdict: {cmp.verdict}**")
            table = _metrics_table({"Foundry IQ (baseline)": baseline, "Custom": custom})
            table["Δ (custom−base)"] = pd.Series(cmp.deltas)
            table["winner"] = pd.Series(cmp.winners)
            st.dataframe(table)
            st.caption(f"Custom config: `{custom.config}`")


def main() -> None:
    try:
        settings = _settings()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Configuration error — check your .env file.\n\n{type(exc).__name__}: {exc}")
        st.stop()

    st.title("📄 Document Query Assistant")
    st.caption(
        "Ask questions about SGS policies grounded in the indexed documents, "
        "and compare a tunable Custom RAG config against the Default Foundry IQ baseline."
    )

    mode, overrides = _sidebar(settings)
    ask_tab, eval_tab = st.tabs(["💬 Ask", "📊 Evaluate"])
    with ask_tab:
        _ask_tab(settings, mode, overrides)
    with eval_tab:
        _evaluate_tab(settings)


if __name__ == "__main__":
    main()
