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

import pandas as pd
import streamlit as st

from src.config import get_settings
from src.embeddings import build_client
from src.evaluation import (
    ALL_METRICS,
    BASELINE_FILENAME,
    EvalReport,
    compare,
    get_or_compute_baseline,
    run_evaluation,
    save_report,
)
from src.pdf_loader import discover_pdfs
from src.pipelines import MODE_LABELS, get_pipeline
from src.rag import ingest_files
from src.search_index import document_count

ROOT = Path(__file__).parent
LABEL_TO_MODE = {v: k for k, v in MODE_LABELS.items()}

st.set_page_config(page_title="SGS Document Assistant", page_icon="📄", layout="wide")


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
            st.subheader("Custom parameters")
            overrides["top_k"] = st.slider(
                "Top-K retrieved chunks", 1, 15, settings.top_k
            )
            overrides["reranker_threshold"] = st.slider(
                "Reranker threshold (0-4)",
                0.0,
                4.0,
                float(settings.reranker_threshold),
                0.1,
                help="Minimum semantic reranker score for a chunk to be used / answered.",
            )
            overrides["chunk_size"] = st.slider(
                "Chunk size (tokens)", 128, 1024, settings.chunk_size, 32
            )
            overrides["chunk_overlap"] = st.slider(
                "Chunk overlap (tokens)", 0, 256, settings.chunk_overlap, 16
            )
            st.caption(
                "Top-K and threshold apply live. Chunk size/overlap only take effect "
                "after re-indexing with these settings (button below)."
            )
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

        if mode == "custom" and st.button("Re-index with these settings"):
            # Apply the chunking overrides to the shared settings, then re-ingest.
            settings.chunk_size = overrides["chunk_size"]
            settings.chunk_overlap = overrides["chunk_overlap"]
            seed_pdfs = discover_pdfs(
                [d for d in settings.docs_dirs_list if d != settings.custom_docs_dir],
                root=ROOT,
            )
            custom_pdfs = discover_pdfs([settings.custom_docs_dir], root=ROOT)
            with st.spinner("Re-indexing all documents with new chunk settings..."):
                total = 0
                for group, source in ((seed_pdfs, "seed"), (custom_pdfs, "custom")):
                    if group:
                        total += ingest_files(
                            settings, group, doc_source=source,
                            client=_embed_client(settings),
                        ).chunks_uploaded
            st.success(f"Re-indexed {total} chunk(s) with chunk_size="
                       f"{overrides['chunk_size']}, overlap={overrides['chunk_overlap']}.")

        st.divider()
        st.caption("🛡️ Guardrails: **Microsoft.DefaultV2** (enforced on gpt-5-1 — "
                   "applies to both modes and cannot be tuned away).")

    return mode, overrides


def _ask_tab(settings, mode: str, overrides: dict) -> None:
    st.subheader(f"Ask — {MODE_LABELS[mode]}")
    question = st.text_input(
        "Your question",
        placeholder="e.g. What does the SGS anti-corruption policy prohibit?",
    )
    if st.button("Ask", type="primary") and question.strip():
        result = None
        error: Exception | None = None
        with st.spinner("Retrieving and generating an answer..."):
            try:
                pipeline = get_pipeline(settings, mode, **overrides)
                result = pipeline.answer(question.strip())
            except Exception as exc:  # noqa: BLE001
                error = exc

        if error is not None:
            st.error(f"Error: {type(error).__name__}: {error}")
            return

        st.markdown("### Answer")
        st.write(result.text)
        if result.sources:
            st.markdown("### Sources")
            for i, src in enumerate(result.sources, 1):
                label = (f"{i}. {src.source_file}"
                         + (f" — p.{src.page}" if src.page else "")
                         + (f" (reranker {src.reranker_score:.2f})"
                            if src.reranker_score else ""))
                with st.expander(label):
                    st.write(src.content)


def _metrics_table(reports: dict[str, EvalReport]) -> pd.DataFrame:
    data = {label: {m: rep.metrics.get(m) for m in ALL_METRICS}
            for label, rep in reports.items()}
    return pd.DataFrame(data)


def _evaluate_tab(settings) -> None:
    st.subheader("Evaluate & compare vs the Foundry IQ baseline")
    dataset_path = st.text_input("Ground-truth dataset (JSONL)", settings.eval_dataset_path)
    if not Path(dataset_path).exists():
        st.warning(
            f"`{dataset_path}` not found. Generate it with "
            "`python -m eval.generate_ground_truth`, or point to "
            "`eval/seed_questions.jsonl`."
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        metrics = st.multiselect("Metrics", list(ALL_METRICS), default=list(ALL_METRICS))
    with col2:
        limit = st.number_input(
            "Max questions (0 = all)", min_value=0, value=5,
            help="Cap questions to stay within gpt-5.1 quota during demos.",
        )
    lim = None if limit == 0 else int(limit)

    # --- Baseline (Default Foundry IQ), cached ---
    st.markdown("#### 1. Default Foundry IQ baseline")
    baseline_path = Path(settings.eval_results_dir) / BASELINE_FILENAME
    cols = st.columns([1, 1])
    force = cols[1].checkbox("Recompute baseline", value=False)
    if cols[0].button("Compute / load baseline"):
        bar = st.progress(0.0, text="Running Foundry IQ baseline...")
        try:
            baseline = get_or_compute_baseline(
                settings, dataset_path, force=force, limit=lim,
                progress=lambda d, t: bar.progress(d / t, text=f"Baseline {d}/{t}"),
            )
            st.session_state["baseline"] = baseline.to_json()
            st.success(f"Baseline ready ({baseline.n_questions} questions).")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Baseline failed: {type(exc).__name__}: {exc}")
    elif baseline_path.exists() and "baseline" not in st.session_state:
        st.session_state["baseline"] = __import__("json").loads(baseline_path.read_text())

    baseline = (
        EvalReport.from_json(st.session_state["baseline"])
        if "baseline" in st.session_state else None
    )
    if baseline:
        st.caption(f"Baseline config: `{baseline.config}`")
        st.dataframe(_metrics_table({"Foundry IQ (baseline)": baseline}))

    # --- Custom run + comparison ---
    st.markdown("#### 2. Evaluate current Custom config")
    st.caption("Set the Custom parameters in the sidebar, then run this.")
    # Rebuild custom overrides from the sidebar sliders via session widgets.
    overrides = {
        "top_k": settings.top_k,
        "reranker_threshold": settings.reranker_threshold,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    if st.button("Run Custom evaluation", type="primary"):
        if baseline is None:
            st.warning("Compute the baseline first (step 1).")
            return
        bar = st.progress(0.0, text="Running Custom evaluation...")
        try:
            pipeline = get_pipeline(settings, "custom", **overrides)
            custom = run_evaluation(
                pipeline, dataset_path, settings,
                metrics=tuple(metrics), limit=lim,
                progress=lambda d, t: bar.progress(d / t, text=f"Custom {d}/{t}"),
            )
            save_report(settings, custom, f"custom_{int(custom.timestamp)}.json")
            st.session_state["custom"] = custom.to_json()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Custom eval failed: {type(exc).__name__}: {exc}")
            return

    if "custom" in st.session_state and baseline is not None:
        custom = EvalReport.from_json(st.session_state["custom"])
        cmp = compare(baseline, custom)
        st.markdown("#### 3. Comparison")
        st.info(f"**Verdict: {cmp.verdict}**")
        table = _metrics_table(
            {"Foundry IQ (baseline)": baseline, "Custom": custom}
        )
        table.loc["— winner —"] = [cmp.winners.get(m, "n/a") for m in ALL_METRICS]
        table["Δ (custom−base)"] = pd.Series(cmp.deltas)
        st.dataframe(table)
        st.caption(f"Custom config: `{custom.config}`")


def main() -> None:
    try:
        settings = _settings()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Configuration error — check your .env file.\n\n{type(exc).__name__}: {exc}")
        st.stop()

    st.title("📄 SGS Document Assistant")
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
