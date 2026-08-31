"""RAG orchestration: ingestion, plus a back-compat answering entry point.

Answering now lives in the pipeline strategies (``src/pipelines/``). This module
keeps the ingestion pipeline and a thin ``answer_question`` shim that delegates to
``CustomRagPipeline`` so existing callers (eval scripts, notebooks) keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openai import AzureOpenAI

from . import search_index
from .chunker import chunk_pages
from .config import Settings
from .embeddings import build_client, embed_texts
from .pdf_loader import extract_pages
from .pipelines.base import Answer
from .pipelines.custom_rag import SYSTEM_PROMPT, CustomRagPipeline

# Re-exported for back-compat with modules that imported them from here.
__all__ = ["Answer", "SYSTEM_PROMPT", "IngestResult", "ingest_files", "answer_question"]


@dataclass
class IngestResult:
    files_processed: int
    chunks_uploaded: int
    skipped_files: list[str]


def ingest_files(
    settings: Settings,
    pdf_paths: list[Path],
    doc_source: str,
    client: AzureOpenAI | None = None,
    replace: bool = True,
) -> IngestResult:
    """Extract, chunk, embed and push the given PDFs to the search index.

    ``doc_source`` tags each chunk ("seed" or "custom"). When ``replace`` is
    True, existing chunks for each file are deleted first so re-ingesting a file
    doesn't create duplicates.
    """
    search_index.create_index_if_not_exists(settings)
    client = client or build_client(settings)

    total_chunks = 0
    processed = 0
    skipped: list[str] = []

    for pdf_path in pdf_paths:
        source_file = pdf_path.name
        pages = extract_pages(pdf_path)
        if not pages:
            skipped.append(source_file)
            continue

        chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            skipped.append(source_file)
            continue

        if replace:
            search_index.delete_documents_for_file(settings, source_file)

        vectors = embed_texts(client, settings, [c.text for c in chunks])
        documents = [
            {
                "id": search_index.make_doc_id(source_file, c.chunk_index),
                "content": c.text,
                "content_vector": vector,
                "source_file": source_file,
                "page": c.page_number,
                "chunk_index": c.chunk_index,
                "doc_source": doc_source,
            }
            for c, vector in zip(chunks, vectors)
        ]
        total_chunks += search_index.upload_documents(settings, documents)
        processed += 1

    return IngestResult(
        files_processed=processed,
        chunks_uploaded=total_chunks,
        skipped_files=skipped,
    )


def answer_question(
    settings: Settings,
    question: str,
    embed_client: AzureOpenAI | None = None,
    chat_client: AzureOpenAI | None = None,
) -> Answer:
    """Back-compat shim: answer via the Custom RAG pipeline with default settings.

    New code should construct a ``Pipeline`` (``CustomRagPipeline`` or
    ``FoundryIQPipeline``) directly and call ``.answer()``.
    """
    pipeline = CustomRagPipeline(
        settings, embed_client=embed_client, chat_client=chat_client
    )
    return pipeline.answer(question)
