"""Retrieval/answering strategies behind a common Pipeline interface."""

from __future__ import annotations

from ..config import Settings
from .base import Answer, Pipeline
from .custom_rag import CustomRagPipeline
from .foundry_iq import FoundryIQPipeline

__all__ = [
    "Answer",
    "Pipeline",
    "CustomRagPipeline",
    "FoundryIQPipeline",
    "get_pipeline",
]

# Stable mode identifiers ↔ UI labels.
MODE_LABELS = {
    "custom": "Custom RAG",
    "foundry_iq": "Default (Foundry IQ)",
}


def get_pipeline(settings: Settings, mode: str, **overrides) -> Pipeline:
    """Construct a pipeline for the given mode.

    ``overrides`` are forwarded to the Custom pipeline (top_k, reranker_threshold,
    chunk_size, chunk_overlap); the Foundry IQ pipeline ignores them (it manages
    its own retrieval).
    """
    if mode == "foundry_iq":
        return FoundryIQPipeline(settings)
    if mode == "custom":
        return CustomRagPipeline(settings, **overrides)
    raise ValueError(f"Unknown RAG mode: {mode!r}")
