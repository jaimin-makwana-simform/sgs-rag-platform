"""Custom RAG pipeline: the transparent, tunable local hybrid-search strategy.

Retrieval uses Azure AI Search hybrid (BM25 + vector, RRF) with a semantic
reranker relevance gate; generation runs on the guardrailed Foundry chat model
(Microsoft.DefaultV2 is applied at the deployment level, so tuning the knobs here
cannot bypass guardrails). Chunk size / overlap / Top-K / reranker threshold are
all user-tunable per session.
"""

from __future__ import annotations

from openai import AzureOpenAI

from .. import search_index
from ..config import Settings
from ..embeddings import build_chat_client, build_client, embed_query
from .base import Answer, Pipeline

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about SGS policy and "
    "general-conditions documents. Answer ONLY using the provided context. "
    "If the answer is not contained in the context, say you don't have that "
    "information in the available documents. Cite the sources you use inline "
    "using the format [source_file p.PAGE]. Be concise and accurate.\n"
    "When — and only when — the provided context contains BOTH a general/default "
    "rule AND a narrower region- or case-specific clause (e.g. a 'Special Terms' "
    "override, or a rule that applies only if a stated condition is met) that both "
    "bear on the question, present both: state the default rule, then the "
    "override and the exact condition under which it applies. Do not invent, "
    "assume, or imply an override that is not present in the context — if only a "
    "single rule is given, answer with just that rule."
)


def format_context(chunks: list) -> str:
    """Render retrieved chunks into the grounded-context block for the prompt."""
    blocks = []
    for c in chunks:
        blocks.append(f"[{c.source_file} p.{c.page}]\n{c.content}")
    return "\n\n---\n\n".join(blocks)


class CustomRagPipeline(Pipeline):
    """Local hybrid-search RAG with user-tunable retrieval parameters.

    Per-session overrides (``top_k``, ``reranker_threshold``, ``chunk_size``,
    ``chunk_overlap``) fall back to the values in ``Settings`` when not provided.
    ``chunk_size``/``chunk_overlap`` do not affect ``answer()`` directly (they are
    ingest-time), but are tracked here so re-indexing and evaluation records use
    the same effective config.
    """

    mode = "custom"

    def __init__(
        self,
        settings: Settings,
        *,
        embed_client: AzureOpenAI | None = None,
        chat_client: AzureOpenAI | None = None,
        top_k: int | None = None,
        reranker_threshold: float | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.settings = settings
        self.embed_client = embed_client or build_client(settings)
        self.chat_client = chat_client or build_chat_client(settings)
        self.top_k = top_k if top_k is not None else settings.top_k
        self.reranker_threshold = (
            reranker_threshold
            if reranker_threshold is not None
            else settings.reranker_threshold
        )
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        )

    def answer(self, question: str) -> Answer:
        query_vector = embed_query(self.embed_client, self.settings, question)
        retrieved = search_index.hybrid_search(
            self.settings, question, query_vector, self.top_k
        )

        # Relevance gate: keep only chunks whose semantic reranker score clears the
        # threshold. For out-of-scope questions nothing clears it, so we skip the LLM
        # call entirely and return no sources.
        relevant = [
            r for r in retrieved if r.reranker_score >= self.reranker_threshold
        ]
        if not relevant:
            return Answer(
                text=(
                    "I couldn't find anything relevant in the available SGS documents, "
                    "so I can't answer that. Try asking about the SGS policies or "
                    "general-conditions documents."
                ),
                sources=[],
            )

        context = format_context(relevant)
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the context above, with inline [source p.PAGE] citations."
        )

        # GPT-5 family models require `max_completion_tokens` instead of `max_tokens`
        # and only support the default temperature — so we don't send `temperature`.
        # These params are also accepted by gpt-4.x models, so this works across
        # model generations.
        response = self.chat_client.chat.completions.create(
            model=self.settings.azure_openai_chat_deployment,
            max_completion_tokens=self.settings.chat_max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return Answer(text=response.choices[0].message.content, sources=relevant)

    def describe_config(self) -> dict:
        return {
            "mode": self.mode,
            "chat_model": self.settings.azure_openai_chat_deployment,
            "embedding_model": self.settings.azure_openai_embedding_deployment,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "reranker_threshold": self.reranker_threshold,
        }
