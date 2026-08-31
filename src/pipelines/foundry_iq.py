"""Default "Foundry IQ" pipeline: a Foundry Agent + Knowledge Base.

Retrieval and answer synthesis are delegated to a Foundry Prompt Agent that is
wired to a Foundry IQ Knowledge Base (over the SGS documents in blob) via an MCP
``knowledge_base_retrieve`` tool. Foundry manages chunking, query planning,
retrieval and reranking with its default settings — the "no knobs" counterpart to
the Custom pipeline. Generation runs on the same guardrailed ``gpt-5-1`` model, so
Microsoft.DefaultV2 is enforced here too.

The agent is invoked through the project's OpenAI-compatible Responses API using an
``agent_reference`` (this is what the Foundry service expects for prompt agents).
"""

from __future__ import annotations

import time

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from ..config import Settings
from ..search_index import Retrieved
from .base import Answer, Pipeline

# Transient gpt-5.1 rate limits are common on shared, low-quota deployments. The
# token bucket renews every 60s, so back off ~30s between tries (a couple of tries
# usually clears a burst-induced 429 without waiting a full renewal window).
_MAX_RETRIES = 4
_BACKOFF_SECONDS = 30


class FoundryIQPipeline(Pipeline):
    """Answer via a Foundry Agent backed by a Foundry IQ Knowledge Base."""

    mode = "foundry_iq"

    def __init__(
        self,
        settings: Settings,
        *,
        project_client: AIProjectClient | None = None,
    ) -> None:
        self.settings = settings
        self.agent_name = settings.foundry_agent_name
        self._client = project_client or AIProjectClient(
            endpoint=settings.foundry_project_url,
            credential=DefaultAzureCredential(),
        )
        self._openai = self._client.get_openai_client()

    def answer(self, question: str) -> Answer:
        response = self._create_response(question)
        text = getattr(response, "output_text", "") or ""
        sources = self._extract_sources(response)
        return Answer(text=text, sources=sources)

    def _create_response(self, question: str):
        """Call the agent via the Responses API, retrying transient rate limits."""
        agent_ref = {
            "agent_reference": {"name": self.agent_name, "type": "agent_reference"}
        }
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._openai.responses.create(
                    input=question, extra_body=agent_ref
                )
            except Exception as exc:  # noqa: BLE001 - retry only on rate limits
                last_exc = exc
                if "429" in str(exc) or "rate_limit" in str(exc).lower():
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(_BACKOFF_SECONDS)
                        continue
                raise
        raise last_exc  # pragma: no cover

    @staticmethod
    def _extract_sources(response) -> list[Retrieved]:
        """Best-effort mapping of Responses-API citations into ``Retrieved``.

        The Foundry knowledge tool returns citations as annotations on the output
        text (and, depending on the tool, as tool-call results). We collect any
        annotation that names a source document so the UI can show provenance;
        fields we can't determine (page, score) fall back to neutral defaults.
        """
        try:
            data = response.model_dump()
        except Exception:  # noqa: BLE001
            return []

        sources: list[Retrieved] = []
        seen: set[str] = set()

        def add(title: str | None, snippet: str, page: int = 0) -> None:
            name = (title or "").strip() or "knowledge-base"
            key = f"{name}::{snippet[:60]}"
            if key in seen:
                return
            seen.add(key)
            sources.append(
                Retrieved(
                    content=snippet,
                    source_file=name,
                    page=page,
                    chunk_index=0,
                    doc_source="foundry_iq",
                    score=0.0,
                    reranker_score=0.0,
                )
            )

        def walk(node) -> None:
            if isinstance(node, dict):
                ntype = node.get("type")
                if ntype in ("file_citation", "url_citation", "container_file_citation"):
                    add(
                        node.get("filename")
                        or node.get("title")
                        or node.get("url"),
                        str(node.get("text") or node.get("snippet") or ""),
                    )
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(data.get("output", data))
        return sources

    def describe_config(self) -> dict:
        return {
            "mode": self.mode,
            "chat_model": self.settings.azure_openai_chat_deployment,
            "agent_name": self.agent_name,
            "knowledge_base": self.settings.foundry_knowledge_base_name,
            "retrieval": "foundry-iq-managed",
        }
