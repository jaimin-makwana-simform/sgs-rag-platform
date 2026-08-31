"""Azure OpenAI embedding + chat clients."""

from __future__ import annotations

from openai import AzureOpenAI

from .config import Settings

# Azure OpenAI embedding requests are capped; batch to stay well under limits.
_EMBED_BATCH_SIZE = 64


def build_client(settings: Settings) -> AzureOpenAI:
    """Create the AzureOpenAI client for EMBEDDINGS (the main resource)."""
    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )


def build_chat_client(settings: Settings) -> AzureOpenAI:
    """Create the AzureOpenAI client for CHAT.

    Uses the optional chat-specific endpoint/key if set, otherwise falls back to
    the main resource — so by default this is the same resource as embeddings.
    """
    return AzureOpenAI(
        azure_endpoint=settings.chat_endpoint,
        api_key=settings.chat_api_key,
        api_version=settings.chat_api_version,
    )


def embed_texts(
    client: AzureOpenAI,
    settings: Settings,
    texts: list[str],
) -> list[list[float]]:
    """Embed a list of texts, batching requests. Order is preserved."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        response = client.embeddings.create(
            model=settings.azure_openai_embedding_deployment,
            input=batch,
            dimensions=settings.embedding_dimensions,
        )
        vectors.extend(item.embedding for item in response.data)
    return vectors


def embed_query(client: AzureOpenAI, settings: Settings, text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts(client, settings, [text])[0]
