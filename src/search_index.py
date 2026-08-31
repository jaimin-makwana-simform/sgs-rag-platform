"""Azure AI Search: index management, document upload, and hybrid retrieval.

We push pre-computed embeddings (the "push API" pattern) rather than using
integrated vectorization, because source documents stay local (not in Blob Storage).

Retrieval is hybrid: passing both ``search_text`` (BM25 keyword) and a
``vector_queries`` entry (HNSW vector) in one request makes Azure AI Search fuse
the two result sets with Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from azure.search.documents.models import QueryType, VectorizedQuery

from .config import Settings

_VECTOR_PROFILE = "hnsw-cosine-profile"
_VECTOR_ALGORITHM = "hnsw-config"
_SEMANTIC_CONFIG = "sgs-semantic"


@dataclass
class Retrieved:
    """A retrieved chunk with metadata for grounding / citation."""

    content: str
    source_file: str
    page: int
    chunk_index: int
    doc_source: str
    score: float          # RRF fusion score (rank-based; not a relevance magnitude)
    reranker_score: float  # semantic reranker score 0-4 (used for the relevance gate)


def make_doc_id(source_file: str, chunk_index: int) -> str:
    """Deterministic, key-safe document id (stable across re-ingests)."""
    raw = f"{source_file}::{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _credential(settings: Settings):
    """Search credential: API key if one is set, else Azure AD (RBAC).

    Some Azure AI Search services have local (key) auth disabled and require
    RBAC. Leave AZURE_SEARCH_API_KEY blank in that case and authenticate via
    DefaultAzureCredential (e.g. `az login`, managed identity, env vars). The
    signed-in principal needs the "Search Index Data Contributor" and
    "Search Service Contributor" roles on the service.
    """
    if settings.azure_search_api_key:
        return AzureKeyCredential(settings.azure_search_api_key)
    return DefaultAzureCredential()


def _index_client(settings: Settings) -> SearchIndexClient:
    return SearchIndexClient(
        endpoint=settings.azure_search_endpoint,
        credential=_credential(settings),
    )


def search_client(settings: Settings) -> SearchClient:
    return SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=settings.azure_search_index_name,
        credential=_credential(settings),
    )


def _build_index(settings: Settings) -> SearchIndex:
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=settings.embedding_dimensions,
            vector_search_profile_name=_VECTOR_PROFILE,
        ),
        SimpleField(
            name="source_file",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
            sortable=True,
        ),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
        SimpleField(
            name="doc_source",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=_VECTOR_ALGORITHM,
                parameters=HnswParameters(
                    metric=VectorSearchAlgorithmMetric.COSINE,
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                ),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name=_VECTOR_PROFILE,
                algorithm_configuration_name=_VECTOR_ALGORITHM,
            )
        ],
    )

    # Semantic configuration powers the reranker (relevance gate + reranking).
    # It's index metadata — adding it needs no re-embedding/re-indexing.
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=_SEMANTIC_CONFIG,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")],
                ),
            )
        ]
    )

    return SearchIndex(
        name=settings.azure_search_index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


def create_index_if_not_exists(settings: Settings) -> bool:
    """Create the index if missing; otherwise ensure the semantic config exists.

    Returns True if the index was created. On an existing index, the semantic
    configuration is patched in if absent (a metadata-only update — existing
    documents/vectors are untouched).
    """
    client = _index_client(settings)
    existing = {name for name in client.list_index_names()}
    index = _build_index(settings)

    if settings.azure_search_index_name not in existing:
        client.create_index(index)
        return True

    current = client.get_index(settings.azure_search_index_name)
    has_semantic = bool(
        current.semantic_search and current.semantic_search.configurations
    )
    if not has_semantic:
        current.semantic_search = index.semantic_search
        client.create_or_update_index(current)
    return False


def delete_documents_for_file(settings: Settings, source_file: str) -> int:
    """Delete all chunks for a given source file (used before re-ingesting it)."""
    client = search_client(settings)
    results = client.search(
        search_text="*",
        filter=f"source_file eq '{source_file}'",
        select=["id"],
        top=1000,
    )
    ids = [{"id": r["id"]} for r in results]
    if ids:
        client.delete_documents(documents=ids)
    return len(ids)


def upload_documents(settings: Settings, documents: list[dict]) -> int:
    """Upload/merge documents in batches. Returns number uploaded."""
    client = search_client(settings)
    batch_size = 1000
    uploaded = 0
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        client.upload_documents(documents=batch)
        uploaded += len(batch)
    return uploaded


def document_count(settings: Settings) -> int:
    """Total number of documents (chunks) in the index."""
    client = search_client(settings)
    return client.get_document_count()


def hybrid_search(
    settings: Settings,
    query_text: str,
    query_vector: list[float],
    top_k: int,
) -> list[Retrieved]:
    """Run a hybrid (BM25 + vector) query fused with RRF."""
    client = search_client(settings)
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="content_vector",
    )
    results = client.search(
        search_text=query_text,
        vector_queries=[vector_query],
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name=_SEMANTIC_CONFIG,
        select=["content", "source_file", "page", "chunk_index", "doc_source"],
        top=top_k,
    )
    return [
        Retrieved(
            content=r["content"],
            source_file=r["source_file"],
            page=r["page"],
            chunk_index=r["chunk_index"],
            doc_source=r["doc_source"],
            score=r["@search.score"],
            reranker_score=r.get("@search.reranker_score") or 0.0,
        )
        for r in results
    ]
