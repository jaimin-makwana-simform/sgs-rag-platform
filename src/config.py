"""Central configuration, loaded from environment / .env.

Every tunable (chunk size/overlap, models, top-k, generation params) lives here so
the whole pipeline can be reconfigured without touching code.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings sourced from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Azure AI Search ----
    azure_search_endpoint: str = Field(..., alias="AZURE_SEARCH_ENDPOINT")
    # Blank when the search service has local (key) auth disabled — the app then
    # authenticates via Azure AD (RBAC) using DefaultAzureCredential.
    azure_search_api_key: str = Field("", alias="AZURE_SEARCH_API_KEY")
    azure_search_index_name: str = Field("sgs-docs", alias="AZURE_SEARCH_INDEX_NAME")

    # ---- Azure OpenAI ----
    azure_openai_endpoint: str = Field(..., alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(..., alias="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str = Field("2024-10-21", alias="AZURE_OPENAI_API_VERSION")
    azure_openai_embedding_deployment: str = Field(
        "text-embedding-3-small", alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    )
    embedding_dimensions: int = Field(1536, alias="EMBEDDING_DIMENSIONS")
    azure_openai_chat_deployment: str = Field(
        "gpt-5.1", alias="AZURE_OPENAI_CHAT_DEPLOYMENT"
    )

    # ---- Azure AI Foundry (project + agent for the Default "Foundry IQ" mode) ----
    # The Foundry project endpoint (services.ai.azure.com) and project name host the
    # agent + knowledge base. Auth is Azure AD (DefaultAzureCredential / `az login`).
    foundry_project_endpoint: str = Field("", alias="FOUNDRY_PROJECT_ENDPOINT")
    foundry_project_name: str = Field("", alias="FOUNDRY_PROJECT_NAME")
    foundry_agent_name: str = Field("sgs-policy-assistant", alias="FOUNDRY_AGENT_NAME")
    foundry_knowledge_base_name: str = Field(
        "sgs-blob-storage", alias="FOUNDRY_KNOWLEDGE_BASE_NAME"
    )

    @property
    def foundry_project_url(self) -> str:
        """Full AIProjectClient endpoint: ``<account>/api/projects/<project>``.

        Accepts either a bare account endpoint (services.ai.azure.com) plus a
        project name, or an already-complete project URL in the endpoint field.
        """
        ep = self.foundry_project_endpoint.rstrip("/")
        if "/api/projects/" in ep or not self.foundry_project_name:
            return ep
        return f"{ep}/api/projects/{self.foundry_project_name}"

    # ---- Azure AI Speech (voice input/output) ----
    # Reuses the AIServices resource (dev3) by default: leave the key blank to fall
    # back to AZURE_OPENAI_API_KEY, and set the region the resource lives in.
    speech_region: str = Field("eastus", alias="SPEECH_REGION")
    speech_api_key: str = Field("", alias="SPEECH_API_KEY")
    speech_voice: str = Field("en-US-AvaMultilingualNeural", alias="SPEECH_VOICE")

    @property
    def effective_speech_key(self) -> str:
        """Speech key, falling back to the shared AIServices/OpenAI key."""
        return self.speech_api_key or self.azure_openai_api_key

    # ---- Voice streaming backend (FastAPI SSE service for concurrent TTS) ----
    voice_backend_url: str = Field("http://localhost:8000", alias="VOICE_BACKEND_URL")
    voice_backend_host: str = Field("localhost", alias="VOICE_BACKEND_HOST")
    voice_backend_port: int = Field(8000, alias="VOICE_BACKEND_PORT")

    # ---- Pipeline mode ----
    # Which retrieval/answering strategy to use by default. The UI overrides this per
    # session. "custom" = local hybrid pipeline (tunable); "foundry_iq" = Foundry Agent.
    rag_mode: str = Field("custom", alias="RAG_MODE")

    # ---- Evaluation ----
    eval_dataset_path: str = Field("eval/ground_truth.jsonl", alias="EVAL_DATASET_PATH")
    eval_results_dir: str = Field("eval/results", alias="EVAL_RESULTS_DIR")

    # Optional: run the CHAT model on a DIFFERENT Azure OpenAI resource than the
    # embeddings. Leave blank to use the same resource as above (the portable,
    # self-contained default). Handy when this resource can't host a chat model
    # yet (e.g. a deployment/quota block) — point chat at any resource that has
    # one, without moving embeddings or re-indexing.
    azure_openai_chat_endpoint: str = Field("", alias="AZURE_OPENAI_CHAT_ENDPOINT")
    azure_openai_chat_api_key: str = Field("", alias="AZURE_OPENAI_CHAT_API_KEY")
    azure_openai_chat_api_version: str = Field("", alias="AZURE_OPENAI_CHAT_API_VERSION")

    @property
    def chat_endpoint(self) -> str:
        return self.azure_openai_chat_endpoint or self.azure_openai_endpoint

    @property
    def chat_api_key(self) -> str:
        return self.azure_openai_chat_api_key or self.azure_openai_api_key

    @property
    def chat_api_version(self) -> str:
        return self.azure_openai_chat_api_version or self.azure_openai_api_version

    # ---- Chunking (tokens) ----
    chunk_size: int = Field(512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(128, alias="CHUNK_OVERLAP")

    # ---- Retrieval / generation ----
    top_k: int = Field(5, alias="TOP_K")
    # Relevance gate: minimum semantic reranker score (0-4 scale) a chunk must
    # reach to be used. If nothing clears it, the app refuses instead of calling
    # the LLM. Calibrate with eval/ (answerable vs unanswerable questions).
    reranker_threshold: float = Field(1.8, alias="RERANKER_THRESHOLD")
    chat_temperature: float = Field(0.0, alias="CHAT_TEMPERATURE")  # unused for GPT-5 models (default only)
    # GPT-5 reasoning tokens count toward this limit, so keep generous headroom
    # or answers can come back empty.
    chat_max_tokens: int = Field(2048, alias="CHAT_MAX_TOKENS")

    # ---- Local document folders ----
    docs_dirs: str = Field(".,General_Conditions,custom_docs", alias="DOCS_DIRS")
    custom_docs_dir: str = Field("custom_docs", alias="CUSTOM_DOCS_DIR")

    @property
    def docs_dirs_list(self) -> list[str]:
        return [d.strip() for d in self.docs_dirs.split(",") if d.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
