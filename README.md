# SGS Document Assistant — RAG Chatbot POC

A proof-of-concept Retrieval-Augmented Generation (RAG) platform over a set of SGS
policy / general-conditions PDFs. Users can also upload their own PDFs (stored
locally under `custom_docs/`). Built with **Streamlit**, **Azure AI Search**
(vector + metadata store, hybrid retrieval), **Azure OpenAI** (embeddings + chat),
and **Azure AI Foundry** (agent + Foundry IQ knowledge base).

It offers **two interchangeable retrieval strategies behind one UI**, plus an
**evaluation harness** that compares them:

1. **Default (Foundry IQ)** — a Foundry Agent backed by a Foundry IQ Knowledge Base;
   Microsoft-managed chunking, query planning, retrieval, and reranking. No knobs.
2. **Custom RAG** — the local hybrid pipeline with user-tunable **chunk size,
   chunk overlap, Top-K, and reranker threshold**.

Both modes generate on the same guardrailed `gpt-5-1` deployment
(**Microsoft.DefaultV2** content safety is applied at the deployment level, so tuning
the Custom knobs can't bypass guardrails), share the Foundry model + agent identity,
and feed a common evaluation pipeline. See **[Two modes + evaluation](#two-modes--evaluation)**.

## Quick start

Run from the project root. Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/),
Azure CLI logged in (`az login`), and the Bicep extension (`az bicep install`).

```bash
# 1. Provision Azure AI Search + Azure OpenAI and auto-write .env
#    (skip if you already have a valid .env)
./infra/deploy.sh
#    Optional overrides:
#    SUBSCRIPTION="My Sub" RESOURCE_GROUP=sgs-rag-poc LOCATION=eastus ENV_NAME=sgs-rag ./infra/deploy.sh

# 2. (Only if you skipped step 1) configure by hand
cp .env.example .env        # then edit endpoints / keys / deployment names

# 3. Create the environment from pyproject.toml + uv.lock
uv sync

# 4. Ingest the 10 SGS PDFs (creates the Azure AI Search index + uploads chunks)
uv run python ingest.py

# 5. Launch the chatbot
uv run streamlit run app.py
```

From the app you can ask questions (grounded answers with `[file p.N]` citations),
upload your own PDFs in the sidebar → **Save & index uploads**, and adjust `TOP_K`
live. Re-run step 4 after changing chunking or the embedding model (if you change
`EMBEDDING_DIMENSIONS`, drop the index first — vector dimensions are fixed at
index creation). Teardown: `RESOURCE_GROUP=sgs-rag-poc ./infra/teardown.sh`
(add `DELETE_GROUP=true` to remove the whole resource group).

Detailed explanations of each step are in the sections below.

## How it works

```
PDFs ──► PyMuPDF (text per page) ──► token-based chunking ──► Azure OpenAI embeddings
                                                                      │
                                                                      ▼
                                        Azure AI Search index (HNSW vector + BM25 text)
                                                                      │
   question ──► embed ──► HYBRID query (vector + keyword, RRF fused) ─┘
                                                                      │
                                          top-K chunks ──► Azure OpenAI chat ──► grounded answer + citations
```

- **Retrieval:** hybrid search — vector similarity (HNSW, cosine) + BM25 keyword,
  fused with Reciprocal Rank Fusion. This is Microsoft's recommended classic-RAG
  pattern (semantic reranker intentionally omitted for the POC).
- **Ingestion:** the "push API" pattern — we chunk and embed locally, then push
  documents to the index. Source files never leave the machine / repo.
- **Everything tunable:** chunk size, overlap, embedding model, chat model,
  top-K, temperature — all via `.env` (see `.env.example`).

## Two modes + evaluation

The app is structured as a **strategy pattern**: a small `Pipeline` interface
(`src/pipelines/base.py`) with two implementations, selected by a sidebar toggle.
Everything below the pipeline (the `gpt-5-1` model, Microsoft.DefaultV2 guardrails,
agent identity, the evaluation harness) is shared.

```
                    Streamlit UI  (sidebar toggle: Default | Custom)
                                     │  Pipeline.answer(q)
              ┌──────────────────────┴───────────────────────┐
     FoundryIQPipeline                                CustomRagPipeline
  (Foundry Agent + Foundry IQ KB,                (local hybrid search you tune:
   MCP knowledge_base_retrieve)                   chunk/overlap/top-k/threshold)
              └──────────────────────┬───────────────────────┘
                        common gpt-5-1 · Microsoft.DefaultV2 · agent identity
                                     │
                          Evaluation (src/evaluation.py)
```

- **Pipelines:** `src/pipelines/foundry_iq.py` invokes the Foundry agent via the
  project's Responses API (`agent_reference`); `src/pipelines/custom_rag.py` runs the
  local hybrid retrieval + generation. `src/pipelines/get_pipeline(settings, mode)`
  is the factory. `src/rag.py::answer_question` remains as a back-compat shim.
- **Foundry agent:** verify or (re)create it with
  `python -m src.foundry_provision [--create]`. It reuses the existing
  `sgs-policy-assistant` agent + `sgs-blob-storage` knowledge base by default
  (configurable in `.env`).
- **Evaluation (Default = baseline):** the **Evaluate** tab runs
  `azure-ai-evaluation` (Groundedness, Relevance, Retrieval, Response Completeness,
  F1) plus retrieval recall@k over `eval/ground_truth.jsonl`. The **Default Foundry
  IQ** result is computed once and **cached as the baseline** (keyed by a dataset
  fingerprint under `eval/results/`); every Custom run is diffed against it with
  per-metric deltas, win/lose badges, and an overall verdict.

> **Quota note:** the shared `gpt-5-1` deployment is capped at ~10K tokens/min. The
> Foundry agent and the LLM-judge evaluators both use it, so evaluation is throttled
> — use the **Max questions** cap in the Evaluate tab for demos, or raise the
> deployment capacity for full runs.

## Voice: speak your question, hear the answer

The **Ask** tab has an input toggle — **⌨️ Type** or **🎙️ Speak**. In Speak mode you record a
question (`st.audio_input`), it's transcribed with **Azure AI Speech STT**, the **Custom RAG**
pipeline **streams** the answer token-by-token (live on screen), and with **🔊 Speak answer** on,
each sentence is synthesized (**Azure Speech TTS**) and played **gaplessly** via a small
audio-queue component — so the answer is spoken as it comes together.

- Speech reuses the existing **AIServices resource** (`hitl-agent-dev3-foundry`, eastus): leave
  `SPEECH_API_KEY` blank to fall back to `AZURE_OPENAI_API_KEY`. Voice/region in `.env`
  (`SPEECH_REGION`, `SPEECH_VOICE`).
- Code: `src/speech.py` (`transcribe`, `synthesize_sentence`, `sentence_chunks`),
  `src/audio_player.py` (gapless queue), `CustomRagPipeline.answer_stream()` (streaming), UI in
  `app.py::_ask_tab`.
- Notes: streaming is wired for **Custom RAG** (the Foundry path falls back to non-streaming);
  first spoken word waits on gpt-5.1's reasoning phase (~5–7s, minimized via
  `reasoning_effort="minimal"`), and the shared `gpt-5-1` 10K-TPM quota still applies.

### Voice output modes: "Streaming (live)" vs "After generation"

The 🔊 **Speak answer** control (Custom mode) offers two voice-output modes:

- **Streaming (live)** — *concurrent* playback: audio begins after the **first sentence** and
  plays gaplessly while later sentences are still generated. This is served by a small
  **FastAPI backend** (`server.py`) that runs the answer once and streams
  **sources → text → per-sentence audio** to the browser over **SSE** (`src/audio_player.py::render_voice_stream`).
  Keys stay server-side. Requires the backend to be running.
- **After generation** — the in-Streamlit fallback: text streams, then all sentences are
  synthesized (in parallel) and played. No backend needed.

**Run both together** with the launcher (starts the FastAPI backend + Streamlit, Ctrl-C stops both):

```bash
./run.sh
```

Or run them separately: `uv run uvicorn server:app --host localhost --port 8000` and
`uv run streamlit run app.py`. Backend URL/host/port are configured via `VOICE_BACKEND_URL` /
`VOICE_BACKEND_HOST` / `VOICE_BACKEND_PORT` in `.env`. The FastAPI `/voice/stream` endpoint is
also the reusable backend a future non-Streamlit frontend (or the talking avatar) would call.

### Agent avatar (Azure real-time TTS Avatar)

An opt-in **talking-head avatar** speaks the answer using **Azure real-time TTS Avatar**
(a photorealistic standard avatar that lip-syncs over WebRTC). It reuses the **same voice
input** (record → STT → RAG) and the same `/voice/stream` answer — only the *speech output*
is handled by the avatar, which synthesizes its own voice + video from the answer text.

- **Enable it:** set `AVATAR_ENABLED=true` in `.env`. It's **off by default** because the
  real-time avatar **bills per minute**. Character/style are configurable
  (`AVATAR_CHARACTER=lisa`, `AVATAR_STYLE=casual-sitting`); the avatar's voice reuses
  `SPEECH_VOICE`, and the session reuses `SPEECH_REGION` + the Speech key.
- **Region:** `SPEECH_REGION` must support real-time avatar (`eastus`, `eastus2`, `westus2`,
  `westeurope`, `swedencentral`, `centralindia`, `southcentralus`, `southeastasia`,
  `italynorth`, `northeurope`, `francecentral`).
- **Use it:** in **Custom** mode pick **Voice output → 🧑 Avatar**, ask (type or 🎙️ speak),
  then click **Connect avatar** in the embedded panel. Needs the backend running (`run.sh`).
- **How it's wired:** the backend serves the avatar client at its own origin
  (`GET /avatar`) plus `GET /avatar/token` (mints the WebRTC ICE relay creds + a short-lived
  Speech auth token — the raw key never reaches the browser). The Streamlit app embeds
  `/avatar` via an iframe; the page opens `/voice/stream?...&audio=off` (text-only) and calls
  `avatarSynthesizer.speakTextAsync()` per sentence. Client code: `static/avatar/`.
- **Cost hygiene:** the avatar session closes on **Disconnect** / tab close, and auto-closes
  after 5 min idle / 30 min max (Azure limits). Keeping it opt-in/default-off is the main
  cost lever.

## Architecture decision: direct `azure-search-documents` SDK (not LangChain's `AzureSearch`)

The retrieval layer talks to Azure AI Search through the **`azure-search-documents`
SDK directly**, not through LangChain's `AzureSearch` vector store. (LangChain is used
only for token-based chunking via `langchain-text-splitters`.) Both would connect to
the same Azure AI Search resource — LangChain's `AzureSearch` is a wrapper over this
same SDK — so this is a choice of control vs. abstraction, not capability.

We keep the lean SDK because, for this POC:

- **First-class metadata schema.** We define typed, filterable fields (`source_file`,
  `page`, `chunk_index`, `doc_source`). LangChain's default packs metadata into a
  single JSON string field, which makes per-field filtering awkward. Our schema is
  what lets the app filter/delete by `source_file` (the re-index feature) and what
  lets the evaluation set score retrieval by document and page (`relevant_docs` /
  `relevant_pages` — see `eval/`). Matching this on LangChain means overriding its
  `fields`, which gives back most of the convenience.
- **Transparent hybrid + RRF.** We issue the exact hybrid query (BM25 `search_text` +
  `VectorizedQuery`) and can see/tune scores directly — no wrapper layer to reason
  through or that lags behind new Azure AI Search features.
- **Lean dependencies.** No `langchain-community` / extra transitive deps to keep in
  sync.

**When to switch to LangChain's `AzureSearch`:** if this grows into a broader
LangChain/LangGraph app — multi-step chains, agents, conversational memory, or a
desire to swap vector stores/models freely. Then the ecosystem glue (retrievers,
LCEL, `create_retrieval_chain`) is worth the extra abstraction. The refactor would be
localized to `search_index.py` and `rag.py`; the same endpoint, key, and index schema
(with custom `fields` to preserve `source_file`/`page` filtering) still apply.

## Project layout

```
app.py               Streamlit UI (mode toggle, Ask + Evaluate tabs, upload, re-index)
server.py            FastAPI backend: "Streaming (live)" voice + avatar page/token endpoints
static/avatar/       agent-avatar client (WebRTC + Azure real-time TTS Avatar SDK)
run.sh               launcher: starts the voice backend + Streamlit together
ingest.py            CLI to ingest the seed PDFs
src/
  speech.py          Azure Speech STT + TTS wrapper (transcribe, synthesize, sentences)
  audio_player.py    browser audio components (gapless queue + SSE voice stream)
  config.py          Settings (pydantic-settings, from .env) incl. Foundry + eval
  pdf_loader.py      PyMuPDF text extraction
  chunker.py         token-based recursive chunking (tiktoken)
  embeddings.py      Azure OpenAI embedding/chat client
  search_index.py    Azure AI Search index + hybrid query (key or RBAC auth)
  rag.py             ingestion + back-compat answering shim
  evaluation.py      eval harness: metrics, cached Foundry IQ baseline, compare()
  foundry_provision.py  verify/create the Foundry agent + KB wiring
  pipelines/
    base.py          Pipeline interface + Answer
    custom_rag.py    Custom RAG strategy (tunable local hybrid search)
    foundry_iq.py    Default strategy (Foundry Agent + Foundry IQ knowledge base)
    __init__.py      get_pipeline() factory + mode labels
custom_docs/         user-uploaded PDFs (local only)
foundry/             agent instructions + knowledge-base setup notes
infra/               Bicep IaC (Azure AI Search + Azure OpenAI) + deploy/teardown
eval/                ground-truth dataset + generator (see eval/README.md)
  results/           cached baseline + saved eval runs (gitignored)
```

The project uses **[uv](https://docs.astral.sh/uv/)** for environment/dependency
management (`pyproject.toml` + `uv.lock`). `requirements.txt` is kept as a pip fallback.

## 1. Provision Azure resources

Requires the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)
(with the Bicep extension — `az bicep install`).

### Option A — Bicep (recommended)

`infra/` provisions **Azure AI Search** + **Azure OpenAI** (chat + embedding
deployments) and writes the resulting endpoints/keys straight into `.env`.

```bash
# optionally override defaults via env vars first:
#   SUBSCRIPTION="My Sub"  RESOURCE_GROUP=sgs-rag-poc  LOCATION=eastus  ENV_NAME=sgs-rag
./infra/deploy.sh
```

The script validates + what-ifs the template, asks for confirmation, deploys, then
fetches the keys and writes `.env` (or `.env.generated` if `.env` already exists).

Bicep layout:

```
infra/
  main.bicep            orchestrates the two modules; emits endpoints as outputs
  main.parameters.json  default environmentName / location
  modules/
    openai.bicep        Azure OpenAI account + gpt-4o-mini + text-embedding-3-small
    search.bicep        Azure AI Search (basic; vector + hybrid; AAD-or-key auth)
  deploy.sh             validate → what-if → deploy → write .env
  teardown.sh           delete the resources (or the whole RG with DELETE_GROUP=true)
```

Model/SKU/capacity/tier are all Bicep parameters (see the top of `main.bicep`).

### Option B — Azure portal

Prefer clicking through the portal? See the step-by-step in the project chat notes,
or create an **Azure AI Search** service (Basic) + an **Azure OpenAI** resource with
`text-embedding-3-small` and `gpt-4o-mini` deployments, then fill `.env` manually
(next step). The app auto-creates the `sgs-docs` index on first ingest.

## 2. Configure

If you used Option B (or want to review what `deploy.sh` wrote):

```bash
cp .env.example .env
# then edit .env with your endpoints / keys / deployment names
```

## 3. Install (uv)

```bash
uv sync                 # creates .venv from pyproject.toml + uv.lock
```

<details>
<summary>pip fallback (no uv)</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
</details>

## 4. Ingest the seed documents

```bash
uv run python ingest.py
```

You should see the index get created and a count of uploaded chunks.

## 5. Run the app

```bash
uv run streamlit run app.py
```

Then:
- Ask a question, e.g. *"What does the SGS anti-corruption policy prohibit?"* —
  you'll get an answer with `[file p.N]` citations and expandable source chunks.
- Upload a PDF in the sidebar → **Save & index uploads** → ask about it. Its
  sources will be tagged `custom`.

## Tuning

Change any value in `.env` and re-ingest (for chunking/embedding changes) or just
re-ask (for `TOP_K`, temperature). `TOP_K` can also be adjusted live from the
sidebar slider.

| Setting | Meaning | Default |
| --- | --- | --- |
| `CHUNK_SIZE` | chunk size in tokens | 512 |
| `CHUNK_OVERLAP` | overlap in tokens | 128 |
| `EMBEDDING_DIMENSIONS` | embedding vector size | 1536 |
| `TOP_K` | chunks retrieved per query | 5 |
| `CHAT_TEMPERATURE` | generation randomness | 0.0 |
| `CHAT_MAX_TOKENS` | max answer length | 800 |

> Note: if you change `EMBEDDING_DIMENSIONS` or the embedding model, delete/recreate
> the index (drop it in the Azure portal) since vector dimensions are fixed at
> index creation, then re-ingest.
