# Document Query Assistant — RAG Chatbot POC

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

## Quick start (end to end)

Run everything from the **project root**. This gets you from nothing to a working POC —
chat, the Foundry IQ vs Custom RAG comparison, the evaluation harness, **and** the
voice + talking-avatar features.

**Prerequisites**
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python env/deps).
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli), logged in (`az login`),
  with the Bicep extension (`az bicep install`).
- Azure access to: **Azure AI Search** + **Azure OpenAI** (`gpt-5-1` chat + `text-embedding-3-small`).
  For **voice + avatar** you also need an **Azure AI Speech** (or multi-service *AIServices*)
  resource in an avatar-supported region (e.g. `eastus`). For **Default (Foundry IQ)** mode you
  need an **Azure AI Foundry** project.
- A **Chromium browser (Chrome or Edge)** to view the talking avatar (Firefox needs Coturn ICE).

```bash
# 1. Provision core infra — Azure AI Search + Azure OpenAI (gpt-5-1 + embeddings) — and
#    auto-write .env. Deploys into an EXISTING resource group (see infra/deploy.sh).
#    NOTE: this does NOT create the Speech or Foundry resources (configure those in step 2).
./infra/deploy.sh
#    Optional overrides:
#    SUBSCRIPTION="My Sub" RESOURCE_GROUP=AI-CoE-rg LOCATION=eastus ENV_NAME=sgs-rag ./infra/deploy.sh

# 2. Configure .env
#    - If you skipped step 1:            cp .env.example .env   # then fill endpoints/keys
#    - Voice + avatar (Azure Speech):    set SPEECH_REGION (avatar-supported, e.g. eastus);
#                                        set SPEECH_API_KEY unless AZURE_OPENAI_* is a
#                                        multi-service AIServices resource that includes Speech
#                                        (then leave it blank to reuse AZURE_OPENAI_API_KEY).
#    - Avatar (opt-in; bills per min):   AVATAR_ENABLED=true   (AVATAR_CHARACTER/STYLE optional)
#    - Default / Foundry IQ mode:        FOUNDRY_PROJECT_ENDPOINT + FOUNDRY_PROJECT_NAME

# 3. Create the Python environment from pyproject.toml + uv.lock
uv sync

# 4. Ingest the seed SGS PDFs (creates the Azure AI Search index + uploads chunks)
uv run python ingest.py

# 5. (Default / Foundry IQ mode only) verify or create the Foundry agent + KB
uv run python -m src.foundry_provision --create      # omit --create to only verify

# 6. Run EVERYTHING — FastAPI voice/avatar backend (:8000) + Streamlit UI (:8501).
#    Ctrl-C stops both. Use this instead of `streamlit run` so voice + avatar work.
./run.sh
```

Open **http://localhost:8501**. You can ask questions (grounded answers with `[file p.N]`
citations), toggle **Default (Foundry IQ) ↔ Custom RAG** in the sidebar, upload your own PDFs
→ **Save & index uploads**, run the **Evaluate** tab, and in Custom mode pick **Voice output →
🔊 Streaming (live)** or **🧑 Avatar**.

**Feature checklist** (what each one needs):

| Feature | Requires |
| --- | --- |
| Chat / Custom RAG | steps 1–4, then `./run.sh` (or `uv run streamlit run app.py`) |
| Default (Foundry IQ) mode | `FOUNDRY_*` in `.env` + step 5 + `az login` (RBAC) |
| Voice (STT + TTS streaming) | Azure Speech in `.env` (`SPEECH_REGION`/`SPEECH_API_KEY`) + `./run.sh` |
| 🧑 Agent avatar | voice prerequisites + `AVATAR_ENABLED=true` + avatar-supported `SPEECH_REGION` + Chrome/Edge |
| Evaluation | Foundry IQ mode (baseline) + `gpt-5-1` judge (mind the 10K-TPM quota) |

Re-run step 4 after changing chunking or the embedding model (if you change
`EMBEDDING_DIMENSIONS`, drop the index first — vector dimensions are fixed at index creation).
Uploads are capped at **5 MB** (`.streamlit/config.toml`). Teardown:
`RESOURCE_GROUP=AI-CoE-rg ./infra/teardown.sh` (add `DELETE_GROUP=true` to remove the whole
resource group). Detailed explanations of each step are in the sections below.

## Reusing shared team resources (no provisioning)

If a teammate already provisioned the Azure resources in your shared resource group, you
**don't create anything** — reuse theirs. Skip the provisioning steps and just wire up locally.

**Skip these** (already exist, shared):
- `./infra/deploy.sh` — the Search service, Azure OpenAI (`gpt-5-1` + embeddings), Speech, and
  Foundry project already exist.
- `python -m src.foundry_provision --create` — the `sgs-policy-assistant` agent + KB already
  exist (at most run it *without* `--create` to verify).
- `uv run python ingest.py` — the Azure AI Search index (`sgs-docs`) is **shared**; if the seed
  PDFs are already ingested, the chunks are there. Ingest only needs to run **once, by anyone**.

**Do these locally:**
```bash
# 1. Get the .env — it's gitignored, so it is NOT in the repo. Ask the owner to share it
#    securely (it contains keys), or copy .env.example and fill in the shared endpoints/keys.
cp .env.example .env        # then paste the shared values

# 2. Python env
uv sync

# 3. Only if you use Default (Foundry IQ) mode or RBAC-auth Search: sign in.
#    Your identity also needs the role assignments (e.g. "Search Index Data Reader" and
#    access to the Foundry project). Key-based resources (chat, Speech) need no login.
az login

# 4. For the avatar: set AVATAR_ENABLED=true in .env (Chrome/Edge required).

# 5. Run everything
./run.sh
```

**Auth, at a glance** (this setup): chat = API key (works for anyone with the key), **Search =
RBAC**, **Foundry agent = RBAC** (both need `az login` + a role assignment on *your* identity),
Speech = API key. So a teammate who only wants **Custom RAG + voice + avatar** (all key-based)
may not need `az login` at all; **Default (Foundry IQ) mode** does.

**Shared-resource cautions:**
- **`gpt-5-1` is a shared 10K-TPM deployment** — several people running chat/voice/eval at once
  will hit 429s much sooner. Keep eval **Max questions** low.
- **The `sgs-docs` index is shared** — sidebar uploads (**Save & index uploads**) and especially
  **changing Custom chunk size / overlap** rebuild the shared index, affecting *everyone*. Don't
  change chunking on a shared index unless the team agrees.

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
.streamlit/config.toml  Streamlit config (upload limit capped at 5 MB)
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
    openai.bicep        Azure OpenAI account (kind=OpenAI) + gpt-5-1 + text-embedding-3-small
    search.bicep        Azure AI Search (basic; vector + hybrid; AAD-or-key auth)
  deploy.sh             validate → what-if → deploy → write .env
  teardown.sh           delete the resources (or the whole RG with DELETE_GROUP=true)
```

Model/SKU/capacity/tier are all Bicep parameters (see the top of `main.bicep`).

> **Not provisioned by Bicep:** Azure **Speech** (for voice/avatar) and the Azure **AI
> Foundry** project/agent (for Default mode). The OpenAI resource is `kind=OpenAI`, so its
> key does **not** cover Speech — set `SPEECH_REGION`/`SPEECH_API_KEY` in `.env` for a Speech
> (or multi-service *AIServices*) resource, and provision the Foundry agent with
> `python -m src.foundry_provision --create`.

### Option B — Azure portal

Prefer clicking through the portal? Create an **Azure AI Search** service (Basic) + an
**Azure OpenAI** resource with `text-embedding-3-small` and `gpt-5-1` deployments, then fill
`.env` manually (next step). For voice/avatar also create an **Azure AI Speech** (or
multi-service *AIServices*) resource in an avatar-supported region and set `SPEECH_*`. The app
auto-creates the `sgs-docs` index on first ingest.

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

## 5. (Default / Foundry IQ mode) provision the Foundry agent

Only needed for the **Default (Foundry IQ)** mode and its evaluation baseline. On the shared
dev3 project the agent + KB already exist, so this just verifies them; `--create` makes them
idempotently on a fresh project (then grant the agent identity the "Search Index Data Reader"
role — see `src/foundry_provision.py`). Requires `az login` (RBAC).

```bash
uv run python -m src.foundry_provision --create   # omit --create to only verify
```

## 6. Run the app

Use the launcher so the **voice backend + avatar** work (it starts FastAPI on `:8000` and
Streamlit on `:8501`, and stops both on Ctrl-C):

```bash
./run.sh
```

Chat-only (no voice/avatar) can run Streamlit alone: `uv run streamlit run app.py`.

Then, at **http://localhost:8501**:
- Ask a question, e.g. *"What does the SGS anti-corruption policy prohibit?"* —
  you'll get an answer with `[file p.N]` citations and expandable source chunks.
- Toggle **Default (Foundry IQ) ↔ Custom RAG** in the sidebar.
- Upload a PDF in the sidebar → **Save & index uploads** → ask about it (sources tagged `custom`).
- In **Custom** mode, set **Voice output → 🔊 Streaming (live)** or **🧑 Avatar** (see
  [Voice](#voice-speak-your-question-hear-the-answer) / [Agent avatar](#agent-avatar-azure-real-time-tts-avatar)).

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
