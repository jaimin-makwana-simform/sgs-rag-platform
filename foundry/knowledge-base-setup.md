# Foundry Knowledge — Setup

> **What this repo actually uses (Default "Foundry IQ" mode).** The Streamlit app's
> Default mode talks to a **Foundry Prompt Agent** wired to a **Foundry IQ knowledge
> base** via the MCP `knowledge_base_retrieve` tool — not the file-upload path below.
> On the current project this is already provisioned:
>
> - **Project:** `hitl-agent-dev3-project` (endpoint `…services.ai.azure.com/api/projects/<project>`)
> - **Agent:** `sgs-policy-assistant` (model `gpt-5-1`, Microsoft.DefaultV2 guardrails)
> - **Knowledge base:** `sgs-blob-storage` (Azure AI Search agentic retrieval over the
>   SGS PDFs in the `policies` blob container), exposed at
>   `…/knowledgebases/sgs-blob-storage/mcp?api-version=2026-05-01-preview`
> - **Agent identity:** system-managed; needs **Search Index Data Reader** on the search
>   service to read the knowledge base.
>
> Verify or (re)create this wiring with:
> ```bash
> python -m src.foundry_provision            # verify + describe the agent
> python -m src.foundry_provision --create   # create it on a fresh project
> ```
> Configure names via `.env` (`FOUNDRY_PROJECT_ENDPOINT`, `FOUNDRY_PROJECT_NAME`,
> `FOUNDRY_AGENT_NAME`, `FOUNDRY_KNOWLEDGE_BASE_NAME`). The agent is invoked from
> `src/pipelines/foundry_iq.py` through the project's Responses API using an
> `agent_reference`.
>
> The portal, file-upload walkthrough below remains a simpler alternative for
> standing up a grounded agent by hand.

---

# Foundry Knowledge (File Upload) — Quick Setup

Short, portal-only guide to ground a Foundry agent in the 10 SGS PDFs by **uploading
the files directly from your device** — no separate storage account, no Blob Storage.
Pair this with [`agent-instructions.md`](./agent-instructions.md) for the system prompt.

> This uses Foundry's **Files / file-search** knowledge tool. When you upload local
> files, Foundry ingests them into a **vector store** in the project's own managed
> storage (chunking + embeddings handled for you). You don't create or manage any
> storage resource yourself.

## Prerequisites
- A **Microsoft Foundry** project — sign in at <https://ai.azure.com> (**New Foundry** toggle **on**).
- A deployed **chat model** in the project (e.g. `gpt-4o-mini`).
- Roles: **Foundry User** on the project, and **Storage Blob Data Contributor** on the
  project's (auto-created) storage account — needed so file upload can write to project storage.
- The 10 PDFs on your machine (this repo's root + `General_Conditions/`).

## Step 1 — Create the agent
1. In your Foundry project, open the **Agents** tab → **+ New agent**.
2. Select your chat model deployment.
3. Paste the system prompt from **`agent-instructions.md`** into the agent's **Instructions**.

## Step 2 — Add Files knowledge (upload from device)
1. In the agent's **Setup** pane (right side), scroll to **Knowledge** → **Add**.
2. Choose **Files**.
3. **Upload** all 10 SGS PDFs from your device (drag-and-drop or browse):
   - the 4 PDFs in the project root, and
   - the 6 PDFs in `General_Conditions/`.
4. Confirm/finish. Foundry creates a **vector store**, ingests the files, and attaches
   the file-search tool to the agent. Wait until ingestion shows complete.

That's it — the uploaded files are now the agent's knowledge base.

## Step 3 — Test in the playground
Open the agent's **Playground** and run the sample prompts from `agent-instructions.md`:
- **Factual:** "What is SGS's policy on facilitation payments?" → grounded answer **with a citation**.
- **Disambiguation:** "Which law governs the China General Conditions?" → **Switzerland / Paris arbitration**.
- **Comparison:** "Sample retention under the Philippines vs China conditions?" → **2 months vs 3 months**.
- **Out-of-scope:** "How many vacation days do SGS employees get?" → **refusal** (not in the documents).

## Updating the document set later
To add or remove documents, reopen **Setup → Knowledge → Files** and upload/delete files;
the vector store re-ingests automatically. (This mirrors the "upload custom docs" feature
in the Streamlit app — here Foundry manages the storage instead of the local `custom_docs/` folder.)

## Notes
- File search has **additional charges** beyond model tokens (ingestion + vector store).
- File-search availability can vary by region; if the **Files** option isn't offered,
  create the project in a supported region.
- This file-upload path is the simplest way to see a grounded prompt-agent working. The
  blob-based **Foundry IQ knowledge base** (agentic retrieval, multi-source, reusable
  across agents) is the heavier alternative — not needed for this POC.
