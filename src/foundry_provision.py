"""Verify (or create) the Foundry Agent used by the Default "Foundry IQ" mode.

On the current dev3 project the agent + Foundry IQ knowledge base already exist, so
this script's default action is to *verify* and describe them. With ``--create`` it
will (idempotently) create a Prompt Agent wired to the knowledge base via the MCP
``knowledge_base_retrieve`` tool — the reproducible path for a fresh project.

Usage:
    python -m src.foundry_provision            # verify + describe
    python -m src.foundry_provision --create   # create the agent if missing

After first creation, grant the agent identity the "Search Index Data Reader" role
on the search service (printed at the end).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from .config import Settings, get_settings

ROOT = Path(__file__).resolve().parent.parent
INSTRUCTIONS_FILE = ROOT / "foundry" / "agent-instructions.md"
KB_API_VERSION = "2026-05-01-preview"


def _client(settings: Settings) -> AIProjectClient:
    return AIProjectClient(
        endpoint=settings.foundry_project_url, credential=DefaultAzureCredential()
    )


def _find_agent(client: AIProjectClient, name: str):
    for agent in client.agents.list():
        if agent.name == name:
            return client.agents.get(agent_name=name)
    return None


def _find_kb_connection(client: AIProjectClient, kb_name: str) -> str | None:
    """Return the RemoteTool project-connection id for the knowledge base."""
    remote_tools = []
    for conn in client.connections.list():
        if str(getattr(conn, "type", "")).endswith("RemoteTool") or getattr(
            conn, "type", ""
        ) == "RemoteTool":
            remote_tools.append(conn.name)
    # Prefer a connection whose name references the KB.
    for name in remote_tools:
        if kb_name.split("-")[0] in name or "kb" in name.lower():
            return name
    return remote_tools[0] if remote_tools else None


def _agent_definition(settings: Settings, connection_id: str) -> dict:
    instructions = (
        INSTRUCTIONS_FILE.read_text(encoding="utf-8")
        if INSTRUCTIONS_FILE.exists()
        else "You are an SGS policy knowledge assistant. Answer only from the "
        "knowledge base and cite sources."
    )
    server_url = (
        f"{settings.azure_search_endpoint.rstrip('/')}/knowledgebases/"
        f"{settings.foundry_knowledge_base_name}/mcp?api-version={KB_API_VERSION}"
    )
    return {
        "kind": "prompt",
        "model": settings.azure_openai_chat_deployment,
        "instructions": instructions,
        "tools": [
            {
                "type": "mcp",
                "server_label": connection_id,
                "server_url": server_url,
                "require_approval": "never",
                "project_connection_id": connection_id,
            }
        ],
    }


def _describe(agent) -> None:
    d = agent.as_dict()
    latest = d.get("versions", {}).get("latest", {})
    defn = latest.get("definition", {})
    ident = latest.get("instance_identity", {})
    print(f"  name        : {d.get('name')}")
    print(f"  version     : {latest.get('version')}  (state={d.get('state')})")
    print(f"  model       : {defn.get('model')}")
    for tool in defn.get("tools", []):
        print(f"  tool        : {tool.get('type')} -> {tool.get('server_url', tool)}")
    print(f"  identity    : principal_id={ident.get('principal_id')}")


def provision(settings: Settings, create: bool = False) -> None:
    client = _client(settings)
    name = settings.foundry_agent_name
    agent = _find_agent(client, name)

    if agent is not None:
        print(f"✓ Agent '{name}' exists.")
        _describe(agent)
        return

    print(f"✗ Agent '{name}' not found.")
    if not create:
        print("  Re-run with --create to create it.")
        return

    connection_id = _find_kb_connection(client, settings.foundry_knowledge_base_name)
    if not connection_id:
        raise RuntimeError(
            "No RemoteTool (knowledge base) connection found in the project. "
            "Create the Foundry IQ knowledge base + connection first."
        )
    print(f"  Using knowledge-base connection: {connection_id}")
    definition = _agent_definition(settings, connection_id)
    version = client.agents.create_version(agent_name=name, definition=definition)
    print(f"✓ Created agent '{name}' version {getattr(version, 'version', '?')}.")
    agent = _find_agent(client, name)
    if agent:
        _describe(agent)
    print(
        "\nNEXT: grant the agent identity the 'Search Index Data Reader' role on the "
        "search service so it can read the knowledge base, e.g.:\n"
        "  az role assignment create --assignee <agent principal_id> \\\n"
        "    --role 'Search Index Data Reader' --scope <search service resource id>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify/create the SGS Foundry agent.")
    parser.add_argument("--create", action="store_true", help="Create the agent if missing.")
    args = parser.parse_args()
    provision(get_settings(), create=args.create)


if __name__ == "__main__":
    main()
