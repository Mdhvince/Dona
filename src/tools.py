import asyncio
import os
from pathlib import Path

from langchain_core.tools import StructuredTool, tool

from src.prompt import RAG_TOOL_DESCRIPTION


def format_context(docs):
    """Number each chunk and name its file so the agent can tell documents
    (and years) apart."""
    lines = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source")
        name = Path(source).name if source else "?"
        page = d.metadata.get("page_label", d.metadata.get("page", "?"))
        lines.append(f"[{i}] ({name}, page {page}) {d.page_content}")
    return "\n\n".join(lines)


def _source_of(doc):
    page = doc.metadata.get("page_label", doc.metadata.get("page"))
    return {"path": doc.metadata.get("source"),
            "page": None if page is None else str(page)}


def sync_mcp_tool(mcp_tool, name, fixed_args=None, description=None):
    """MCP adapter tools are async-only; wrap them so the sync agent and
    Flask stack can call them (one short-lived event loop per call, the
    adapter opens a fresh MCP session each time). fixed_args are forced over
    whatever the model passes: they pin per-instance values the model must
    not control (the account of a multi-account server, typically)."""
    def run(**kwargs):
        kwargs.update(fixed_args or {})
        return asyncio.run(mcp_tool.ainvoke(kwargs))

    return StructuredTool.from_function(
        func=run, name=name, description=description or mcp_tool.description,
        args_schema=mcp_tool.args_schema)


def load_mcp_tools(config):
    """Connect every [[mcp]] server from config.toml and return its
    whitelisted tools, renamed <server>_<tool> so the same server can run
    once per account (calendar_pro, calendar_perso...). A server that fails
    to load is skipped with a warning: the agent must keep working without
    it (OAuth not done yet, server down...)."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    tools = []
    for server in config.get("mcp", []):
        client = MultiServerMCPClient({server["name"]: {
            "transport": "stdio",
            "command": server["command"],
            "args": server.get("args", []),
            "env": {**os.environ, **server.get("env", {})}}})
        try:
            loaded = asyncio.run(client.get_tools())
        except Exception as exc:
            print(f"⚠ MCP {server['name']} indisponible : {exc}")
            continue
        allowed = server.get("tools")
        if allowed:
            for missing in sorted(set(allowed) - {t.name for t in loaded}):
                print(f"⚠ MCP {server['name']} : outil \"{missing}\" absent du serveur "
                      f"(disponibles : {sorted(t.name for t in loaded)})")
        suffix = server.get("description_suffix", "")
        selected = [sync_mcp_tool(mcp_tool, f"{server['name']}_{mcp_tool.name}".replace("-", "_"),
                                  server.get("fixed_args"),
                                  f"{mcp_tool.description} {suffix}".strip() if suffix else None)
                    for mcp_tool in loaded
                    if not allowed or mcp_tool.name in allowed]
        tools.extend(selected)
        print(f"MCP {server['name']} : {len(selected)} outil(s) chargé(s)")
    return tools


def make_rag_tool(retriever):
    """Builds the RAG tool around an already constructed retriever, so the
    tool schema only exposes what the agent must decide: the queries."""

    @tool("rag_medhys_files", description=RAG_TOOL_DESCRIPTION,
          response_format="content_and_artifact")
    def rag_medhys_files(queries: list[str]) -> tuple[str, list]:
        docs = retriever.search(queries)
        if not docs:
            return "Aucun extrait trouvé pour ces requêtes.", []
        return format_context(docs), [_source_of(doc) for doc in docs]

    return rag_medhys_files
