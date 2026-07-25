import asyncio
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langchain_core.tools import StructuredTool, tool

from src.prompt import CALENDAR_FINDER_DESCRIPTION, RAG_TOOL_DESCRIPTION


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


def _calendar_events(result):
    """Events from a calendar MCP tool result (a list of content blocks
    whose text is JSON); empty list on anything unexpected."""
    try:
        if isinstance(result, list):
            result = result[0].get("text", "")
        return json.loads(result).get("events", [])
    except Exception:
        return []


def _event_line(account, event):
    start = event.get("start", {})
    end = event.get("end", {})
    return (f"- [{account}] {event.get('summary', '(sans titre)')} | "
            f"{start.get('dateTime', start.get('date', '?'))} -> "
            f"{end.get('dateTime', end.get('date', '?'))}")


def make_calendar_finder(mcp_tools, max_results=30):
    """Composite tool encoding the event-search strategy in code instead of
    relying on model discipline: search every account with each
    discriminating term, fall back to a wide listing when searches are
    literal misses, and hand the deduplicated candidates back to the model,
    whose only job is to recognize the right one. Returns None when no
    calendar account is available."""
    accounts = {}
    for mcp_tool in mcp_tools:
        match = re.fullmatch(r"calendar_(\w+?)_(search_events|list_events)", mcp_tool.name)
        if match:
            accounts.setdefault(match.group(1), {})[match.group(2)] = mcp_tool
    accounts = {name: tools for name, tools in accounts.items()
                if "search_events" in tools and "list_events" in tools}
    if not accounts:
        return None

    def iso_second(moment):
        """The calendar MCP server rejects fractional seconds."""
        return moment.strftime("%Y-%m-%dT%H:%M:%SZ")

    @tool("calendar_find_event", description=CALENDAR_FINDER_DESCRIPTION)
    def calendar_find_event(query: str) -> str:
        now = datetime.now(timezone.utc)
        window = {"calendarId": "primary",
                  "timeMin": iso_second(now - timedelta(days=30)),
                  "timeMax": iso_second(now + timedelta(days=365))}
        terms = re.findall(r"\w{3,}", query.lower()) or [query]

        candidates, seen = [], set()

        def collect(account, events):
            for event in events:
                key = event.get("id") or (event.get("summary"), str(event.get("start")))
                if key not in seen:
                    seen.add(key)
                    candidates.append((account, event))

        for account, tools in accounts.items():
            for term in terms:
                try:
                    collect(account, _calendar_events(
                        tools["search_events"].invoke({**window, "query": term})))
                except Exception:
                    continue
        if not candidates:
            listing = {**window, "timeMax": iso_second(now + timedelta(days=90))}
            for account, tools in accounts.items():
                try:
                    collect(account, _calendar_events(tools["list_events"].invoke(listing)))
                except Exception:
                    continue

        if not candidates:
            return ("Aucun événement trouvé sur la période (comptes : "
                    + ", ".join(accounts) + ").")
        lines = [_event_line(account, event) for account, event in candidates[:max_results]]
        if len(candidates) > max_results:
            lines.append(f"(+{len(candidates) - max_results} autres)")
        return "Événements candidats :\n" + "\n".join(lines)

    return calendar_find_event


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
