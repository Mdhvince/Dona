import asyncio
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from langchain_core.tools import StructuredTool, tool

from src.prompt import CALENDAR_FINDER_DESCRIPTION, RAG_TOOL_DESCRIPTION

# Sentinel emitted by the tool layer when a call fails; scanned in code to
# set the structured error status of the answer (never narrated by the LLM)
TOOL_ERROR = "[TOOL_ERROR]"


def format_context(docs, chunk_ids):
    """Tag each chunk with the marker the agent cites it by, and name its
    file so it can tell documents (and years) apart."""
    lines = []
    for chunk_id, d in zip(chunk_ids, docs):
        source = d.metadata.get("source")
        name = Path(source).name if source else "?"
        page = d.metadata.get("page_label", d.metadata.get("page", "?"))
        lines.append(f"[{chunk_id}] ({name}, page {page}) {d.page_content}")
    return "\n\n".join(lines)


def _source_of(doc, chunk_id):
    page = doc.metadata.get("page_label", doc.metadata.get("page"))
    return {"id": chunk_id,
            "path": doc.metadata.get("source"),
            "page": None if page is None else str(page)}


_RELATIVE_TIME = re.compile(r"@now(?:([+-])(\d+)([dh]))?$")


def resolve_default(value):
    """Config defaults are static, but time windows are not: "@now-30d" and
    "@now+365d" resolve at call time to ISO-8601 seconds (the calendar
    server rejects fractional seconds)."""
    if not isinstance(value, str):
        return value
    match = _RELATIVE_TIME.fullmatch(value)
    if not match:
        return value
    moment = datetime.now(timezone.utc)
    sign, amount, unit = match.groups()
    if amount:
        delta = timedelta(days=int(amount)) if unit == "d" else timedelta(hours=int(amount))
        moment = moment - delta if sign == "-" else moment + delta
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def sync_mcp_tool(mcp_tool, name, description=None, json_result=False,
                  fixed_args=None, default_args=None):
    """MCP adapter tools are async-only; wrap them so the sync agent and
    Flask stack can call them (one short-lived event loop per call, the
    adapter opens a fresh MCP session each time). json_result declares that
    the server answers JSON payloads: any non-JSON result is then an error
    surfaced as content and gets the TOOL_ERROR sentinel, so failures are
    detected in code. fixed_args are forced over whatever the model passes
    (values the model must not control: the account of a multi-account
    server); default_args only fill keys the model omitted (required params
    with an obvious value, like calendarId="primary")."""
    def run(**kwargs):
        for key, value in (default_args or {}).items():
            if kwargs.get(key) is None:
                kwargs[key] = resolve_default(value)
        kwargs.update(fixed_args or {})
        try:
            result = asyncio.run(mcp_tool.ainvoke(kwargs))
        except Exception as exc:
            return f"{TOOL_ERROR} {str(exc)[:200]}"
        if json_result:
            try:
                json.loads(_result_text(result))
            except Exception:
                return f"{TOOL_ERROR} {_result_text(result)[:200]}"
        return result

    return StructuredTool.from_function(
        func=run, name=name, description=description or mcp_tool.description,
        args_schema=mcp_tool.args_schema)


def confirmed_tool_names(config):
    """Renamed names of the tools a server declares as side-effecting: the
    agent may only run them after an explicit confirmation."""
    names = []
    for server in config.get("mcp", []):
        for tool_name in server.get("confirm", []):
            names.append(f"{server['name']}_{tool_name}".replace("-", "_"))
    return names


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
                                  f"{mcp_tool.description} {suffix}".strip() if suffix else None,
                                  server.get("json_result", False),
                                  server.get("fixed_args"),
                                  server.get("default_args"))
                    for mcp_tool in loaded
                    if not allowed or mcp_tool.name in allowed]
        tools.extend(selected)
        print(f"MCP {server['name']} : {len(selected)} outil(s) chargé(s)")
    return tools


def _result_text(result):
    if isinstance(result, list) and result:
        return result[0].get("text", "")
    return str(result)


def _calendar_events(result):
    """Events from a calendar MCP tool result. None when the payload is not
    an events response (tool error surfaced as plain text, unexpected JSON):
    an error must never be mistaken for an empty agenda."""
    try:
        data = json.loads(_result_text(result))
    except Exception:
        return None
    for key in ("events", "items"):
        if key in data:
            return data[key]
    return None


def _event_line(account, event):
    start = event.get("start", {})
    end = event.get("end", {})
    return (f"- [{account}] {event.get('summary', '(sans titre)')} | "
            f"{start.get('dateTime', start.get('date', '?'))} -> "
            f"{end.get('dateTime', end.get('date', '?'))}")


def make_calendar_finder(mcp_tools, max_results=30):
    """Composite tool encoding the event-search strategy in code instead of
    relying on model discipline: full-text search on every account with each
    discriminating term, fall back to a wide listing when searches are
    literal misses, and hand the deduplicated candidates back to the model,
    whose only job is to recognize the right one. Returns None when no
    calendar account is available."""
    accounts = {}
    for mcp_tool in mcp_tools:
        match = re.fullmatch(r"calendar_(\w+?)_(search_events|list_events)", mcp_tool.name)
        if match:
            accounts.setdefault(match.group(1), {})[match.group(2)] = mcp_tool
    accounts = {name: pair for name, pair in accounts.items()
                if "search_events" in pair and "list_events" in pair}
    if not accounts:
        return None

    def iso_second(moment):
        return moment.strftime("%Y-%m-%dT%H:%M:%SZ")

    @tool("calendar_find_event", description=CALENDAR_FINDER_DESCRIPTION)
    def calendar_find_event(query: str) -> str:
        now = datetime.now(timezone.utc)
        window = {"calendarId": "primary",
                  "timeMin": iso_second(now - timedelta(days=30)),
                  "timeMax": iso_second(now + timedelta(days=365))}
        terms = re.findall(r"\w{3,}", query.lower()) or [query]

        candidates, seen, errors = [], set(), {}

        def collect(account, events):
            for event in events:
                key = event.get("id") or (event.get("summary"), str(event.get("start")))
                if key not in seen:
                    seen.add(key)
                    candidates.append((account, event))

        def gather(account, tool_, args):
            try:
                result = tool_.invoke(args)
            except Exception as exc:
                errors[account] = str(exc)[:150]
                return
            events = _calendar_events(result)
            if events is None:
                errors[account] = _result_text(result)[:150]
                return
            collect(account, events)

        for account, pair in accounts.items():
            for term in terms:
                gather(account, pair["search_events"], {**window, "query": term})
        if not candidates:
            listing = {**window, "timeMax": iso_second(now + timedelta(days=90))}
            for account, pair in accounts.items():
                gather(account, pair["list_events"], listing)

        if not candidates and errors:
            return (f"{TOOL_ERROR} accès aux calendriers impossible :\n"
                    + "\n".join(f"- {account} : {message}"
                                for account, message in errors.items()))
        if not candidates:
            return ("Aucun événement trouvé sur la période (comptes : "
                    + ", ".join(accounts) + ").")
        if errors:
            candidates_note = ", ".join(errors)
            lines = [_event_line(account, event) for account, event in candidates[:max_results]]
            return ("Événements candidats (accès en erreur pour : "
                    f"{candidates_note}) :\n" + "\n".join(lines))
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
        # Random ids rather than 1..n: markers must stay unique across the
        # several tool calls of a single turn for citations to resolve
        chunk_ids = [uuid4().hex[:4] for _ in docs]
        return (format_context(docs, chunk_ids),
                [_source_of(doc, chunk_id) for doc, chunk_id in zip(docs, chunk_ids)])

    return rag_medhys_files
