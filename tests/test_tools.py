import json
import re

from langchain_core.documents import Document
from pydantic import BaseModel

from src.tools import (tools_needing_confirmation, make_calendar_finder, make_rag_tool,
                       resolve_default, sync_mcp_tool)


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def search(self, queries):
        self.calls.append(queries)
        return self.docs


DOCS = [
    Document(page_content="solde 9570", metadata={"source": "/tmp/avis_2024.pdf", "page": 2}),
    Document(page_content="iban", metadata={"source": "/tmp/rib.pdf"}),
]


def call(tool_obj, queries):
    return tool_obj.invoke({"type": "tool_call", "id": "1",
                            "name": "rag_medhys_files", "args": {"queries": queries}})


def test_tool_tags_each_chunk_with_the_marker_of_its_artifact():
    message = call(make_rag_tool(FakeRetriever(DOCS)), ["avis imposition 2024"])
    first, second = message.artifact
    assert f"[{first['id']}] (avis_2024.pdf, page 2) solde 9570" in message.content
    assert f"[{second['id']}] (rib.pdf, page ?) iban" in message.content
    assert (first["path"], first["page"]) == ("/tmp/avis_2024.pdf", "2")
    assert (second["path"], second["page"]) == ("/tmp/rib.pdf", None)


def test_tool_markers_are_unique_across_calls():
    rag = make_rag_tool(FakeRetriever(DOCS))
    first = {info["id"] for info in call(rag, ["a"]).artifact}
    second = {info["id"] for info in call(rag, ["b"]).artifact}
    assert len(first) == 2 and not (first & second)


def test_tool_passes_all_queries_to_retriever():
    retriever = FakeRetriever(DOCS)
    call(make_rag_tool(retriever), ["a", "b"])
    assert retriever.calls == [["a", "b"]]


def test_tool_reports_empty_results():
    message = call(make_rag_tool(FakeRetriever([])), ["introuvable"])
    assert "Aucun extrait" in message.content
    assert message.artifact == []


def test_tool_exposes_name_description_and_schema():
    rag = make_rag_tool(FakeRetriever(DOCS))
    assert rag.name == "rag_medhys_files"
    assert "Myelink" in rag.description
    assert "queries" in rag.args


class EchoArgs(BaseModel):
    message: str


class FakeAsyncTool:
    name = "echo"
    description = "renvoie le message"
    args_schema = EchoArgs

    async def ainvoke(self, kwargs):
        return f"echo:{kwargs['message']}"


def test_sync_mcp_tool_wraps_async_invocation():
    wrapped = sync_mcp_tool(FakeAsyncTool(), "demo_echo")
    assert wrapped.name == "demo_echo"
    assert wrapped.description == "renvoie le message"
    assert wrapped.invoke({"message": "x"}) == "echo:x"


class FakeAsyncBlocksTool:
    name = "list_events"
    description = "liste"
    args_schema = EchoArgs

    def __init__(self, text):
        self.text = text

    async def ainvoke(self, kwargs):
        return [{"type": "text", "text": self.text}]


def test_sync_mcp_tool_json_contract_flags_non_json_results():
    from src.tools import TOOL_ERROR
    errored = sync_mcp_tool(FakeAsyncBlocksTool("The caller does not have permission"),
                            "t", json_result=True)
    assert TOOL_ERROR in str(errored.invoke({"message": "x"}))
    valid = sync_mcp_tool(FakeAsyncBlocksTool('{"events": []}'), "t", json_result=True)
    assert TOOL_ERROR not in str(valid.invoke({"message": "x"}))


S_PORTO = {"id": "e1", "summary": "S porto",
           "start": {"dateTime": "2026-07-25T13:30:00+02:00"},
           "end": {"dateTime": "2026-07-25T14:00:00+02:00"}}


def blocks(events, key="events"):
    return [{"type": "text", "text": json.dumps({key: events})}]


class FakeCalendarTool:
    def __init__(self, name, by_query=None, listing=None, key="events", error=None):
        self.name = name
        self.by_query = by_query or {}
        self.listing = listing or []
        self.key = key
        self.error = error
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        if self.error:
            return [{"type": "text", "text": self.error}]
        if "query" in args:
            return blocks(self.by_query.get(args["query"], []), self.key)
        return blocks(self.listing, self.key)


def make_account(name, by_query=None, listing=None, key="events", error=None):
    return [FakeCalendarTool(f"calendar_{name}_search_events", by_query=by_query,
                             key=key, error=error),
            FakeCalendarTool(f"calendar_{name}_list_events", listing=listing,
                             key=key, error=error)]


def test_finder_returns_none_without_calendar_tools():
    assert make_calendar_finder([FakeCalendarTool("autre_outil")]) is None


def test_finder_searches_every_account_with_each_term():
    pro = make_account("pro")
    perso = make_account("perso", by_query={"porto": [S_PORTO]})
    result = make_calendar_finder(pro + perso).invoke({"query": "Stefani Porto"})
    assert "[perso] S porto" in result
    assert {c["query"] for c in pro[0].calls} == {"stefani", "porto"}


def test_finder_falls_back_to_listing_on_literal_miss():
    pro = make_account("pro")
    perso = make_account("perso", listing=[S_PORTO])
    result = make_calendar_finder(pro + perso).invoke({"query": "Stefani Porto"})
    assert "[perso] S porto" in result
    assert pro[1].calls and perso[1].calls


def test_finder_reads_items_key_and_deduplicates():
    pro = make_account("pro", by_query={"porto": [S_PORTO], "stefani": [S_PORTO]},
                       key="items")
    result = make_calendar_finder(pro).invoke({"query": "Stefani Porto"})
    assert result.count("S porto") == 1


def test_finder_reports_empty():
    result = make_calendar_finder(make_account("pro")).invoke({"query": "Zzz Yyy"})
    assert "Aucun événement" in result


def test_finder_reports_tool_errors_instead_of_empty_agenda():
    from src.tools import TOOL_ERROR
    errored = make_account("pro", error="The caller does not have permission")
    result = make_calendar_finder(errored).invoke({"query": "rdv"})
    assert TOOL_ERROR in result
    assert "permission" in result
    assert "Aucun événement trouvé" not in result


def test_sync_mcp_tool_forces_fixed_args_over_model_values():
    wrapped = sync_mcp_tool(FakeAsyncTool(), "demo_echo", fixed_args={"message": "pinned"})
    assert wrapped.invoke({"message": "autre"}) == "echo:pinned"


class OptionalArgs(BaseModel):
    message: str | None = None


class FakeOptionalTool:
    name = "echo"
    description = "renvoie le message"
    args_schema = OptionalArgs

    async def ainvoke(self, kwargs):
        return f"echo:{kwargs.get('message')}"


def test_tools_needing_confirmation_uses_renamed_tools():
    config = {"mcp": [
        {"name": "calendar_pro", "confirm": ["create-event"]},
        {"name": "calendar_perso", "confirm": ["create-event"]},
        {"name": "readonly_server"},
    ]}
    assert tools_needing_confirmation(config) == ["calendar_pro_create_event",
                                            "calendar_perso_create_event"]


ISO_SECONDS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def test_resolve_default_expands_relative_times():
    now = resolve_default("@now")
    past = resolve_default("@now-30d")
    future = resolve_default("@now+365d")
    assert all(ISO_SECONDS.fullmatch(v) for v in (now, past, future))
    assert past < now < future


def test_resolve_default_leaves_other_values_untouched():
    assert resolve_default("primary") == "primary"
    assert resolve_default(7) == 7


def test_sync_mcp_tool_default_args_fill_missing_keys_only():
    # Mirrors the real case: the client schema allows omitting the key, the
    # server requires it - the default fills the gap without overriding
    wrapped = sync_mcp_tool(FakeOptionalTool(), "demo_echo",
                            default_args={"message": "défaut"})
    assert wrapped.invoke({}) == "echo:défaut"
    assert wrapped.invoke({"message": "choisi"}) == "echo:choisi"
