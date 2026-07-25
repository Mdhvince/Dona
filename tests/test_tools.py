import json

from langchain_core.documents import Document
from pydantic import BaseModel

from src.tools import make_calendar_finder, make_rag_tool, sync_mcp_tool


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


def test_tool_returns_formatted_content_and_sources_artifact():
    message = call(make_rag_tool(FakeRetriever(DOCS)), ["avis imposition 2024"])
    assert "[1] (avis_2024.pdf, page 2) solde 9570" in message.content
    assert message.artifact == [{"path": "/tmp/avis_2024.pdf", "page": "2"},
                                {"path": "/tmp/rib.pdf", "page": None}]


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


class FakeListEvents:
    def __init__(self, name, by_fulltext=None, listing=None, key="events"):
        self.name = name
        self.by_fulltext = by_fulltext or {}
        self.listing = listing or []
        self.key = key
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        if "fullText" in args:
            return blocks(self.by_fulltext.get(args["fullText"], []), self.key)
        return blocks(self.listing, self.key)


def test_finder_returns_none_without_calendar_tools():
    assert make_calendar_finder([FakeListEvents("autre_outil")]) is None


def test_finder_searches_every_account_with_each_term():
    pro = FakeListEvents("calendar_pro_list_events")
    perso = FakeListEvents("calendar_perso_list_events", by_fulltext={"porto": [S_PORTO]})
    result = make_calendar_finder([pro, perso]).invoke({"query": "Stefani Porto"})
    assert "[perso] S porto" in result
    assert {c["fullText"] for c in pro.calls} == {"stefani", "porto"}


def test_finder_falls_back_to_listing_on_literal_miss():
    pro = FakeListEvents("calendar_pro_list_events")
    perso = FakeListEvents("calendar_perso_list_events", listing=[S_PORTO])
    result = make_calendar_finder([pro, perso]).invoke({"query": "Stefani Porto"})
    assert "[perso] S porto" in result
    assert any("fullText" not in c for c in pro.calls)


def test_finder_reads_items_key_and_deduplicates():
    pro = FakeListEvents("calendar_pro_list_events",
                         by_fulltext={"porto": [S_PORTO], "stefani": [S_PORTO]}, key="items")
    result = make_calendar_finder([pro]).invoke({"query": "Stefani Porto"})
    assert result.count("S porto") == 1


def test_finder_reports_empty():
    pro = FakeListEvents("calendar_pro_list_events")
    result = make_calendar_finder([pro]).invoke({"query": "Zzz Yyy"})
    assert "Aucun événement" in result


class ErrorListEvents:
    name = "calendar_pro_list_events"

    def invoke(self, args):
        return [{"type": "text", "text": "The caller does not have permission"}]


def test_finder_reports_tool_errors_instead_of_empty_agenda():
    from src.tools import TOOL_ERROR
    result = make_calendar_finder([ErrorListEvents()]).invoke({"query": "rdv"})
    assert TOOL_ERROR in result
    assert "permission" in result
    assert "Aucun événement trouvé" not in result
