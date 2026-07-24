from langchain_core.documents import Document
from pydantic import BaseModel

from src.tools import make_rag_tool, sync_mcp_tool


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


def test_sync_mcp_tool_forces_fixed_args_over_model_values():
    wrapped = sync_mcp_tool(FakeAsyncTool(), "demo_echo", {"message": "pinned"})
    assert wrapped.invoke({"message": "autre"}) == "echo:pinned"
