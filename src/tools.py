from pathlib import Path

from langchain_core.tools import tool

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
