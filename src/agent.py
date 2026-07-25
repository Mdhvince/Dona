from datetime import date
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from src.prompt import CITATION_PROMPT, SYSTEM_PROMPT
from src.tools import make_rag_tool


class CitedSource(BaseModel):
    file: str = Field(description="Nom exact du fichier, tel qu'affiché dans la liste")
    page: int | str | None = Field(default=None, description="Page du document utilisé")


class CitedSources(BaseModel):
    """Structured output of the citation extraction; field descriptions are
    French prompt content. Strict schema on purpose: the extraction call has
    no competing tool-calling concern, and a lax union lets models cite as
    plain strings that validation would then drop."""
    sources: list[CitedSource] = Field(
        default_factory=list,
        description="Documents réellement utilisés par la réponse, vide si aucun")


@dynamic_prompt
def system_prompt(request: ModelRequest) -> str:
    """Resolved at every model call so the date never goes stale in a
    long-lived process."""
    return SYSTEM_PROMPT.format(date=date.today().strftime("%d/%m/%Y"))


def build_agent(retriever, llm, checkpointer=None, extra_tools=()):
    """Agent over the personal documents RAG plus any extra tools (MCP
    servers: calendar, gmail...). Passing a checkpointer enables multi-turn
    conversations (one thread_id per conversation). No response_format on
    purpose: a constrained output grammar competes with tool calling on some
    models; citations are extracted afterwards by extract_citations()."""
    return create_agent(
        model=llm,
        tools=[make_rag_tool(retriever), *extra_tools],
        middleware=[system_prompt],
        checkpointer=checkpointer)


def extract_citations(llm, answer_text, retrieved):
    """Post-hoc structured call, decoupled from the agent loop so it stays
    model-agnostic: given the final answer and the retrieved documents, the
    model ticks the ones the answer actually uses; validate_citations then
    drops anything that does not match a really retrieved document. Returns
    [] when there is nothing to cite or the extraction fails."""
    if not retrieved or not answer_text:
        return []
    documents = "\n".join(f"- {Path(info['path']).name}, page {info['page']}"
                          for info in retrieved)
    try:
        result = llm.with_structured_output(CitedSources).invoke(
            CITATION_PROMPT.format(answer=answer_text, documents=documents))
    except Exception:
        return []
    return validate_citations(result.sources, retrieved)


def current_turn(messages):
    """Messages of the ongoing exchange: everything from the last human
    message onward (the checkpointer brings the whole history back)."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages


def collect_sources(messages):
    """Deduplicated {path, page} of every document retrieved during the run,
    read from the tool message artifacts (never from the LLM output)."""
    sources, seen = [], set()
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        for info in message.artifact or []:
            key = (info.get("path"), info.get("page"))
            if info.get("path") and key not in seen:
                seen.add(key)
                sources.append(info)
    return sources


def validate_citations(cited, retrieved):
    """Keep only the citations matching a document actually retrieved during
    the run, matched on (file name, page); paths always come from the
    artifacts, never from the LLM. Order and deduplication follow the
    citations."""
    by_key = {(Path(info["path"]).name, info["page"]): info for info in retrieved}
    validated, seen = [], set()
    for citation in cited:
        if not isinstance(citation, CitedSource):
            continue
        key = (citation.file, None if citation.page is None else str(citation.page))
        info = by_key.get(key)
        if info and key not in seen:
            seen.add(key)
            validated.append(info)
    return validated


def collect_queries(messages):
    """Search queries the agent actually sent to the RAG tool, in order."""
    queries = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            queries.extend(call.get("args", {}).get("queries") or [])
    return queries
