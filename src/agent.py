from datetime import date
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from src.prompt import SYSTEM_PROMPT
from src.tools import make_rag_tool


class CitedSource(BaseModel):
    file: str = Field(description="Nom exact du fichier, tel qu'affiché dans l'extrait")
    page: int | str | None = Field(default=None, description="Page de l'extrait utilisé")


class AgentAnswer(BaseModel):
    """Final structured output; field descriptions are French prompt content."""
    response: str = Field(description="La réponse en Markdown, sans mention des sources")
    sources: list[CitedSource] = Field(
        default_factory=list,
        description="Extraits réellement utilisés pour répondre, vide si rien trouvé")


@dynamic_prompt
def system_prompt(request: ModelRequest) -> str:
    """Resolved at every model call so the date never goes stale in a
    long-lived process."""
    return SYSTEM_PROMPT.format(date=date.today().strftime("%d/%m/%Y"))


def build_agent(retriever, llm):
    """Single-tool agent over the personal documents RAG: the model decides
    when to search and with which queries."""
    return create_agent(
        model=llm,
        tools=[make_rag_tool(retriever)],
        middleware=[system_prompt],
        response_format=ProviderStrategy(AgentAnswer))


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
