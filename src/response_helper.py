from pathlib import Path
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class LLMAnswer(BaseModel):
    """Structured LLM output: the source is only a chunk number, the link is
    rebuilt in code (no hallucinatable URI). Field descriptions are in French
    because they are part of the prompt sent to the model."""
    response: str = Field(
        description="La réponse à la question, en Markdown, sans mention de la source.")
    source: Optional[int] = Field(
        default=None,
        description="Numéro [i] de l'extrait du contexte utilisé pour répondre, null si aucun.")


class Answer(LLMAnswer):
    # Fields filled by code, never by the LLM
    source_readable: Optional[str] = None  # Markdown link, for the terminal
    source_path: Optional[str] = None      # raw path, for other interfaces (Flask...)
    source_page: Optional[str] = None


RAG_PROMPT = ChatPromptTemplate.from_messages([
      ("system",
       "Tu es l'assistant personnel de Medhy Vinceslas, un freelance Data Scientist qui a sa propre entreprise appelée Myelink."
       "Tu réponds uniquement à partir du contexte fourni. Si l'information "
       "n'y figure pas, dis-le clairement. Réponds en Markdown, de façon "
       "concise et structurée. Mets ta réponse dans le champ `response`, sans "
       "y citer la source. Si tu t'appuies sur un extrait, mets son numéro "
       "dans le champ `source`."),
      ("human", "Contexte :\n{context}\n\nQuestion : {question}"),
  ])


def format_context(docs):
    """Number each chunk so the LLM can reference it in `source`."""
    return "\n\n".join(
        f"[{i}] (page {d.metadata.get('page_label', d.metadata.get('page', '?'))}) {d.page_content}"
        for i, d in enumerate(docs, 1)
    )


def cited_source(docs, num):
    """(path, page) of chunk number `num` (1-indexed) from its metadata.
    (None, None) when the number is missing, out of bounds or has no source."""
    if num is None or not 1 <= num <= len(docs):
        return None, None
    d = docs[num - 1]
    source = d.metadata.get("source")
    if not source:
        return None, None
    page = d.metadata.get("page_label", d.metadata.get("page"))
    return source, None if page is None else str(page)


def source_link(path, page):
    """Clickable Markdown link for the terminal."""
    if not path:
        return None
    link = f"[{Path(path).name}]({Path(path).as_uri()})"
    return f"{link}, page {page}" if page is not None else link


def answer(question, retriever, llm_client):
    docs = retriever.invoke(question)
    chain = RAG_PROMPT | llm_client.with_structured_output(LLMAnswer)
    raw = chain.invoke({"context": format_context(docs), "question": question})
    path, page = cited_source(docs, raw.source)
    return Answer(**raw.model_dump(),
                  source_readable=source_link(path, page),
                  source_path=path,
                  source_page=page)
