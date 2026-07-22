from pathlib import Path
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class LLMAnswer(BaseModel):
    """Sortie structurée du LLM : la source n'est qu'un numéro d'extrait,
    le lien est reconstruit en dur côté code (pas d'URI hallucinable)."""
    response: str = Field(
        description="La réponse à la question, en Markdown, sans mention de la source.")
    source: Optional[int] = Field(
        default=None,
        description="Numéro [i] de l'extrait du contexte utilisé pour répondre, null si aucun.")


class Answer(LLMAnswer):
    # Champs remplis par le code, jamais par le LLM
    source_readable: Optional[str] = None  # lien Markdown, pour le terminal
    source_path: Optional[str] = None      # chemin brut, pour les autres interfaces (Flask...)
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
    """Numérote chaque chunk pour que le LLM puisse le référencer dans `source`."""
    return "\n\n".join(
        f"[{i}] (page {d.metadata.get('page_label', d.metadata.get('page', '?'))}) {d.page_content}"
        for i, d in enumerate(docs, 1)
    )


def cited_source(docs, num):
    """(chemin, page) de l'extrait n° `num` (1-indexé) depuis ses métadonnées.
    (None, None) si le numéro est absent, hors bornes ou sans source."""
    if num is None or not 1 <= num <= len(docs):
        return None, None
    d = docs[num - 1]
    source = d.metadata.get("source")
    if not source:
        return None, None
    page = d.metadata.get("page_label", d.metadata.get("page"))
    return source, None if page is None else str(page)


def source_link(path, page):
    """Lien Markdown cliquable pour le terminal."""
    if not path:
        return None
    lien = f"[{Path(path).name}]({Path(path).as_uri()})"
    return f"{lien}, page {page}" if page is not None else lien


def answer(question, retriever, llm_client):
    docs = retriever.invoke(question)
    chain = RAG_PROMPT | llm_client.with_structured_output(LLMAnswer)
    raw = chain.invoke({"context": format_context(docs), "question": question})
    path, page = cited_source(docs, raw.source)
    return Answer(**raw.model_dump(),
                  source_readable=source_link(path, page),
                  source_path=path,
                  source_page=page)
