from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class LLMAnswer(BaseModel):
    """Structured LLM output: sources are only chunk numbers, the links are
    rebuilt in code (no hallucinatable URI). Field descriptions are in French
    because they are part of the prompt sent to the model."""
    response: str = Field(
        description="La réponse à la question, en Markdown, sans mention des sources.")
    sources: list[int] = Field(
        default_factory=list,
        description="Numéros [i] des extraits du contexte utilisés pour répondre, liste vide si aucun.")


class Answer(LLMAnswer):
    # Fields filled by code, never by the LLM
    source_readable: str | None = None  # Markdown links, for the terminal
    sources_info: list = []             # [{"path", "page"}], for other interfaces (Flask...)


RAG_PROMPT = ChatPromptTemplate.from_messages([
      ("system",
       "Tu es mon assistant personnel. Je m'appelle Medhy Vinceslas, je suis "
       "freelance Data Scientist et j'ai ma propre entreprise appelée Myelink. "
       "Si je demande quelque chose sans mentionner un nom, ou en utilisant un "
       "déterminant possessif, tu dois comprendre que je parle de moi-même ou "
       "de mon entreprise selon la question. "
       "Tu réponds uniquement à partir du contexte fourni. Si l'information "
       "n'y figure pas, dis-le clairement. "
       "Réponds en Markdown, de façon concise et structurée. Mets ta réponse "
       "dans le champ `response`, sans y citer les sources. Mets les numéros "
       "des extraits sur lesquels tu t'appuies dans le champ `sources`."),
      ("human", "Contexte :\n{context}\n\nQuestion : {question}"),
  ])


def format_context(docs):
    """Number each chunk so the LLM can reference it in `sources`, and name
    the file so it can tell documents (and years) apart."""
    lines = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source")
        name = Path(source).name if source else "?"
        page = d.metadata.get("page_label", d.metadata.get("page", "?"))
        lines.append(f"[{i}] ({name}, page {page}) {d.page_content}")
    return "\n\n".join(lines)


def cited_sources(docs, nums):
    """Deduplicated [(path, page)] for the chunk numbers cited by the LLM
    (1-indexed). Out-of-bounds numbers and chunks without a source are
    skipped."""
    cited, seen = [], set()
    for num in nums or []:
        if not 1 <= num <= len(docs):
            continue
        d = docs[num - 1]
        source = d.metadata.get("source")
        if not source:
            continue
        page = d.metadata.get("page_label", d.metadata.get("page"))
        key = (source, page)
        if key not in seen:
            seen.add(key)
            cited.append((source, None if page is None else str(page)))
    return cited


def source_links(sources):
    """Clickable Markdown links for the terminal, one per cited source."""
    if not sources:
        return None
    links = []
    for path, page in sources:
        link = f"[{Path(path).name}]({Path(path).as_uri()})"
        links.append(f"{link}, page {page}" if page is not None else link)
    return " ; ".join(links)


def answer(question, retriever, llm_client):
    docs = retriever.invoke(question)
    chain = RAG_PROMPT | llm_client.with_structured_output(LLMAnswer)
    raw = chain.invoke({"context": format_context(docs), "question": question})
    cited = cited_sources(docs, raw.sources)
    return Answer(**raw.model_dump(),
                  source_readable=source_links(cited),
                  sources_info=[{"path": path, "page": page} for path, page in cited])
