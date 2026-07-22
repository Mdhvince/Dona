from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


RAG_PROMPT = ChatPromptTemplate.from_messages([
      ("system",
       "Tu es l'assistant personnel de Medhy Vinceslas, un freelance Data Scientist qui a sa propre entreprise appelée Myelink."
       "Tu réponds uniquement à partir du contexte fourni. Si l'information "
       "n'y figure pas, dis-le clairement. Réponds en Markdown, de façon "
       "concise et structurée. Quand tu t'appuies sur un extrait, termine la "
       "phrase par son numéro, par exemple [1]."),
      ("human", "Contexte :\n{context}\n\nQuestion : {question}"),
  ])


def format_context(docs):
    """Numérote chaque chunk pour que le LLM puisse le citer."""
    return "\n\n".join(
        f"[{i}] (page {d.metadata.get('page_label', d.metadata.get('page', '?'))}) {d.page_content}"
        for i, d in enumerate(docs, 1)
    )


def format_sources(docs):
    """Construit la liste des sources depuis les métadonnées."""
    lines = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source")
        fichier = f"[{Path(source).name}]({Path(source).as_uri()})" if source else "?"
        page = d.metadata.get("page_label", d.metadata.get("page", "?"))
        extrait = d.page_content[:70].replace("\n", " ")
        lines.append(f"[{i}] {fichier}, page {page} — « {extrait}… »")
    return "\n".join(lines)


def answer(question, retriever, llm_client):
    docs = retriever.invoke(question)
    chain = RAG_PROMPT | llm_client | StrOutputParser()
    reponse = chain.invoke({"context": format_context(docs), "question": question})
    return f"{reponse}\n\n---\n**Sources :**\n{format_sources(docs)}"
