import re

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class HybridRetriever(BaseRetriever):
    """
    Recherche hybride : sémantique (Chroma) + keyword (BM25), fusionnées par
    Reciprocal Rank Fusion. Chaque document reçoit un score 1/(rrf_k + rang)
    dans chaque classement, et les scores s'additionnent — un document bien
    classé par les deux recherches remonte en tête.

    Paramètres BM25 :
    - k1 : saturation de la fréquence des termes (typiquement 1.2 à 2.0)
    - b  : normalisation par la longueur du document (0.0 à 1.0)
    """

    vectordb: object
    documents: list
    bm25: object
    k: int = 4
    rrf_k: int = 60

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_vectordb(cls, vectordb, k1=1.5, b=0.75, k=4):
        """Construit l'index BM25 depuis les chunks déjà stockés dans Chroma."""
        data = vectordb.get()
        documents = [Document(page_content=text, metadata=meta)
                     for text, meta in zip(data["documents"], data["metadatas"])]
        bm25 = BM25Okapi([tokenize(d.page_content) for d in documents], k1=k1, b=b)
        return cls(vectordb=vectordb, documents=documents, bm25=bm25, k=k)

    def _keyword_search(self, query):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.documents[i] for i in ranked[:self.k] if scores[i] > 0]

    def _get_relevant_documents(self, query, *, run_manager=None):
        rankings = [
            self.vectordb.similarity_search(query, k=self.k),
            self._keyword_search(query),
        ]
        scores, docs_by_key = {}, {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking):
                key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content)
                docs_by_key[key] = doc
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        fused = sorted(scores, key=scores.get, reverse=True)
        return [docs_by_key[key] for key in fused[:self.k]]
