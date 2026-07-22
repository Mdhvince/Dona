import re
import unicodedata

import nltk
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from rank_bm25 import BM25Okapi

try:
    _STOPWORDS_FR = stopwords.words("french")
except LookupError:
    nltk.download("stopwords", quiet=True)
    _STOPWORDS_FR = stopwords.words("french")


def strip_accents(text):
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


STOPWORDS = {strip_accents(w) for w in _STOPWORDS_FR}
STEMMER = SnowballStemmer("french")


def tokenize(text):
    """Normalize so the question and the documents meet: lowercase without
    accents ("numero" matches "numéro"), stopwords removed (BM25 focuses on
    discriminating terms), and stemmed ("création" and "créer" share the
    root "cre")."""
    tokens = re.findall(r"\w+", strip_accents(text.lower()))
    return [STEMMER.stem(t) for t in tokens if t not in STOPWORDS]


class HybridRetriever(BaseRetriever):
    """
    Hybrid search: semantic (Chroma) + keyword (BM25), fused with Reciprocal
    Rank Fusion. Each document gets a 1/(rrf_k + rank) score in each ranking
    and the scores add up: a document ranked well by both searches rises to
    the top.

    BM25 parameters:
    - k1: term frequency saturation (typically 1.2 to 2.0)
    - b:  document length normalization (0.0 to 1.0)
    """

    vectordb: object
    documents: list
    bm25: object
    k: int = 4         # number of documents returned after fusion
    fetch_k: int = 10  # depth of each ranking before fusion
    rrf_k: int = 60

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_vectordb(cls, vectordb, k1=1.5, b=0.75, k=4, fetch_k=10, rrf_k=60):
        """Build the BM25 index from the chunks already stored in Chroma."""
        data = vectordb.get()
        documents = [Document(page_content=text, metadata=meta)
                     for text, meta in zip(data["documents"], data["metadatas"])]
        bm25 = BM25Okapi([tokenize(d.page_content) for d in documents], k1=k1, b=b)
        return cls(vectordb=vectordb, documents=documents, bm25=bm25,
                   k=k, fetch_k=fetch_k, rrf_k=rrf_k)

    def _keyword_search(self, query):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.documents[i] for i in ranked[:self.fetch_k] if scores[i] > 0]

    def _get_relevant_documents(self, query, *, run_manager=None):
        rankings = [
            self.vectordb.similarity_search(query, k=self.fetch_k),
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
