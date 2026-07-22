import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_chroma import Chroma

from llm_clients import llm, embedding_model
from response_helper import answer
from retrieval import HybridRetriever

PERSIST_DIR = str(Path(__file__).parent.parent / "vectordb")

# Paramètres BM25 (keyword search)
BM25_K1 = 1.5  # saturation de la fréquence des termes [1.2 - 2.0]
BM25_B = 0.75  # normalisation par la longueur du document [0.0 - 1.0]


if __name__ == "__main__":
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    model_id = "gpt-oss:20b"

    if not Path(PERSIST_DIR).exists():
        raise SystemExit("Base vectorielle introuvable — lance d'abord : python src/ingest.py")

    llm_client = llm("ollama", model_id, base_url="http://localhost:11434/v1")
    embedding_client = embedding_model(api_key)

    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_client)

    # Hybrid search: semantic (Chroma) + keyword (BM25), fused with Reciprocal Rank Fusion.
    hybrid_retriever = HybridRetriever.from_vectordb(vectordb, k1=BM25_K1, b=BM25_B)

    # MultiQueryRetriever generates multiple versions of the query to improve retrieval performance.
    retriever = MultiQueryRetriever.from_llm(retriever=hybrid_retriever, llm=llm_client)

    print(answer("Quel est le numero de SIRET de Myelink?", retriever, llm_client))
