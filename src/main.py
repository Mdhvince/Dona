import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_chroma import Chroma

from config import load_config
from llm_clients import llm, embedding_model
from response_helper import answer
from retrieval import HybridRetriever

PERSIST_DIR = str(Path(__file__).parent.parent / "vectordb")


if __name__ == "__main__":
    load_dotenv()
    config = load_config()
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not Path(PERSIST_DIR).exists():
        raise SystemExit("Base vectorielle introuvable — lance d'abord : python src/ingest.py")

    llm_client = llm("ollama", config["llm"]["model"], base_url=config["llm"]["base_url"])
    embedding_client = embedding_model(api_key,
                                       model_id=config["embedding"]["model"],
                                       base_url=config["embedding"]["base_url"])

    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_client)

    # Hybrid search: semantic (Chroma) + keyword (BM25), fused with Reciprocal Rank Fusion.
    hybrid_retriever = HybridRetriever.from_vectordb(vectordb, **config["retriever"])

    # MultiQueryRetriever generates multiple versions of the query to improve retrieval performance.
    retriever = MultiQueryRetriever.from_llm(retriever=hybrid_retriever, llm=llm_client, include_original=True)

    print(answer("Quel est le numero de siret de Myelink?", retriever, llm_client))
