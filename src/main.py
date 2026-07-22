from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_chroma import Chroma

import response_helper
from config import load_config, llm_client, embedding_client
from response_helper import answer
from retrieval import HybridRetriever

PERSIST_DIR = str(Path(__file__).parent.parent / "vectordb")


if __name__ == "__main__":
    load_dotenv()
    config = load_config()

    if not Path(PERSIST_DIR).exists():
        raise SystemExit("Base vectorielle introuvable - lance d'abord : python src/ingest.py")

    chat_llm = llm_client(config)
    vectordb = Chroma(persist_directory=PERSIST_DIR,
                      embedding_function=embedding_client(config))

    # Hybrid search: semantic (Chroma) + keyword (BM25), fused with Reciprocal Rank Fusion.
    hybrid_retriever = HybridRetriever.from_vectordb(vectordb, **config["retriever"])

    # MultiQueryRetriever generates multiple versions of the query to improve retrieval performance.
    retriever = MultiQueryRetriever.from_llm(retriever=hybrid_retriever, llm=chat_llm, include_original=True)

    ans = answer("Quel est le numero de siret de Myelink?", retriever, chat_llm)
    response = ans.response
    source = ans.source_readable or ""
    print(f"{response}\n{source}")
