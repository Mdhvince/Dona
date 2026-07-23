from pathlib import Path

from langchain_chroma import Chroma

from config import load_config, llm_client, rewriter_client, embedding_client
from response_helper import answer
from retrieval import HybridRetriever

PERSIST_DIR = str(Path(__file__).parent.parent / "vectordb")


if __name__ == "__main__":
    config = load_config()

    if not Path(PERSIST_DIR).exists():
        raise SystemExit("Base vectorielle introuvable - lance d'abord : python src/ingest.py")

    chat_llm = llm_client(config)
    rewriter_llm = rewriter_client(config)
    vectordb = Chroma(persist_directory=PERSIST_DIR,
                      embedding_function=embedding_client(config))

    # Hybrid search: semantic (Chroma) + keyword (BM25), fused with Reciprocal Rank Fusion.
    retriever = HybridRetriever.from_vectordb(vectordb, **config["retriever"])

    ans = answer("Quel est le numero de siret de Myelink?", retriever, chat_llm, rewriter_llm)
    response = ans.response
    source = ans.source_readable or ""
    print(f"{response}\n{source}")
