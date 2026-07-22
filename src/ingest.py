import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pypdf import PdfReader

from config import load_config
from document_processing import text_splitter
from llm_clients import embedding_model

HERE = Path(__file__).parent
PERSIST_DIR = str(HERE.parent / "vectordb")

# Racines à indexer : montages Google Drive (sous-dossiers inclus)
DOCS_DIRS = [
    Path("/Users/medhyvinceslas/Library/CloudStorage/GoogleDrive-mvinceslas@myelink.io/Mon Drive"),
    Path("/Users/medhyvinceslas/Library/CloudStorage/GoogleDrive-medhy.vinceslas@gmail.com/Mon Drive"),
]


def load_pdf(path):
    return [Document(page_content=page.extract_text() or "",
                     metadata={"source": str(path), "page": i})
            for i, page in enumerate(PdfReader(str(path)).pages, start=1)]


def load_text(path):
    return [Document(page_content=path.read_text(encoding="utf-8"),
                     metadata={"source": str(path)})]


LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_text,
    ".md": load_text,
}


def load_documents(docs_dirs):
    documents = []
    for root in docs_dirs:
        if not root.exists():
            print(f"⚠ racine introuvable, ignorée : {root}")
            continue
        for path in sorted(root.rglob("*")):
            loader = LOADERS.get(path.suffix.lower())
            if loader is None:
                continue
            try:
                documents.extend(loader(path))
                print(f"  chargé : {path.relative_to(root)}")
            except Exception as exc:
                print(f"⚠ échec sur {path.name} : {exc}")
    return documents


def ingest(docs_dirs, embedding_client, persist_directory, chunk_size, chunk_overlap):
    shutil.rmtree(persist_directory, ignore_errors=True)

    documents = load_documents(docs_dirs)
    chunks = text_splitter(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    Chroma.from_documents(documents=chunks,
                          embedding=embedding_client,
                          persist_directory=persist_directory)
    print(f"{len(chunks)} chunks indexés depuis {len(documents)} pages/documents")


if __name__ == "__main__":
    load_dotenv()
    config = load_config()
    api_key = os.environ.get("OPENROUTER_API_KEY")

    embedding_client = embedding_model(api_key,
                                       model_id=config["embedding"]["model"],
                                       base_url=config["embedding"]["base_url"])
    ingest(DOCS_DIRS, embedding_client, PERSIST_DIR,
           chunk_size=config["ingestion"]["chunk_size"],
           chunk_overlap=config["ingestion"]["chunk_overlap"])
