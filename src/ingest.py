import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pypdf import PdfReader

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
            for i, page in enumerate(PdfReader(str(path)).pages)]


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


def ingest(docs_dirs, embedding_client, persist_directory):
    shutil.rmtree(persist_directory, ignore_errors=True)

    documents = load_documents(docs_dirs)
    chunks = text_splitter(documents, chunk_size=500, chunk_overlap=20)
    Chroma.from_documents(documents=chunks,
                          embedding=embedding_client,
                          persist_directory=persist_directory)
    print(f"{len(chunks)} chunks indexés depuis {len(documents)} pages/documents")


if __name__ == "__main__":
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")

    embedding_client = embedding_model(api_key)
    ingest(DOCS_DIRS, embedding_client, PERSIST_DIR)
