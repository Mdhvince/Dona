import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pypdf import PdfReader

from config import load_config, embedding_client
from document_processing import text_splitter

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
            # Dossiers privés : tout chemin traversant un dossier "_..." est exclu
            if any(part.startswith("_") for part in path.relative_to(root).parts[:-1]):
                continue
            try:
                docs = loader(path)
                # Un tag par niveau de dossier sous la racine, filtrable dans Chroma :
                # "05 - Clients/Techplaces/x.pdf" -> tag_1="05 - Clients", tag_2="Techplaces"
                tags = {f"tag_{i}": name
                        for i, name in enumerate(path.relative_to(root).parts[:-1], 1)}
                for doc in docs:
                    doc.metadata.update(tags)
                documents.extend(docs)
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

    ingest(DOCS_DIRS, embedding_client(config, api_key), PERSIST_DIR,
           chunk_size=config["ingestion"]["chunk_size"],
           chunk_overlap=config["ingestion"]["chunk_overlap"])
