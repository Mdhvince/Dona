import base64
import io
from pathlib import Path

import pypdfium2 as pdfium
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from src.config import load_config, docs_dirs, embedding_client, vlm_client
from src.document_processing import markdown_splitter
from src.prompt import IMAGE_TRANSCRIPTION_PROMPT, PDF_TRANSCRIPTION_PROMPT

PERSIST_DIR = str(Path(__file__).parent.parent / "vectordb")

TEXT_SUFFIXES = {".txt", ".md"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
RENDER_DPI = 150


def llm_bases_img2text(vlm, image_bytes, mime, prompt):
    """
    This function converts an image to text using a vision language model (VLM).
    :param vlm: The vision language model to use for the conversion.
    :param image_bytes: The image data in bytes format.
    :param mime: The MIME type of the image (e.g., "image/png", "image/jpeg").
    :param prompt: The prompt to guide the VLM in the conversion process.
    :return: The text output from the VLM after processing the image.
    """
    b64 = base64.b64encode(image_bytes).decode()
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ])
    return vlm.invoke([message]).content


def pdf2png(page, dpi=RENDER_DPI):
    """
    Render a PDF page to PNG bytes at the given resolution
    :param page: The PDF page to render.
    :param dpi: The resolution in dots per inch (default is 150).
    :return: The rendered PNG image in bytes format.
    """
    image = page.render(scale=dpi / 72).to_pil()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def load_pdf(path, vlm):
    """
    Render each page to an image and let the vision model transcribe it to Markdown.
    :param path: The path to the PDF file.
    :param vlm: The vision language model to use for transcription.
    :return: A list of Document objects
    """
    pdf = pdfium.PdfDocument(str(path))
    documents = []
    for page_index in range(len(pdf)):
        print(f"\t\t{path.name} : page {page_index + 1}/{len(pdf)}...", flush=True)
        png_version = pdf2png(pdf[page_index])
        text_version = llm_bases_img2text(vlm, png_version, "image/png", PDF_TRANSCRIPTION_PROMPT)

        if not (text_version or "").strip():
            print(f"Page {page_index + 1} vide après transcription", flush=True)
            continue

        documents.append(Document(page_content=text_version,  metadata={"source": str(path), "page": page_index + 1}))
    return documents


def load_image(path, vlm):
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return [Document(page_content=llm_bases_img2text(vlm, path.read_bytes(), mime, IMAGE_TRANSCRIPTION_PROMPT),
                     metadata={"source": str(path)})]


def load_text(path):
    return [Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path)})]


def iter_files(roots, suffixes=TEXT_SUFFIXES | IMAGE_SUFFIXES | {".pdf"}):
    """
    Yield (root, path) for every ingestable file, skipping private folders (any path component starting with "_").
    :param roots: A list of root directories to search for files.
    :param suffixes: A set of file suffixes to include (default includes text, image, and PDF files).
    :return: A generator yielding tuples of (root, path)
    """
    for root in roots:
        if not root.exists():
            print(f"Chemin introuvable, ignorée : {root}")
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in suffixes:
                print(f"\t\t{path.name} : suffixe ignoré", flush=True)
                continue
            if any(part.startswith("_") for part in path.relative_to(root).parts[:-1]):
                continue
            yield root, path


def load_file(root, path, vlm):
    """Transcribed documents of a single file, tagged with their metadata."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        documents = load_pdf(path, vlm)
    elif suffix in IMAGE_SUFFIXES:
        documents = load_image(path, vlm)
    else:
        documents = load_text(path)
    # One tag per folder level under the root, filterable in Chroma:
    # "05 - Clients/Techplaces/x.pdf" -> tag_1="05 - Clients", tag_2="Techplaces"
    tags = {f"tag_{i}": name
            for i, name in enumerate(path.relative_to(root).parts[:-1], 1)}
    for doc in documents:
        doc.metadata.update(tags, mtime=path.stat().st_mtime)
    return documents


def indexed_files(vectordb):
    """{source: {ids, mtime}} for everything currently in the store: the
    chunk ids to delete on update, and the mtime to compare against disk."""
    stored = vectordb.get(include=["metadatas"])
    indexed = {}
    for chunk_id, metadata in zip(stored["ids"], stored["metadatas"]):
        entry = indexed.setdefault(metadata["source"],
                                   {"ids": [], "mtime": metadata.get("mtime", 0)})
        entry["ids"].append(chunk_id)
    return indexed


def drop_deleted(vectordb, indexed, on_disk):
    """Remove the chunks of files that no longer exist on disk."""
    removed = [source for source in indexed if source not in on_disk]
    for source in removed:
        vectordb.delete(ids=indexed[source]["ids"])
        print(f"  retiré : {source}")
    return removed


def outdated_files(indexed, on_disk):
    """Files to transcribe again: absent from the index, or modified since.
    1s tolerance: cloud storage mounts (Google Drive) jitter sub-second
    mtimes between stat calls, which would re-index the same files forever."""
    return [(source, root, path) for source, (root, path) in on_disk.items()
            if source not in indexed
            or path.stat().st_mtime > indexed[source]["mtime"] + 1]


def index_file(vectordb, entry, root, path, vlm, chunk_size, chunk_overlap):
    """Transcribe one file and replace its chunks in the store. The old
    chunks are deleted only once the new ones are ready, so a crash mid-file
    leaves the previous version in place."""
    documents = load_file(root, path, vlm)
    chunks = markdown_splitter(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if entry:
        vectordb.delete(ids=entry["ids"])
    if chunks:
        vectordb.add_documents(chunks)
    return chunks


def sync(docs_dirs, vectordb, vlm, chunk_size, chunk_overlap, on_progress=None):
    """Incremental ingestion: transcribe and index new or modified files, drop
    the chunks of deleted files, leave everything else untouched. A full
    rebuild is simply sync() against an empty vector store.
    on_progress(done, total, current_name) is called as files get processed."""
    indexed = indexed_files(vectordb)
    on_disk = {str(path): (root, path) for root, path in iter_files(docs_dirs)}
    removed = drop_deleted(vectordb, indexed, on_disk)

    to_process = outdated_files(indexed, on_disk)
    total = len(to_process)
    if on_progress:
        on_progress(0, total, None)

    added = updated = 0
    failures = []
    for done, (source, root, path) in enumerate(to_process, 1):
        try:
            chunks = index_file(vectordb, indexed.get(source), root, path,
                                vlm, chunk_size, chunk_overlap)
        except Exception as exc:
            print(f"⚠ échec sur {path.name} : {exc}")
            failures.append({"file": path.name, "message": str(exc)})
            continue
        finally:
            if on_progress:
                on_progress(done, total, path.name)
        if source in indexed:
            updated += 1
        else:
            added += 1
        print(f"  indexé : {path.relative_to(root)} ({len(chunks)} chunks) [{done}/{total}]")

    print(f"Synchronisation terminée : {added} ajouté(s), {updated} mis à jour, "
          f"{len(removed)} retiré(s), {len(failures)} échec(s)")
    return {"added": added, "updated": updated, "removed": len(removed),
            "warnings": failures}


if __name__ == "__main__":
    import sys

    config = load_config()
    vectordb = Chroma(persist_directory=PERSIST_DIR,
                      embedding_function=embedding_client(config))

    # --full recreates the collection from scratch (required after an
    # embedding model change: vector spaces are not compatible)
    if "--full" in sys.argv:
        vectordb.reset_collection()

    # Roots to index (Google Drive mounts) come from DOCS_DIRS in .env:
    # personal paths stay out of the repo (see .env.example)
    sync(docs_dirs(), vectordb, vlm_client(config),
         chunk_size=config["ingestion"]["chunk_size"],
         chunk_overlap=config["ingestion"]["chunk_overlap"])
