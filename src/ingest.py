import base64
import io
import re
from pathlib import Path

import pypdfium2 as pdfium
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pypdf import PdfReader

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


def _collapse_digit_groups(text):
    """Glue thousands-separated numbers back together: "9 570" -> "9570"."""
    return re.sub(r"(?<=\d)\s+(?=\d)", "", text)


def extract_numbers(text):
    return set(re.findall(r"\d+", text))


def find_invented_numbers(raw_text, transcribed_texts):
    """Numbers present in the transcriptions but absent from the reference
    text. Two granularities (raw and collapsed) tolerate digit-grouping
    differences, plus a substring test for long identifiers that the
    reference segments differently. Pure function, testable without a PDF."""
    collapsed_raw = _collapse_digit_groups(raw_text)
    reference = extract_numbers(raw_text) | extract_numbers(collapsed_raw)
    transcribed = set()
    for text in transcribed_texts:
        transcribed |= extract_numbers(_collapse_digit_groups(text))
    return {n for n in transcribed - reference
            if len(n) >= 2 and n not in collapsed_raw}


def validate_transcription(path, documents):
    """Anti-hallucination guard: every number transcribed by the VLM must
    exist in the PDF text layer, which holds the real values even when
    classic extraction detaches them from their labels.
    Returns a list of warnings; empty when everything checks out."""
    try:
        raw = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    except Exception as exc:
        return [f"couche texte illisible ({exc}) : transcription non vérifiable"]
    if not raw.strip():
        return ["PDF scanné sans couche texte : transcription non vérifiable, contrôle manuel conseillé"]

    invented = find_invented_numbers(raw, [doc.page_content for doc in documents])
    if invented:
        return ["nombres absents de la couche texte : " + ", ".join(sorted(invented)[:10])]
    return []


def load_pdf(path, vlm):
    """Render each page to an image and let the vision model transcribe it to
    Markdown: the only approach that keeps labels and values together on
    column-based PDFs (tax notices...) where both text extractors and layout
    parsers fail."""
    pdf = pdfium.PdfDocument(str(path))
    documents = []
    for i in range(len(pdf)):
        # flush: page-by-page progress must show up even when stdout is piped
        print(f"    {path.name} : page {i + 1}/{len(pdf)}...", flush=True)
        documents.append(
            Document(page_content=llm_bases_img2text(vlm, pdf2png(pdf[i]), "image/png", PDF_TRANSCRIPTION_PROMPT),
                     metadata={"source": str(path), "page": i + 1}))
    warnings = validate_transcription(path, documents)
    for warning in warnings:
        print(f"  ⚠ validation {path.name} : {warning}")
    return documents, warnings


def load_image(path, vlm):
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return [Document(page_content=llm_bases_img2text(vlm, path.read_bytes(), mime, IMAGE_TRANSCRIPTION_PROMPT),
                     metadata={"source": str(path)})]


def load_text(path):
    return [Document(page_content=path.read_text(encoding="utf-8"),
                     metadata={"source": str(path)})]


def iter_files(roots, suffixes=TEXT_SUFFIXES | IMAGE_SUFFIXES | {".pdf"}):
    """Yield (root, path) for every ingestable file, skipping private folders
    (any path component starting with "_")."""
    for root in roots:
        if not root.exists():
            print(f"⚠ racine introuvable, ignorée : {root}")
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in suffixes:
                continue
            if any(part.startswith("_") for part in path.relative_to(root).parts[:-1]):
                continue
            yield root, path


def load_file(root, path, vlm):
    """Returns (documents, warnings) for a single file; warnings only come
    from the PDF transcription validation."""
    suffix = path.suffix.lower()
    warnings = []
    if suffix == ".pdf":
        documents, warnings = load_pdf(path, vlm)
    elif suffix in IMAGE_SUFFIXES:
        documents = load_image(path, vlm)
    else:
        documents = load_text(path)
    # One tag per folder level under the root, filterable in Chroma:
    # "05 - Clients/Techplaces/x.pdf" -> tag_1="05 - Clients", tag_2="Techplaces"
    tags = {f"tag_{i}": name
            for i, name in enumerate(path.relative_to(root).parts[:-1], 1)}
    # mtime lets sync() detect modified files on the next run
    for doc in documents:
        doc.metadata.update(tags, mtime=path.stat().st_mtime)
    return documents, warnings


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
    """Transcribe one file and replace its chunks in the store. Returns its
    warnings; the old chunks are deleted only once the new ones are ready,
    so a crash mid-file leaves the previous version in place."""
    documents, warnings = load_file(root, path, vlm)
    chunks = markdown_splitter(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if entry:
        vectordb.delete(ids=entry["ids"])
    if chunks:
        vectordb.add_documents(chunks)
    return chunks, warnings


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
    all_warnings = []
    for done, (source, root, path) in enumerate(to_process, 1):
        try:
            chunks, warnings = index_file(vectordb, indexed.get(source), root, path,
                                          vlm, chunk_size, chunk_overlap)
        except Exception as exc:
            print(f"⚠ échec sur {path.name} : {exc}")
            all_warnings.append({"file": path.name, "message": f"échec : {exc}"})
            continue
        finally:
            if on_progress:
                on_progress(done, total, path.name)
        all_warnings.extend({"file": path.name, "message": warning} for warning in warnings)
        if source in indexed:
            updated += 1
        else:
            added += 1
        print(f"  indexé : {path.relative_to(root)} ({len(chunks)} chunks) [{done}/{total}]")

    print(f"Synchronisation terminée : {added} ajouté(s), {updated} mis à jour, "
          f"{len(removed)} retiré(s), {len(all_warnings)} alerte(s)")
    return {"added": added, "updated": updated, "removed": len(removed),
            "warnings": all_warnings}


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
