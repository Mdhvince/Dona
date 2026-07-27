from pathlib import Path

import pypdfium2 as pdfium
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from markitdown import MarkItDown

from src.config import load_config, docs_dirs, embedding_client, vlm_client
from src.prompt import IMAGE_TRANSCRIPTION_PROMPT, PDF_TRANSCRIPTION_PROMPT
from src.ingestion_utilities import iter_ingestable_files, transcribe_image, render_page_to_png

PERSIST_DIR = str(Path(__file__).parent.parent / "vectordb")

MARKDOWN_SUFFIXES = {".md"}
TEXT_SUFFIXES = {".txt"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MARKUP_SUFFIXES = {".docx", ".pptx", ".html", ".htm"}
INGESTABLE_SUFFIXES = MARKDOWN_SUFFIXES | TEXT_SUFFIXES | IMAGE_SUFFIXES | MARKUP_SUFFIXES | {".pdf"}

MARKDOWN_HEADER_LEVELS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


class Ingestor:
    """
    Ingestion pipeline: transcribe files, chunk them, and keep the vector store in sync with the disk.
    """

    def __init__(self, vectordb, vlm, chunk_size, chunk_overlap):
        self.vectordb = vectordb
        self.vlm = vlm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_pdf(self, path):
        """
        Render each page to an image and let the vision model transcribe it to Markdown.
        :param path: The path to the PDF file.
        :return: A list of Document objects
        """
        pdf = pdfium.PdfDocument(str(path))
        documents = []
        for page_index in range(len(pdf)):
            print(f"\t\t{path.name} : page {page_index + 1}/{len(pdf)}...", flush=True)
            png_version = render_page_to_png(pdf[page_index])
            text_version = transcribe_image(self.vlm, png_version, "image/png", PDF_TRANSCRIPTION_PROMPT)

            if not (text_version or "").strip():
                print(f"Page {page_index + 1} vide après transcription", flush=True)
                continue

            documents.append(Document(page_content=text_version,  metadata={"source": str(path), "page": page_index + 1}))
        return documents

    def load_image(self, path):
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return [Document(page_content=transcribe_image(self.vlm, path.read_bytes(), mime, IMAGE_TRANSCRIPTION_PROMPT),
                         metadata={"source": str(path)})]

    def load_text(self, path):
        return [Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path)})]

    def load_markup(self, path):
        """
        Read a file whose structure is already written in markup (OOXML, HTML) and turn it into
        Markdown, the format specifics being handled by MarkItDown.
        :param path: The path to the file.
        :return: A list of Document objects
        """
        text = MarkItDown().convert(str(path)).text_content
        return [Document(page_content=text, metadata={"source": str(path)})]

    def load_file(self, path):
        """
        Load a file and return the format it produced along with its Document objects. The chunking
        strategy follows that format, not the original suffix: a PDF transcribed by the vision model
        is Markdown, and is chunked as such.
        :param path: The path to the file.
        :return: A tuple (format, list of Document objects)
        """
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            fmt, documents = "markdown", self.load_pdf(path)
        elif suffix in IMAGE_SUFFIXES:
            fmt, documents = "markdown", self.load_image(path)
        elif suffix in MARKUP_SUFFIXES:
            fmt, documents = "markdown", self.load_markup(path)
        elif suffix in MARKDOWN_SUFFIXES:
            fmt, documents = "markdown", self.load_text(path)
        else:
            fmt, documents = "text", self.load_text(path)

        for doc in documents:
            doc.metadata["mtime"] = path.stat().st_mtime
        return fmt, documents

    def create_markdown_based_chunks(self, documents):
        """
        Split documents into chunks based on Markdown headers, then further split into smaller chunks if necessary.
        :param documents: A list of Document objects to be chunked.
        :return: A list of Document objects representing the chunks.
        """
        by_header = MarkdownHeaderTextSplitter(headers_to_split_on=MARKDOWN_HEADER_LEVELS)
        recursive = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                   chunk_overlap=self.chunk_overlap,
                                                   length_function=len)
        chunks = []
        for doc in documents:
            for section in by_header.split_text(doc.page_content):
                # Create some context for the chunk by prepending the header path to the content and add it to the metadata.
                header_path = " > ".join(section.metadata[key] for _, key in MARKDOWN_HEADER_LEVELS if key in section.metadata)

                for piece in recursive.split_text(section.page_content):
                    metadata = dict(doc.metadata)
                    if header_path:
                        metadata["section"] = header_path
                        piece = f"{header_path}\n{piece}"
                    chunks.append(Document(page_content=piece, metadata=metadata))
        return chunks

    def create_text_based_chunks(self, documents):
        """
        Split documents that carry no structure, on paragraph then sentence then word boundaries.
        :param documents: A list of Document objects to be chunked.
        :return: A list of Document objects representing the chunks.
        """
        recursive = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                   chunk_overlap=self.chunk_overlap,
                                                   length_function=len)
        return recursive.split_documents(documents)

    def create_chunks(self, fmt, documents):
        """
        Route the documents to the splitter matching the format their loader produced.
        :param fmt: The format produced by the loader.
        :param documents: A list of Document objects to be chunked.
        :return: A list of Document objects representing the chunks.
        """
        if fmt == "text":
            return self.create_text_based_chunks(documents)
        return self.create_markdown_based_chunks(documents)

    def fetch_indexed_chunks_by_source(self):
        """
        Fetch the files that have already been indexed, and their last modification time.
        :return: A dictionary mapping file paths to their indexed metadata, including chunk IDs and last modification time.
        """
        stored = self.vectordb.get(include=["metadatas"])
        indexed = {}
        for chunk_id, metadata in zip(stored["ids"], stored["metadatas"]):
            entry = indexed.setdefault(metadata["source"], {"ids": [], "mtime": metadata.get("mtime", 0)})
            entry["ids"].append(chunk_id)
        return indexed

    def delete_chunks_of_missing_files(self, indexed, on_disk):
        """
        Remove the chunks of files that no longer exist on disk.
        :param indexed: A dictionary of currently indexed files and their metadata.
        :param on_disk: A dictionary of files currently present on disk.
        :return: A list of removed files.
        """
        removed = [source for source in indexed if source not in on_disk]
        for source in removed:
            self.vectordb.delete(ids=indexed[source]["ids"])
            print(f"\t\tretiré : {source}")
        return removed

    def files_to_reindex(self, indexed, on_disk):
        """
        Determine which files need to be reindexed based on their presence on disk and their last modification time.
        If a file is new or has been modified since it was last indexed, it will be reindexed.
        :param indexed: A dictionary of currently indexed files and their metadata.
        :param on_disk: A dictionary of files currently present on disk.
        :return: A list of tuples (source, root, path) for the files to reindex.
        """
        return [(source, root, path) for source, (root, path) in on_disk.items()
                if source not in indexed
                or path.stat().st_mtime > indexed[source]["mtime"] + 1]

    def index_file(self, indexed_entry, path):
        """
        Transcribe one file and replace its chunks in the store. The old chunks are deleted only once the new
        ones are ready, so a crash mid-file leaves the previous version in place.
        :param indexed_entry: The chunk ids and mtime already stored for this file, or None when it is new.
        :param path: The path to the file.
        :return: A list of Document objects representing the chunks.
        """
        fmt, documents = self.load_file(path)
        chunks = self.create_chunks(fmt, documents)
        if indexed_entry:
            self.vectordb.delete(ids=indexed_entry["ids"])
        if chunks:
            self.vectordb.add_documents(chunks)
        return chunks

    def run(self, docs_dirs):
        """
        Incremental ingestion: transcribe and index new or modified files, drop the chunks of deleted files,
        leave everything else untouched. A full rebuild is simply an ingestion against an empty vector store.
        :param docs_dirs: The root directories to ingest.
        :return: A summary of the run: added, updated, removed counts and the files that failed.
        """
        indexed = self.fetch_indexed_chunks_by_source()
        on_disk = {str(path): (root, path) for root, path in iter_ingestable_files(docs_dirs, suffixes=INGESTABLE_SUFFIXES)}
        removed = self.delete_chunks_of_missing_files(indexed, on_disk)

        to_process = self.files_to_reindex(indexed, on_disk)
        total = len(to_process)

        added = updated = 0
        failures = []
        for done, (source, root, path) in enumerate(to_process, 1):
            try:
                chunks = self.index_file(indexed.get(source), path)
            except Exception as exc:
                print(f"Failed on {path.name} : {exc}")
                failures.append({"file": path.name, "message": str(exc)})
                continue

            was_already_indexed = source in indexed
            updated += 1 if was_already_indexed else 0
            added += 1 if not was_already_indexed else 0
            print(f"\t\tIndexed : {path.relative_to(root)} ({len(chunks)} chunks) [{done}/{total}]")

        print(f"Ingestion done : {added} added, {updated} updated, {len(removed)} removed, {len(failures)} failures")
        return {"added": added, "updated": updated, "removed": len(removed), "warnings": failures}


if __name__ == "__main__":
    import sys

    config = load_config()
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_client(config))

    if "--full" in sys.argv:
        vectordb.reset_collection()

    ingestor = Ingestor(vectordb, vlm_client(config),
                        chunk_size=config["ingestion"]["chunk_size"],
                        chunk_overlap=config["ingestion"]["chunk_overlap"])
    ingestor.run(docs_dirs())
