from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def markdown_splitter(documents, chunk_size, chunk_overlap):
    """Structure-aware chunking: split on Markdown headers (a section stays
    whole, tables included), then re-split only the sections exceeding
    chunk_size. The header path is prepended to each chunk, giving the
    embedding its context, and stored in the metadata."""
    by_header = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS)
    recursive = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                               chunk_overlap=chunk_overlap,
                                               length_function=len)
    chunks = []
    for doc in documents:
        for section in by_header.split_text(doc.page_content):
            header_path = " > ".join(section.metadata[key]
                                     for _, key in HEADERS if key in section.metadata)
            for piece in recursive.split_text(section.page_content):
                metadata = dict(doc.metadata)
                if header_path:
                    metadata["section"] = header_path
                    piece = f"{header_path}\n{piece}"
                chunks.append(Document(page_content=piece, metadata=metadata))
    return chunks
