from langchain_core.documents import Document

from src.ingest import create_markdown_based_chunks


def split(text, chunk_size=1000, chunk_overlap=0, metadata=None):
    return create_markdown_based_chunks([Document(page_content=text, metadata=metadata or {})],
                                        chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_short_section_stays_whole():
    chunks = split("# Titre\nligne 1\nligne 2")
    assert len(chunks) == 1


def test_header_path_is_prepended_and_stored_in_metadata():
    chunks = split("# Avis 2024\n## Calcul du solde\nmontant 9570")
    assert chunks[0].metadata["section"] == "Avis 2024 > Calcul du solde"
    assert chunks[0].page_content.startswith("Avis 2024 > Calcul du solde\n")


def test_long_section_is_resplit():
    chunks = split("# T\n" + "\n\n".join(f"paragraphe {i} " + "x" * 80 for i in range(10)),
                   chunk_size=200)
    assert len(chunks) > 1
    assert all(chunk.metadata["section"] == "T" for chunk in chunks)


def test_document_metadata_is_inherited():
    chunks = split("# T\ncontenu", metadata={"source": "/tmp/doc.pdf", "page": 3})
    assert chunks[0].metadata["source"] == "/tmp/doc.pdf"
    assert chunks[0].metadata["page"] == 3


def test_text_without_headers_still_chunks():
    chunks = split("juste du texte brut sans titre")
    assert len(chunks) == 1
    assert "section" not in chunks[0].metadata
