from langchain_core.documents import Document

from response_helper import cited_sources, format_context, source_links

DOCS = [
    Document(page_content="a", metadata={"source": "/tmp/avis_2024.pdf", "page": 2}),
    Document(page_content="b", metadata={"source": "/tmp/avis_2023.pdf", "page": 1}),
    Document(page_content="c", metadata={"source": "/tmp/avis_2024.pdf", "page": 2}),
    Document(page_content="d", metadata={}),
]


def test_cited_sources_returns_path_and_page():
    assert cited_sources(DOCS, [1]) == [("/tmp/avis_2024.pdf", "2")]


def test_cited_sources_skips_out_of_bounds_numbers():
    assert cited_sources(DOCS, [0, 99, -3]) == []


def test_cited_sources_handles_empty_and_none():
    assert cited_sources(DOCS, []) == []
    assert cited_sources(DOCS, None) == []


def test_cited_sources_deduplicates_same_source_and_page():
    assert cited_sources(DOCS, [1, 3]) == [("/tmp/avis_2024.pdf", "2")]


def test_cited_sources_skips_chunks_without_source():
    assert cited_sources(DOCS, [4]) == []


def test_format_context_numbers_chunks_and_names_files():
    context = format_context(DOCS[:2])
    assert "[1] (avis_2024.pdf, page 2) a" in context
    assert "[2] (avis_2023.pdf, page 1) b" in context


def test_source_links_joins_multiple_sources():
    links = source_links([("/tmp/a.pdf", "1"), ("/tmp/b.pdf", None)])
    assert "[a.pdf]" in links and "[b.pdf]" in links and " ; " in links
    assert source_links([]) is None
