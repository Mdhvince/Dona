from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.ingestor import Ingestor


class FakeVectordb:
    """Minimal stand-in for Chroma: get / delete / add_documents."""

    def __init__(self, ids=None, metadatas=None):
        self.stored = {"ids": ids or [], "metadatas": metadatas or []}
        self.deleted = []
        self.added = []

    def get(self, include=None):
        return self.stored

    def delete(self, ids):
        self.deleted.append(ids)

    def add_documents(self, documents):
        self.added.extend(documents)


class FakePdf:
    """Stand-in for a pypdfium2 document: a sized, indexable list of pages."""

    def __init__(self, pages):
        self.pages = pages

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, index):
        return self.pages[index]


@pytest.fixture
def ingestor():
    return Ingestor(FakeVectordb(), vlm="vlm", chunk_size=1000, chunk_overlap=0)


def transcribe_pages(monkeypatch, transcriptions):
    """Wire ingest_pdf onto fake pages, each transcribed to the given text."""
    monkeypatch.setattr("src.ingestor.pdfium.PdfDocument",
                        lambda _path: FakePdf([f"page{i}" for i in range(len(transcriptions))]))
    monkeypatch.setattr("src.ingestor.pdf2png", lambda page: page.encode())
    pages = iter(transcriptions)
    monkeypatch.setattr("src.ingestor.llm_bases_img2text",
                        lambda vlm, image, mime, prompt: next(pages))


# --- loaders ---

def test_text_is_read_as_utf8(tmp_path, ingestor):
    path = tmp_path / "notes.txt"
    path.write_text("impôts payés à Créteil", encoding="utf-8")
    documents = ingestor.ingest_text(path)
    assert documents[0].page_content == "impôts payés à Créteil"
    assert documents[0].metadata == {"source": str(path)}


def test_markup_is_converted_to_markdown(tmp_path, ingestor):
    path = tmp_path / "page.html"
    path.write_text("<h1>Titre</h1><p>contenu</p>", encoding="utf-8")
    documents = ingestor.ingest_markup(path)
    assert documents[0].page_content.startswith("# Titre")
    assert "contenu" in documents[0].page_content


@pytest.mark.parametrize("name, expected_mime", [("scan.png", "image/png"),
                                                 ("scan.jpg", "image/jpeg"),
                                                 ("scan.JPEG", "image/jpeg")])
def test_image_mime_follows_the_suffix(tmp_path, ingestor, monkeypatch, name, expected_mime):
    seen = {}

    def fake_transcription(vlm, image_bytes, mime, prompt):
        seen.update(vlm=vlm, image_bytes=image_bytes, mime=mime)
        return "texte de l'image"

    monkeypatch.setattr("src.ingestor.llm_bases_img2text", fake_transcription)
    path = tmp_path / name
    path.write_bytes(b"\x89PNG binaire")

    documents = ingestor.ingest_image(path)
    assert seen["mime"] == expected_mime
    assert seen["vlm"] == "vlm" and seen["image_bytes"] == b"\x89PNG binaire"
    assert documents[0].page_content == "texte de l'image"
    assert documents[0].metadata == {"source": str(path)}


def test_pdf_yields_one_document_per_page_numbered_from_one(tmp_path, ingestor, monkeypatch):
    transcribe_pages(monkeypatch, ["# Page une", "# Page deux"])
    documents = ingestor.ingest_pdf(tmp_path / "avis.pdf")
    assert [doc.page_content for doc in documents] == ["# Page une", "# Page deux"]
    assert [doc.metadata["page"] for doc in documents] == [1, 2]
    assert all(doc.metadata["source"] == str(tmp_path / "avis.pdf") for doc in documents)


@pytest.mark.parametrize("blank", ["", "   \n\t", None])
def test_pdf_skips_pages_the_model_returned_empty(tmp_path, ingestor, monkeypatch, blank):
    transcribe_pages(monkeypatch, [blank, "# Page deux"])
    documents = ingestor.ingest_pdf(tmp_path / "avis.pdf")
    assert [doc.metadata["page"] for doc in documents] == [2]


# --- format routing ---

def test_plain_text_and_markdown_are_routed_apart(tmp_path, ingestor):
    (tmp_path / "notes.txt").write_text("du texte", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# Titre\ndu texte", encoding="utf-8")
    assert ingestor.load_file(tmp_path / "notes.txt")[0] == "text"
    assert ingestor.load_file(tmp_path / "notes.md")[0] == "markdown"


def test_an_unknown_suffix_falls_back_to_plain_text(tmp_path, ingestor):
    path = tmp_path / "sortie.log"
    path.write_text("ligne de log", encoding="utf-8")
    fmt, documents = ingestor.load_file(path)
    assert fmt == "text"
    assert documents[0].page_content == "ligne de log"


def test_html_is_loaded_as_markdown(tmp_path, ingestor):
    path = tmp_path / "page.html"
    path.write_text("<h1>Titre</h1>", encoding="utf-8")
    fmt, documents = ingestor.load_file(path)
    assert fmt == "markdown"
    assert documents[0].page_content.startswith("# Titre")


def test_image_is_loaded_as_markdown(tmp_path, ingestor, monkeypatch):
    monkeypatch.setattr("src.ingestor.llm_bases_img2text", lambda *args: "# Photo")
    path = tmp_path / "photo.png"
    path.write_bytes(b"binaire")
    assert ingestor.load_file(path)[0] == "markdown"


def test_pdf_is_loaded_as_markdown_and_every_page_carries_the_mtime(tmp_path, ingestor, monkeypatch):
    path = tmp_path / "avis.pdf"
    path.write_bytes(b"%PDF")
    transcribe_pages(monkeypatch, ["# Une", "# Deux"])
    fmt, documents = ingestor.load_file(path)
    assert fmt == "markdown"
    assert all(doc.metadata["mtime"] == path.stat().st_mtime for doc in documents)


# --- chunk routing ---

def test_chunk_documents_follows_the_markdown_strategy(ingestor):
    chunks = ingestor.chunk_documents("markdown", [Document(page_content="# Titre\ncontenu")])
    assert chunks[0].metadata["section"] == "Titre"


def test_chunk_documents_follows_the_text_strategy(ingestor):
    chunks = ingestor.chunk_documents("text", [Document(page_content="# Titre\ncontenu")])
    assert "section" not in chunks[0].metadata
    assert chunks[0].page_content == "# Titre\ncontenu"


# --- index bookkeeping ---

def test_indexed_files_group_their_chunk_ids_by_source():
    db = FakeVectordb(ids=["a1", "a2", "b1"],
                      metadatas=[{"source": "/a.pdf", "mtime": 10},
                                 {"source": "/a.pdf", "mtime": 10},
                                 {"source": "/b.pdf", "mtime": 20}])
    indexed = Ingestor(db, vlm=None, chunk_size=1000, chunk_overlap=0).fetch_indexed_files()
    assert indexed == {"/a.pdf": {"ids": ["a1", "a2"], "mtime": 10},
                       "/b.pdf": {"ids": ["b1"], "mtime": 20}}


def test_a_chunk_without_mtime_is_treated_as_never_indexed():
    db = FakeVectordb(ids=["a1"], metadatas=[{"source": "/a.pdf"}])
    indexed = Ingestor(db, vlm=None, chunk_size=1000, chunk_overlap=0).fetch_indexed_files()
    assert indexed["/a.pdf"]["mtime"] == 0


def test_an_empty_store_has_nothing_indexed(ingestor):
    assert ingestor.fetch_indexed_files() == {}


def test_only_the_sources_missing_from_disk_are_dropped(tmp_path):
    db = FakeVectordb()
    indexed = {"/reste.pdf": {"ids": ["k1"], "mtime": 0},
               "/disparu.pdf": {"ids": ["k9"], "mtime": 0}}
    removed = Ingestor(db, vlm=None, chunk_size=1000, chunk_overlap=0) \
        .unindex_missing_files_from_disk(indexed, {"/reste.pdf": (tmp_path, tmp_path)})
    assert removed == ["/disparu.pdf"]
    assert db.deleted == [["k9"]]


def test_nothing_is_deleted_when_every_source_is_still_on_disk(tmp_path):
    db = FakeVectordb()
    indexed = {"/reste.pdf": {"ids": ["k1"], "mtime": 0}}
    removed = Ingestor(db, vlm=None, chunk_size=1000, chunk_overlap=0) \
        .unindex_missing_files_from_disk(indexed, {"/reste.pdf": (tmp_path, tmp_path)})
    assert removed == [] and db.deleted == []


def reindex_decision(ingestor, path, indexed_mtime):
    on_disk = {str(path): (path.parent, path)}
    indexed = {} if indexed_mtime is None else {str(path): {"ids": ["k1"], "mtime": indexed_mtime}}
    return ingestor.files_to_reindex(indexed, on_disk)


def test_a_file_absent_from_the_index_is_reindexed(tmp_path, ingestor):
    path = tmp_path / "doc.txt"
    path.write_text("contenu", encoding="utf-8")
    assert reindex_decision(ingestor, path, None) == [(str(path), tmp_path, path)]


def test_a_file_modified_beyond_the_tolerance_is_reindexed(tmp_path, ingestor):
    path = tmp_path / "doc.txt"
    path.write_text("contenu", encoding="utf-8")
    assert reindex_decision(ingestor, path, path.stat().st_mtime - 5) != []


def test_sub_second_mtime_jitter_does_not_trigger_a_reindex(tmp_path, ingestor):
    path = tmp_path / "doc.txt"
    path.write_text("contenu", encoding="utf-8")
    assert reindex_decision(ingestor, path, path.stat().st_mtime - 0.5) == []


# --- index_file ---

def test_a_new_file_is_added_without_any_deletion(tmp_path):
    db = FakeVectordb()
    path = tmp_path / "doc.txt"
    path.write_text("contenu du document", encoding="utf-8")
    chunks = Ingestor(db, vlm=None, chunk_size=1000, chunk_overlap=0).index_file(None, path)
    assert db.deleted == []
    assert db.added == chunks and len(chunks) == 1


def test_old_chunks_are_deleted_only_once_the_new_ones_are_ready(tmp_path):
    db = FakeVectordb()
    path = tmp_path / "doc.txt"
    path.write_text("nouveau contenu", encoding="utf-8")
    Ingestor(db, vlm=None, chunk_size=1000, chunk_overlap=0).index_file({"ids": ["vieux"]}, path)
    assert db.deleted == [["vieux"]]
    assert len(db.added) == 1


def test_an_empty_file_adds_nothing_but_still_drops_its_old_chunks(tmp_path):
    db = FakeVectordb()
    path = tmp_path / "vide.txt"
    path.write_text("", encoding="utf-8")
    chunks = Ingestor(db, vlm=None, chunk_size=1000, chunk_overlap=0).index_file({"ids": ["vieux"]}, path)
    assert chunks == []
    assert db.deleted == [["vieux"]] and db.added == []


# --- run ---

def make_root(tmp_path):
    root = tmp_path / "drive"
    root.mkdir()
    doc = root / "doc.txt"
    doc.write_text("contenu du document", encoding="utf-8")
    return root, doc


def stored_db(doc, mtime):
    return FakeVectordb(ids=["c1"], metadatas=[{"source": str(doc), "mtime": mtime}])


def test_run_ignores_sub_second_mtime_jitter(tmp_path):
    root, doc = make_root(tmp_path)
    db = stored_db(doc, doc.stat().st_mtime - 0.5)
    result = Ingestor(db, vlm=None, chunk_size=1000, chunk_overlap=0).run([root])
    assert result["added"] == 0 and result["updated"] == 0
    assert db.added == [] and db.deleted == []


def test_run_reindexes_really_modified_file(tmp_path):
    root, doc = make_root(tmp_path)
    db = stored_db(doc, doc.stat().st_mtime - 5)
    result = Ingestor(db, vlm=None, chunk_size=1000, chunk_overlap=0).run([root])
    assert result["updated"] == 1
    assert db.deleted == [["c1"]]
    assert len(db.added) >= 1


def test_run_indexes_new_file(tmp_path):
    root, doc = make_root(tmp_path)
    db = FakeVectordb()
    result = Ingestor(db, vlm=None, chunk_size=1000, chunk_overlap=0).run([root])
    assert result["added"] == 1
    assert db.added[0].metadata["source"] == str(doc)


def test_run_removes_deleted_file(tmp_path):
    root, _ = make_root(tmp_path)
    ghost = str(Path(root) / "disparu.txt")
    db = FakeVectordb(ids=["c9"], metadatas=[{"source": ghost, "mtime": 0}])
    result = Ingestor(db, vlm=None, chunk_size=1000, chunk_overlap=0).run([root])
    assert result["removed"] == 1
    assert ["c9"] in db.deleted


def test_run_reports_a_failing_file_and_keeps_going(tmp_path, monkeypatch):
    root, _ = make_root(tmp_path)
    (root / "casse.txt").write_text("contenu", encoding="utf-8")
    ingestor = Ingestor(FakeVectordb(), vlm=None, chunk_size=1000, chunk_overlap=0)

    def index_file(entry, path):
        if path.name == "casse.txt":
            raise RuntimeError("transcription impossible")
        return [Document(page_content="ok", metadata={"source": str(path)})]

    monkeypatch.setattr(ingestor, "index_file", index_file)
    result = ingestor.run([root])
    assert result["added"] == 1
    assert result["warnings"] == [{"file": "casse.txt", "message": "transcription impossible"}]


def test_run_skips_private_folders(tmp_path):
    root, _ = make_root(tmp_path)
    (root / "_prive").mkdir()
    (root / "_prive" / "secret.txt").write_text("confidentiel", encoding="utf-8")
    db = FakeVectordb()
    result = Ingestor(db, vlm=None, chunk_size=1000, chunk_overlap=0).run([root])
    assert result["added"] == 1
    assert all("secret" not in doc.metadata["source"] for doc in db.added)
