from pathlib import Path

from src.ingest import Ingestor


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


# --- sync mtime tolerance ---

def make_root(tmp_path):
    root = tmp_path / "drive"
    root.mkdir()
    doc = root / "doc.txt"
    doc.write_text("contenu du document", encoding="utf-8")
    return root, doc


def stored_db(doc, mtime):
    return FakeVectordb(ids=["c1"], metadatas=[{"source": str(doc), "mtime": mtime}])


def test_sync_ignores_sub_second_mtime_jitter(tmp_path):
    root, doc = make_root(tmp_path)
    db = stored_db(doc, doc.stat().st_mtime - 0.5)
    result = Ingestor(db, vlm=None, chunk_size=1000, chunk_overlap=0).run([root])
    assert result["added"] == 0 and result["updated"] == 0
    assert db.added == [] and db.deleted == []


def test_sync_reindexes_really_modified_file(tmp_path):
    root, doc = make_root(tmp_path)
    db = stored_db(doc, doc.stat().st_mtime - 5)
    result = Ingestor(db, vlm=None, chunk_size=1000, chunk_overlap=0).run([root])
    assert result["updated"] == 1
    assert db.deleted == [["c1"]]
    assert len(db.added) >= 1


def test_sync_indexes_new_file(tmp_path):
    root, doc = make_root(tmp_path)
    db = FakeVectordb()
    result = Ingestor(db, vlm=None, chunk_size=1000, chunk_overlap=0).run([root])
    assert result["added"] == 1
    assert db.added[0].metadata["source"] == str(doc)


def test_sync_removes_deleted_file(tmp_path):
    root, _ = make_root(tmp_path)
    ghost = str(Path(root) / "disparu.txt")
    db = FakeVectordb(ids=["c9"], metadatas=[{"source": ghost, "mtime": 0}])
    result = Ingestor(db, vlm=None, chunk_size=1000, chunk_overlap=0).run([root])
    assert result["removed"] == 1
    assert ["c9"] in db.deleted


# --- format routing ---

def loader():
    return Ingestor(vectordb=None, vlm=None, chunk_size=1000, chunk_overlap=0)


def test_plain_text_and_markdown_are_routed_apart(tmp_path):
    (tmp_path / "notes.txt").write_text("du texte", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# Titre\ndu texte", encoding="utf-8")
    assert loader().load_file(tmp_path / "notes.txt")[0] == "text"
    assert loader().load_file(tmp_path / "notes.md")[0] == "markdown"


def test_html_is_converted_to_markdown(tmp_path):
    path = tmp_path / "page.html"
    path.write_text("<h1>Titre</h1><p>contenu</p>", encoding="utf-8")
    fmt, documents = loader().load_file(path)
    assert fmt == "markdown"
    assert documents[0].page_content.startswith("# Titre")
