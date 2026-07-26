import base64
import io
from types import SimpleNamespace

from PIL import Image

from src.ingestion_utilities import iter_files, llm_bases_img2text, pdf2png


class FakeVlm:
    """Records the messages it receives and returns a fixed transcription."""

    def __init__(self, content="texte transcrit"):
        self.content = content
        self.received = []

    def invoke(self, messages):
        self.received.append(messages)
        return SimpleNamespace(content=self.content)


class FakePage:
    """Stand-in for a pypdfium2 page: records the scale it was rendered at."""

    def __init__(self, image):
        self.image = image
        self.scale = None

    def render(self, scale):
        self.scale = scale
        return SimpleNamespace(to_pil=lambda: self.image)


# --- llm_bases_img2text ---

def test_transcription_returns_the_model_content():
    vlm = FakeVlm(content="# Avis 2024")
    assert llm_bases_img2text(vlm, b"octets", "image/png", "transcris") == "# Avis 2024"


def test_transcription_sends_the_prompt_then_the_image():
    vlm = FakeVlm()
    llm_bases_img2text(vlm, b"octets", "image/png", "transcris cette page")
    parts = vlm.received[0][0].content
    assert parts[0] == {"type": "text", "text": "transcris cette page"}
    assert parts[1]["type"] == "image_url"


def test_transcription_encodes_the_image_as_a_data_uri():
    vlm = FakeVlm()
    image_bytes = bytes(range(256))
    llm_bases_img2text(vlm, image_bytes, "image/jpeg", "transcris")
    url = vlm.received[0][0].content[1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == image_bytes


# --- pdf2png ---

def test_render_produces_real_png_bytes():
    page = FakePage(Image.new("RGB", (12, 8), "white"))
    png = pdf2png(page)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert Image.open(io.BytesIO(png)).size == (12, 8)


def test_render_scales_from_dpi_against_the_72_dpi_baseline():
    page = FakePage(Image.new("RGB", (4, 4), "white"))
    pdf2png(page, dpi=144)
    assert page.scale == 2.0


def test_render_defaults_to_150_dpi():
    page = FakePage(Image.new("RGB", (4, 4), "white"))
    pdf2png(page)
    assert page.scale == 150 / 72


# --- iter_files ---

def make_file(root, relative, content="contenu"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_only_the_listed_suffixes_are_yielded(tmp_path):
    make_file(tmp_path, "garde.txt")
    make_file(tmp_path, "ignore.zip")
    yielded = [path for _, path in iter_files([tmp_path], {".txt"})]
    assert [path.name for path in yielded] == ["garde.txt"]


def test_suffix_matching_ignores_case(tmp_path):
    make_file(tmp_path, "SCAN.PDF")
    assert len(list(iter_files([tmp_path], {".pdf"}))) == 1


def test_private_folders_are_skipped_at_any_depth(tmp_path):
    make_file(tmp_path, "public.txt")
    make_file(tmp_path, "_prive/secret.txt")
    make_file(tmp_path, "compta/_prive/secret.txt")
    yielded = [path for _, path in iter_files([tmp_path], {".txt"})]
    assert [path.name for path in yielded] == ["public.txt"]


def test_a_file_whose_name_starts_with_underscore_is_kept(tmp_path):
    """Only folders mark privacy: the rule must not spill onto file names."""
    make_file(tmp_path, "_notes.txt")
    yielded = [path for _, path in iter_files([tmp_path], {".txt"})]
    assert [path.name for path in yielded] == ["_notes.txt"]


def test_a_directory_named_like_a_document_is_never_yielded(tmp_path):
    """rglob walks directories too, and a folder can carry an ingestable suffix."""
    (tmp_path / "Factures.pdf").mkdir()
    make_file(tmp_path, "Factures.pdf/janvier.pdf")
    yielded = [path for _, path in iter_files([tmp_path], {".pdf"})]
    assert [path.name for path in yielded] == ["janvier.pdf"]


def test_missing_root_is_skipped_without_stopping_the_others(tmp_path):
    existing = tmp_path / "drive"
    existing.mkdir()
    make_file(existing, "doc.txt")
    yielded = list(iter_files([tmp_path / "absent", existing], {".txt"}))
    assert [path.name for _, path in yielded] == ["doc.txt"]


def test_each_file_is_paired_with_the_root_it_came_from(tmp_path):
    first, second = tmp_path / "pro", tmp_path / "perso"
    make_file(first, "a.txt")
    make_file(second, "sous/b.txt")
    pairs = {path.name: root for root, path in iter_files([first, second], {".txt"})}
    assert pairs == {"a.txt": first, "b.txt": second}
