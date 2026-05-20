import os
import json
import numpy as np
from unittest.mock import patch, MagicMock

import faiss


def _make_pdf_stub(tmp_path, name="doc.pdf"):
    """Write an empty file as a PDF placeholder — PdfReader is mocked in all tests."""
    path = str(tmp_path / name)
    open(path, "wb").close()
    return path


def _mock_pdf_reader(text="This is sample text from the document."):
    """Return a mock PdfReader whose pages yield the given text."""
    page = MagicMock()
    page.extract_text.return_value = text
    reader = MagicMock()
    reader.pages = [page]
    return reader


def _mock_embed(dim=4):
    model = MagicMock()
    model.encode.return_value = np.random.rand(1, dim).astype(np.float32)
    return model


# ---------------------------------------------------------------------------
# cmd_ingest_dir — prompting behavior
# ---------------------------------------------------------------------------

def test_ingest_dir_skips_when_user_says_no(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    _make_pdf_stub(pdf_dir, "report.pdf")

    # Pre-write a stub index so the "already indexed" branch triggers
    (store / "report.faiss").write_bytes(b"stub")
    (store / "report.json").write_text(json.dumps([]))
    mtime_before = os.path.getmtime(str(store / "report.faiss"))

    with patch("ingest.PdfReader", return_value=_mock_pdf_reader()), \
         patch("ingest.SentenceTransformer", return_value=_mock_embed()), \
         patch("builtins.input", return_value="n"):
        from ingest import ingest_directory
        ingest_directory(str(pdf_dir), str(store), "all-MiniLM-L6-v2")

    mtime_after = os.path.getmtime(str(store / "report.faiss"))
    assert mtime_before == mtime_after


def test_ingest_dir_overwrites_when_user_says_yes(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    _make_pdf_stub(pdf_dir, "report.pdf")

    (store / "report.faiss").write_bytes(b"old")
    (store / "report.json").write_text(json.dumps([]))

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

    with patch("ingest.PdfReader", return_value=_mock_pdf_reader()), \
         patch("ingest.SentenceTransformer", return_value=mock_model), \
         patch("builtins.input", return_value="y"):
        from ingest import ingest_directory
        ingest_directory(str(pdf_dir), str(store), "all-MiniLM-L6-v2")

    # A real FAISS index was written — stub bytes are gone
    loaded = faiss.read_index(str(store / "report.faiss"))
    assert loaded.ntotal >= 1


def test_ingest_dir_finds_pdfs_recursively(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    pdf_dir = tmp_path / "pdfs"
    subdir = pdf_dir / "sub"
    subdir.mkdir(parents=True)
    _make_pdf_stub(pdf_dir, "top.pdf")
    _make_pdf_stub(subdir, "nested.pdf")

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

    with patch("ingest.PdfReader", return_value=_mock_pdf_reader()), \
         patch("ingest.SentenceTransformer", return_value=mock_model):
        from ingest import ingest_directory
        ingest_directory(str(pdf_dir), str(store), "all-MiniLM-L6-v2")

    faiss_files = [f for f in os.listdir(str(store)) if f.endswith(".faiss")]
    assert len(faiss_files) == 2
