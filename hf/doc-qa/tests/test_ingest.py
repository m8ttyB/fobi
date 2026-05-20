import json
import os
import numpy as np
from unittest.mock import patch, MagicMock

from ingest import chunk_text, index_name_for
import config


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

def test_chunk_text_single_paragraph():
    text = "This is a single paragraph with some content."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_splits_on_double_newline():
    para = ("word " * 100).strip()
    text = para + "\n\n" + para + "\n\n" + para
    chunks = chunk_text(text)
    assert len(chunks) >= 2


def test_chunk_text_merges_short_paragraphs():
    parts = ["Short." for _ in range(5)]
    text = "\n\n".join(parts)
    chunks = chunk_text(text)
    assert len(chunks) < 5


def test_chunk_text_respects_max_chunk_chars():
    long_para = "word " * 300
    text = long_para + "\n\n" + long_para
    chunks = chunk_text(text)
    for chunk in chunks:
        assert len(chunk) <= config.MAX_CHUNK_CHARS + config.OVERLAP_CHARS


def test_chunk_text_adds_overlap():
    para_a = "Alpha " * 100
    para_b = "Beta " * 100
    text = para_a.strip() + "\n\n" + para_b.strip()
    chunks = chunk_text(text)
    if len(chunks) > 1:
        assert "Alpha" in chunks[1]


def test_chunk_text_strips_empty_paragraphs():
    text = "First.\n\n\n\n\nSecond."
    chunks = chunk_text(text)
    for chunk in chunks:
        assert chunk.strip() != ""


# ---------------------------------------------------------------------------
# embed_chunks
# ---------------------------------------------------------------------------

def test_embed_chunks_returns_numpy_array():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    with patch("ingest.SentenceTransformer", return_value=mock_model):
        from ingest import embed_chunks
        result = embed_chunks(["chunk one", "chunk two"], "all-MiniLM-L6-v2")
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)


def test_embed_chunks_normalizes_vectors():
    raw = np.array([[3.0, 4.0]])
    mock_model = MagicMock()
    mock_model.encode.return_value = raw
    with patch("ingest.SentenceTransformer", return_value=mock_model):
        from ingest import embed_chunks
        result = embed_chunks(["anything"], "all-MiniLM-L6-v2")
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# save_index
# ---------------------------------------------------------------------------

def test_save_index_writes_files(tmp_path):
    embeddings = np.random.rand(3, 4).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    chunks = ["chunk one", "chunk two", "chunk three"]

    index_path = str(tmp_path / "index.faiss")
    meta_path = str(tmp_path / "metadata.json")

    from ingest import save_index
    save_index(embeddings, chunks, "test.pdf", index_path=index_path, metadata_path=meta_path)

    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "metadata.json").exists()


def test_save_index_metadata_shape(tmp_path):
    embeddings = np.random.rand(2, 4).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    chunks = ["first chunk", "second chunk"]

    index_path = str(tmp_path / "index.faiss")
    meta_path = str(tmp_path / "metadata.json")

    from ingest import save_index
    save_index(embeddings, chunks, "doc.pdf", index_path=index_path, metadata_path=meta_path)

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert len(metadata) == 2
    assert metadata[0]["text"] == "first chunk"
    assert metadata[0]["source"] == "doc.pdf"
    assert metadata[0]["chunk_index"] == 0


# ---------------------------------------------------------------------------
# index_name_for
# ---------------------------------------------------------------------------

def test_index_name_for_flat_file(tmp_path):
    pdf = str(tmp_path / "report.pdf")
    name = index_name_for(pdf, str(tmp_path))
    assert name == "report"


def test_index_name_for_nested_file(tmp_path):
    subdir = tmp_path / "reports"
    subdir.mkdir()
    pdf = str(subdir / "q1.pdf")
    name = index_name_for(pdf, str(tmp_path))
    assert name == f"reports{os.sep.replace(os.sep, '__')}q1" or name == "reports__q1"


def test_index_name_for_no_extension_collision(tmp_path):
    pdf_a = str(tmp_path / "doc.pdf")
    pdf_b = str(tmp_path / "subdir" / "doc.pdf")
    os.makedirs(str(tmp_path / "subdir"))
    name_a = index_name_for(pdf_a, str(tmp_path))
    name_b = index_name_for(pdf_b, str(tmp_path))
    assert name_a != name_b
