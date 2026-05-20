import json
import numpy as np
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

from ingest import chunk_text
import config


def test_chunk_text_single_paragraph():
    text = "This is a single paragraph with some content."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_splits_on_double_newline():
    # Each paragraph is ~500 chars — two together exceed MAX_CHUNK_CHARS (800)
    # so they cannot all be merged into one chunk
    para = ("word " * 100).strip()
    text = para + "\n\n" + para + "\n\n" + para
    chunks = chunk_text(text)
    assert len(chunks) >= 2


def test_chunk_text_merges_short_paragraphs():
    # Each paragraph is short; they should be merged into fewer chunks
    parts = ["Short." for _ in range(5)]
    text = "\n\n".join(parts)
    chunks = chunk_text(text)
    assert len(chunks) < 5


def test_chunk_text_respects_max_chunk_chars():
    long_para = "word " * 300  # ~1500 chars, well over MAX_CHUNK_CHARS
    text = long_para + "\n\n" + long_para
    chunks = chunk_text(text)
    for chunk in chunks:
        assert len(chunk) <= config.MAX_CHUNK_CHARS + config.OVERLAP_CHARS


def test_chunk_text_adds_overlap():
    # Two paragraphs that together exceed MAX_CHUNK_CHARS should produce
    # chunks where the second chunk starts with content from the first
    para_a = "Alpha " * 100   # ~600 chars
    para_b = "Beta " * 100    # ~600 chars
    text = para_a.strip() + "\n\n" + para_b.strip()
    chunks = chunk_text(text)
    if len(chunks) > 1:
        # The second chunk should contain some overlap from the first
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
    # Normalized vectors should have unit length (L2 norm ~= 1.0)
    raw = np.array([[3.0, 4.0]])  # norm = 5.0
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
    # normalize
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    chunks = ["chunk one", "chunk two", "chunk three"]
    source = "test.pdf"

    index_path = str(tmp_path / "index.faiss")
    meta_path = str(tmp_path / "metadata.json")

    with patch("ingest.config") as mock_cfg:
        mock_cfg.INDEX_PATH = index_path
        mock_cfg.METADATA_PATH = meta_path
        from ingest import save_index
        save_index(embeddings, chunks, source)

    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "metadata.json").exists()


def test_save_index_metadata_shape(tmp_path):
    embeddings = np.random.rand(2, 4).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    chunks = ["first chunk", "second chunk"]

    index_path = str(tmp_path / "index.faiss")
    meta_path = str(tmp_path / "metadata.json")

    with patch("ingest.config") as mock_cfg:
        mock_cfg.INDEX_PATH = index_path
        mock_cfg.METADATA_PATH = meta_path
        from ingest import save_index
        save_index(embeddings, chunks, "doc.pdf")

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert len(metadata) == 2
    assert metadata[0]["text"] == "first chunk"
    assert metadata[0]["source"] == "doc.pdf"
    assert metadata[0]["chunk_index"] == 0
