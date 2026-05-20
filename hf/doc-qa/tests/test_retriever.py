import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# load_index
# ---------------------------------------------------------------------------

def test_load_index_returns_index_and_metadata(tmp_path):
    import faiss

    # Write a minimal FAISS index and metadata file
    dim = 4
    index = faiss.IndexFlatIP(dim)
    vecs = np.random.rand(2, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index.add(vecs)

    index_path = str(tmp_path / "index.faiss")
    meta_path = str(tmp_path / "metadata.json")
    faiss.write_index(index, index_path)
    metadata = [{"text": "chunk one", "source": "doc.pdf", "chunk_index": 0},
                {"text": "chunk two", "source": "doc.pdf", "chunk_index": 1}]
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    with patch("retriever.config") as mock_cfg:
        mock_cfg.INDEX_PATH = index_path
        mock_cfg.METADATA_PATH = meta_path
        from retriever import load_index
        loaded_index, loaded_meta = load_index()

    assert loaded_index.ntotal == 2
    assert len(loaded_meta) == 2
    assert loaded_meta[0]["text"] == "chunk one"


def test_load_index_raises_when_missing(tmp_path):
    with patch("retriever.config") as mock_cfg:
        mock_cfg.INDEX_PATH = str(tmp_path / "missing.faiss")
        mock_cfg.METADATA_PATH = str(tmp_path / "missing.json")
        from retriever import load_index
        with pytest.raises(FileNotFoundError):
            load_index()


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------

def _make_index(dim: int, n: int):
    import faiss
    index = faiss.IndexFlatIP(dim)
    vecs = np.random.rand(n, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index.add(vecs)
    metadata = [{"text": f"chunk {i}", "source": "doc.pdf", "chunk_index": i} for i in range(n)]
    return index, metadata, vecs


def test_retrieve_returns_top_k_results():
    index, metadata, vecs = _make_index(dim=8, n=5)
    query_vec = np.random.rand(1, 8).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)

    mock_embed_model = MagicMock()
    mock_embed_model.encode.return_value = query_vec

    with patch("retriever.SentenceTransformer", return_value=mock_embed_model):
        from retriever import retrieve
        results = retrieve("any question", mock_embed_model, index, metadata, top_k=3)

    assert len(results) == 3


def test_retrieve_results_contain_text_and_score():
    index, metadata, _ = _make_index(dim=8, n=4)
    query_vec = np.random.rand(1, 8).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)

    mock_embed_model = MagicMock()
    mock_embed_model.encode.return_value = query_vec

    with patch("retriever.SentenceTransformer", return_value=mock_embed_model):
        from retriever import retrieve
        results = retrieve("question", mock_embed_model, index, metadata, top_k=2)

    for r in results:
        assert "text" in r
        assert "score" in r
        assert "source" in r


def test_retrieve_top_k_capped_at_index_size():
    index, metadata, _ = _make_index(dim=8, n=2)
    query_vec = np.random.rand(1, 8).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)

    mock_embed_model = MagicMock()
    mock_embed_model.encode.return_value = query_vec

    with patch("retriever.SentenceTransformer", return_value=mock_embed_model):
        from retriever import retrieve
        results = retrieve("question", mock_embed_model, index, metadata, top_k=10)

    assert len(results) <= 2
