import json
import numpy as np
import pytest
from unittest.mock import MagicMock

import faiss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_index(tmp_path, name: str, dim: int, n: int, source: str):
    """Write a FAISS index + metadata JSON pair to tmp_path."""
    index = faiss.IndexFlatIP(dim)
    vecs = np.random.rand(n, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index.add(vecs)
    faiss.write_index(index, str(tmp_path / f"{name}.faiss"))
    metadata = [{"text": f"{source} chunk {i}", "source": source, "chunk_index": i} for i in range(n)]
    (tmp_path / f"{name}.json").write_text(json.dumps(metadata))
    return index, metadata, vecs


def _mock_embed(query_vec):
    model = MagicMock()
    model.encode.return_value = query_vec
    return model


# ---------------------------------------------------------------------------
# load_index
# ---------------------------------------------------------------------------

def test_load_index_returns_index_and_metadata(tmp_path):
    _write_index(tmp_path, "doc", dim=4, n=2, source="doc.pdf")
    from retriever import load_index
    index, meta = load_index(str(tmp_path / "doc.faiss"), str(tmp_path / "doc.json"))
    assert index.ntotal == 2
    assert len(meta) == 2


def test_load_index_raises_when_missing(tmp_path):
    from retriever import load_index
    with pytest.raises(FileNotFoundError):
        load_index(str(tmp_path / "missing.faiss"), str(tmp_path / "missing.json"))


# ---------------------------------------------------------------------------
# load_all_indexes
# ---------------------------------------------------------------------------

def test_load_all_indexes_finds_all_pairs(tmp_path):
    _write_index(tmp_path, "a", dim=4, n=2, source="a.pdf")
    _write_index(tmp_path, "b", dim=4, n=3, source="b.pdf")
    from retriever import load_all_indexes
    pairs = load_all_indexes(str(tmp_path))
    assert len(pairs) == 2
    totals = sorted(idx.ntotal for idx, _ in pairs)
    assert totals == [2, 3]


def test_load_all_indexes_raises_when_store_missing(tmp_path):
    from retriever import load_all_indexes
    with pytest.raises(FileNotFoundError):
        load_all_indexes(str(tmp_path / "nonexistent"))


def test_load_all_indexes_raises_when_store_empty(tmp_path):
    from retriever import load_all_indexes
    with pytest.raises(FileNotFoundError):
        load_all_indexes(str(tmp_path))


def test_load_all_indexes_ignores_faiss_without_json(tmp_path):
    # Write only a .faiss file with no matching .json — should be ignored
    _write_index(tmp_path, "good", dim=4, n=2, source="good.pdf")
    orphan = faiss.IndexFlatIP(4)
    faiss.write_index(orphan, str(tmp_path / "orphan.faiss"))
    from retriever import load_all_indexes
    pairs = load_all_indexes(str(tmp_path))
    assert len(pairs) == 1


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------

def test_retrieve_returns_top_k_results(tmp_path):
    index, metadata, _ = _write_index(tmp_path, "doc", dim=8, n=5, source="doc.pdf")
    query_vec = np.random.rand(1, 8).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)
    from retriever import retrieve
    results = retrieve("question", _mock_embed(query_vec), index, metadata, top_k=3)
    assert len(results) == 3


def test_retrieve_results_contain_required_fields(tmp_path):
    index, metadata, _ = _write_index(tmp_path, "doc", dim=8, n=4, source="doc.pdf")
    query_vec = np.random.rand(1, 8).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)
    from retriever import retrieve
    results = retrieve("question", _mock_embed(query_vec), index, metadata, top_k=2)
    for r in results:
        assert "text" in r
        assert "score" in r
        assert "source" in r


def test_retrieve_top_k_capped_at_index_size(tmp_path):
    index, metadata, _ = _write_index(tmp_path, "doc", dim=8, n=2, source="doc.pdf")
    query_vec = np.random.rand(1, 8).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)
    from retriever import retrieve
    results = retrieve("question", _mock_embed(query_vec), index, metadata, top_k=10)
    assert len(results) <= 2


# ---------------------------------------------------------------------------
# retrieve_multi
# ---------------------------------------------------------------------------

def test_retrieve_multi_merges_results(tmp_path):
    idx_a, meta_a, _ = _write_index(tmp_path, "a", dim=8, n=5, source="a.pdf")
    idx_b, meta_b, _ = _write_index(tmp_path, "b", dim=8, n=5, source="b.pdf")
    query_vec = np.random.rand(1, 8).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)
    from retriever import retrieve_multi
    results = retrieve_multi("question", _mock_embed(query_vec), [(idx_a, meta_a), (idx_b, meta_b)], top_k=4)
    assert len(results) == 4


def test_retrieve_multi_sorted_by_score(tmp_path):
    idx_a, meta_a, _ = _write_index(tmp_path, "a", dim=8, n=5, source="a.pdf")
    idx_b, meta_b, _ = _write_index(tmp_path, "b", dim=8, n=5, source="b.pdf")
    query_vec = np.random.rand(1, 8).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)
    from retriever import retrieve_multi
    results = retrieve_multi("question", _mock_embed(query_vec), [(idx_a, meta_a), (idx_b, meta_b)], top_k=6)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_multi_can_pull_from_multiple_sources(tmp_path):
    idx_a, meta_a, _ = _write_index(tmp_path, "a", dim=8, n=3, source="a.pdf")
    idx_b, meta_b, _ = _write_index(tmp_path, "b", dim=8, n=3, source="b.pdf")
    query_vec = np.random.rand(1, 8).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)
    from retriever import retrieve_multi
    results = retrieve_multi("question", _mock_embed(query_vec), [(idx_a, meta_a), (idx_b, meta_b)], top_k=6)
    sources = {r["source"] for r in results}
    assert len(sources) == 2


# ---------------------------------------------------------------------------
# min_score thresholding
# ---------------------------------------------------------------------------

def test_retrieve_min_score_filters_low_scores(tmp_path):
    index, metadata, _ = _write_index(tmp_path, "doc", dim=8, n=5, source="doc.pdf")
    query_vec = np.random.rand(1, 8).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)
    from retriever import retrieve
    # A threshold of 1.0 is only satisfied by an exact match — should return nothing
    results = retrieve("question", _mock_embed(query_vec), index, metadata, top_k=5, min_score=1.0)
    assert results == []


def test_retrieve_min_score_zero_returns_all(tmp_path):
    index, metadata, _ = _write_index(tmp_path, "doc", dim=8, n=4, source="doc.pdf")
    query_vec = np.random.rand(1, 8).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)
    from retriever import retrieve
    results = retrieve("question", _mock_embed(query_vec), index, metadata, top_k=4, min_score=0.0)
    assert len(results) == 4


def test_retrieve_multi_min_score_filters_low_scores(tmp_path):
    idx_a, meta_a, _ = _write_index(tmp_path, "a", dim=8, n=3, source="a.pdf")
    idx_b, meta_b, _ = _write_index(tmp_path, "b", dim=8, n=3, source="b.pdf")
    query_vec = np.random.rand(1, 8).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)
    from retriever import retrieve_multi
    results = retrieve_multi("question", _mock_embed(query_vec), [(idx_a, meta_a), (idx_b, meta_b)], top_k=6, min_score=1.0)
    assert results == []


def test_retrieve_multi_min_score_applied_before_top_k(tmp_path):
    # With min_score=1.0 (impossible threshold), top_k=4 should return 0, not 4
    index, metadata, _ = _write_index(tmp_path, "doc", dim=8, n=6, source="doc.pdf")
    query_vec = np.random.rand(1, 8).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)
    from retriever import retrieve_multi
    results = retrieve_multi("question", _mock_embed(query_vec), [(index, metadata)], top_k=4, min_score=1.0)
    assert len(results) == 0


def test_retrieve_multi_min_score_zero_default_unchanged(tmp_path):
    idx_a, meta_a, _ = _write_index(tmp_path, "a", dim=8, n=3, source="a.pdf")
    query_vec = np.random.rand(1, 8).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec)
    from retriever import retrieve_multi
    # Default min_score=0.0 should not filter anything
    results = retrieve_multi("question", _mock_embed(query_vec), [(idx_a, meta_a)], top_k=3)
    assert len(results) == 3
