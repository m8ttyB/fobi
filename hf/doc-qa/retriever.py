"""FAISS index loading and chunk retrieval.

Provides two retrieval paths:
- Single-index: load_index / retrieve — used by --ingest single-file flow.
- Multi-index: load_all_indexes / retrieve_multi — used by --chat, which
  searches all documents in the store and merges results by cosine score.
"""

import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer



def load_index(index_path: str, metadata_path: str) -> tuple[faiss.Index, list[dict]]:
    """Load a single FAISS index and its companion metadata file.

    Raises FileNotFoundError with an actionable message if either file is missing.
    """
    try:
        index = faiss.read_index(index_path)
    except Exception:
        raise FileNotFoundError(f"No index found at '{index_path}'. Run --ingest first.")
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"No metadata found at '{metadata_path}'. Run --ingest first.")
    return index, metadata


def load_all_indexes(store_dir: str) -> list[tuple[faiss.Index, list[dict]]]:
    """Load every .faiss / .json pair found in store_dir.

    Orphaned .faiss files with no matching .json are silently skipped.
    Raises FileNotFoundError if the store directory doesn't exist or contains
    no valid index pairs.
    """
    if not os.path.isdir(store_dir):
        raise FileNotFoundError(
            f"No index store found at '{store_dir}'. Run --ingest or --ingest-dir first."
        )
    pairs = []
    for fname in sorted(os.listdir(store_dir)):
        if fname.endswith(".faiss"):
            stem = fname[:-6]
            index_path = os.path.join(store_dir, fname)
            meta_path = os.path.join(store_dir, f"{stem}.json")
            if os.path.exists(meta_path):
                pairs.append(load_index(index_path, meta_path))
    if not pairs:
        raise FileNotFoundError(
            f"No indexes found in '{store_dir}'. Run --ingest or --ingest-dir first."
        )
    return pairs


def retrieve(
    query: str,
    embed_model: SentenceTransformer,
    index: faiss.Index,
    metadata: list[dict],
    top_k: int,
) -> list[dict]:
    """Search a single index and return the top-k chunks by cosine similarity.

    Each returned dict is a copy of the metadata entry with an added 'score' key.
    top_k is capped at the number of vectors in the index.
    """
    query_vec = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True)

    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = dict(metadata[idx])
        entry["score"] = float(score)
        results.append(entry)
    return results


def retrieve_multi(
    query: str,
    embed_model: SentenceTransformer,
    indexes: list[tuple[faiss.Index, list[dict]]],
    top_k: int,
) -> list[dict]:
    """Search all indexes, merge results, and return the global top-k by cosine score.

    Scores are directly comparable across indexes because all embeddings are
    L2-normalized — inner product equals cosine similarity regardless of which
    document a chunk came from.
    """
    query_vec = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True)

    all_results = []
    for index, metadata in indexes:
        k = min(top_k, index.ntotal)
        scores, indices = index.search(query_vec, k)
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(metadata[idx])
            entry["score"] = float(score)
            all_results.append(entry)

    all_results.sort(key=lambda r: r["score"], reverse=True)
    return all_results[:top_k]
