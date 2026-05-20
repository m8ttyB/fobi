import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config


def load_index() -> tuple[faiss.Index, list[dict]]:
    try:
        index = faiss.read_index(config.INDEX_PATH)
    except Exception:
        raise FileNotFoundError(
            f"No index found at '{config.INDEX_PATH}'. Run --ingest first."
        )
    try:
        with open(config.METADATA_PATH) as f:
            metadata = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No metadata found at '{config.METADATA_PATH}'. Run --ingest first."
        )
    return index, metadata


def retrieve(
    query: str,
    embed_model: SentenceTransformer,
    index: faiss.Index,
    metadata: list[dict],
    top_k: int,
) -> list[dict]:
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
