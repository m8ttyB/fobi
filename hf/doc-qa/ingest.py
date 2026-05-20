import json
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config


def _split_oversized(para: str) -> list[str]:
    """Split a paragraph that exceeds MAX_CHUNK_CHARS at word boundaries with overlap."""
    pieces = []
    while len(para) > config.MAX_CHUNK_CHARS:
        split_at = para.rfind(" ", 0, config.MAX_CHUNK_CHARS)
        if split_at == -1:
            split_at = config.MAX_CHUNK_CHARS
        pieces.append(para[:split_at].strip())
        overlap_start = max(0, split_at - config.OVERLAP_CHARS)
        para = para[overlap_start:].strip()
    if para:
        pieces.append(para)
    return pieces


def chunk_text(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # A single paragraph that exceeds the limit gets split immediately
        if len(para) > config.MAX_CHUNK_CHARS:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_oversized(para))
            continue

        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= config.MAX_CHUNK_CHARS:
            current = candidate
        else:
            chunks.append(current)
            overlap = current[-config.OVERLAP_CHARS:] if len(current) > config.OVERLAP_CHARS else current
            current = (overlap + "\n\n" + para).strip()

    if current:
        chunks.append(current)

    return chunks


def embed_chunks(chunks: list[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True).astype(np.float32)
    # normalize to unit length so inner product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


def save_index(embeddings: np.ndarray, chunks: list[str], source: str) -> None:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, config.INDEX_PATH)

    metadata = [
        {"text": chunk, "source": source, "chunk_index": i}
        for i, chunk in enumerate(chunks)
    ]
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
