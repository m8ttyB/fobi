"""Document ingestion pipeline: chunking, embedding, and FAISS index persistence.

Typical call sequence:
    chunks = chunk_text(raw_text)
    embeddings = embed_chunks(chunks, model_name)
    save_index(embeddings, chunks, source=path, index_path=..., metadata_path=...)

For directory ingestion use ingest_directory(), which handles discovery,
per-file prompting on overwrites, and calls the above sequence for each PDF.
"""

import json
import os
import re

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import config


def _split_oversized(para: str) -> list[str]:
    """Split a single paragraph that exceeds MAX_CHUNK_CHARS at word boundaries.

    Each piece overlaps with the previous by OVERLAP_CHARS characters so that
    ideas near a split point appear in full in at least one chunk.
    """
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
    """Split document text into overlapping paragraph-aware chunks.

    Paragraphs (double-newline delimited) are the primary boundary. Short
    paragraphs are merged until they approach MAX_CHUNK_CHARS; oversized
    paragraphs are split at word boundaries. Adjacent chunks share
    OVERLAP_CHARS characters so that ideas near a boundary appear in context
    in at least one chunk.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
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
    """Encode chunks into L2-normalized float32 embeddings.

    Normalization to unit length means inner product == cosine similarity,
    which is what FAISS IndexFlatIP computes.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


def save_index(
    embeddings: np.ndarray,
    chunks: list[str],
    source: str,
    index_path: str,
    metadata_path: str,
) -> None:
    """Persist embeddings to a FAISS index and chunk text to a JSON metadata file.

    Args:
        embeddings: L2-normalized float32 array of shape (n_chunks, dim).
        chunks: Chunk text strings in the same order as embeddings.
        source: Original file path stored in each metadata entry for citation.
        index_path: Destination path for the .faiss binary file.
        metadata_path: Destination path for the .json metadata file.
    """
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)

    metadata = [
        {"text": chunk, "source": source, "chunk_index": i}
        for i, chunk in enumerate(chunks)
    ]
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def index_name_for(pdf_path: str, base_dir: str) -> str:
    """Derive a collision-safe index filename stem from a PDF path.

    Relative path segments are joined with '__' so that
    base_dir/reports/q1.pdf becomes 'reports__q1'.
    """
    abs_pdf = os.path.realpath(pdf_path)
    abs_base = os.path.realpath(base_dir)
    try:
        rel = os.path.relpath(abs_pdf, abs_base)
    except ValueError:
        # On Windows, relpath can fail across drives — fall back to basename
        rel = os.path.basename(pdf_path)
    stem = os.path.splitext(rel)[0]
    return stem.replace(os.sep, "__")


def ingest_directory(dir_path: str, store_dir: str, embed_model_name: str) -> None:
    """Recursively ingest all PDFs found under dir_path into store_dir.

    For each PDF that already has an index, the user is prompted to skip or
    overwrite. PDFs that produce no extractable text are skipped with a warning.
    """

    pdf_paths = []
    for root, _, files in os.walk(dir_path):
        for fname in sorted(files):
            if fname.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, fname))

    if not pdf_paths:
        print(f"No PDF files found in '{dir_path}'.")
        return

    print(f"Found {len(pdf_paths)} PDF(s).")
    os.makedirs(store_dir, exist_ok=True)

    for pdf_path in pdf_paths:
        stem = index_name_for(pdf_path, dir_path)
        index_path = os.path.join(store_dir, f"{stem}.faiss")
        metadata_path = os.path.join(store_dir, f"{stem}.json")

        if os.path.exists(index_path):
            answer = input(f"'{os.path.basename(pdf_path)}' is already indexed. Overwrite? [y/N] ").strip().lower()
            if answer != "y":
                print(f"  Skipping {os.path.basename(pdf_path)}")
                continue

        print(f"  Ingesting {os.path.basename(pdf_path)}...")
        reader = PdfReader(pdf_path)
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)

        if not text.strip():
            print(f"  Warning: no text extracted from '{pdf_path}' — skipping.")
            continue

        chunks = chunk_text(text)
        print(f"    {len(chunks)} chunks")

        embeddings = embed_chunks(chunks, embed_model_name)
        save_index(embeddings, chunks, source=pdf_path, index_path=index_path, metadata_path=metadata_path)

    print("Done.")
