"""Runtime configuration loaded from environment variables.

All values have sensible defaults so the tool works out of the box.
Override any setting by exporting the corresponding env var before running.

Retrieval tuning:
    QA_TOP_K controls how many chunks are retrieved per query.
    QA_MIN_SCORE filters out chunks below a cosine similarity threshold,
    preventing low-quality context from reaching the model.
"""

import os

MODEL_PATH = os.environ.get("QA_MODEL_PATH", "mlx-community/gemma-3-4b-it-4bit")
EMBED_MODEL = os.environ.get("QA_EMBED_MODEL", "all-MiniLM-L6-v2")
STORE_DIR = os.environ.get("QA_STORE_DIR", "doc_store")
TOP_K = int(os.environ.get("QA_TOP_K", "4"))
MIN_SCORE = float(os.environ.get("QA_MIN_SCORE", "0.3"))
MAX_CHUNK_CHARS = 800
OVERLAP_CHARS = 150
MAX_HISTORY_TURNS = 6
