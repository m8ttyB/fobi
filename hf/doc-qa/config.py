"""Runtime configuration loaded from environment variables.

All values have sensible defaults so the tool works out of the box.
Override any setting by exporting the corresponding env var before running.
"""

import os

MODEL_PATH = os.environ.get("QA_MODEL_PATH", "mlx-community/gemma-3-4b-it-4bit")
EMBED_MODEL = os.environ.get("QA_EMBED_MODEL", "all-MiniLM-L6-v2")
STORE_DIR = os.environ.get("QA_STORE_DIR", "doc_store")
TOP_K = int(os.environ.get("QA_TOP_K", "4"))
MAX_CHUNK_CHARS = 800
OVERLAP_CHARS = 150
MAX_HISTORY_TURNS = 6
