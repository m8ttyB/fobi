import os

MODEL_PATH    = os.getenv("EXTRACTOR_MODEL_PATH", "mlx-community/gemma-3-4b-it-4bit")
MAX_CHARS     = int(os.getenv("EXTRACTOR_MAX_CHARS", "20000"))
MAX_RETRIES   = int(os.getenv("EXTRACTOR_MAX_RETRIES", "3"))
CHUNK_CHARS   = int(os.getenv("EXTRACTOR_CHUNK_CHARS", "4000"))
OVERLAP_CHARS = int(os.getenv("EXTRACTOR_OVERLAP_CHARS", "400"))
