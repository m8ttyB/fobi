import os

MODEL_PATH = os.environ.get("QA_MODEL_PATH", "mlx-community/gemma-3-4b-it-4bit")
EMBED_MODEL = os.environ.get("QA_EMBED_MODEL", "all-MiniLM-L6-v2")
INDEX_PATH = os.environ.get("QA_INDEX_PATH", "index.faiss")
METADATA_PATH = os.environ.get("QA_METADATA_PATH", "metadata.json")
TOP_K = int(os.environ.get("QA_TOP_K", "4"))
MAX_CHUNK_CHARS = 800
OVERLAP_CHARS = 150
MAX_HISTORY_TURNS = 6
