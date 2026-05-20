# doc-qa — RAG CLI Implementation Plan

## Context

`doc-qa` is the third project in the `hf/` learning series. It introduces Retrieval-Augmented Generation (RAG): ingesting a document once into a vector index, then answering questions against it by retrieving only the most relevant chunks before generation. This keeps the context window focused and grounds the model's answers in the document.

The fetch-and-extract pipeline from `writing-assistant` established the document acquisition pattern. `doc-qa` builds on that by adding chunking, embedding, and vector search before generation.

## Design decisions (from design discussion)

- **Interface:** unified CLI entry point (`main.py`) with argparse flags
- **Chunking:** paragraph-aware with sentence-level overlap
- **Embedding:** `sentence-transformers` (local, CPU-friendly, ~90 MB)
- **Vector store:** FAISS with cosine similarity; parallel JSON file for chunk text and metadata
- **Generator:** Gemma via `mlx-lm` (same as prior projects)
- **Chat mode:** REPL loop with accumulating history (same pattern as `cli-chat`)
- **Context budget:** retrieved chunks are fixed-size; history is trimmed first when context fills
- **Reset:** `--reset` wipes the FAISS index and metadata with a confirmation prompt

## CLI surface

```bash
python main.py --ingest path/to/document.pdf   # chunk, embed, save index to disk
python main.py --chat                           # REPL loop with document-grounded answers
python main.py --reset                          # clear index and metadata (with confirmation)
python main.py --help                           # usage
```

## Architecture

Six modules:

| Module | Responsibility |
|---|---|
| `config.py` | Env-overridable constants only |
| `ingest.py` | Load PDF, chunk, embed, save FAISS index + metadata |
| `retriever.py` | Load index, embed query, return top-k chunks |
| `prompts.py` | Build messages list with retrieved context for `apply_chat_template` |
| `model.py` | Thin `mlx-lm` wrapper — `load_model`, `stream_response` (yields `(text, chunk)` tuples) |
| `main.py` | argparse entry point; wires modules together for each flag |

### `config.py`

```python
MODEL_PATH     = os.environ.get("QA_MODEL_PATH", "mlx-community/gemma-3-4b-it-4bit")
EMBED_MODEL    = os.environ.get("QA_EMBED_MODEL", "all-MiniLM-L6-v2")
INDEX_PATH     = os.environ.get("QA_INDEX_PATH", "index.faiss")
METADATA_PATH  = os.environ.get("QA_METADATA_PATH", "metadata.json")
TOP_K          = int(os.environ.get("QA_TOP_K", "4"))
MAX_CHUNK_CHARS = 800
OVERLAP_CHARS   = 150
MAX_HISTORY_TURNS = 6
```

### `ingest.py`

```python
def chunk_text(text: str) -> list[str]:
    # split on paragraph boundaries (\n\n)
    # merge short paragraphs until MAX_CHUNK_CHARS
    # add OVERLAP_CHARS of previous chunk to each chunk

def embed_chunks(chunks: list[str], model_name: str) -> np.ndarray:
    # sentence-transformers encode(), normalize for cosine similarity

def save_index(embeddings: np.ndarray, chunks: list[str], source: str) -> None:
    # faiss.IndexFlatIP (inner product on normalized vectors = cosine similarity)
    # write index to INDEX_PATH
    # write [{text, source, chunk_index}] to METADATA_PATH
```

### `retriever.py`

```python
def load_index() -> tuple[faiss.Index, list[dict]]:
    # read INDEX_PATH and METADATA_PATH

def retrieve(query: str, embed_model, index, metadata, top_k: int) -> list[dict]:
    # embed query, normalize
    # index.search() → top_k nearest
    # return list of metadata dicts with scores
```

### `prompts.py`

```python
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a document.
Base your answer only on the provided context. If the answer is not in the context, say so."""

def build_messages(history: list[dict], question: str, chunks: list[dict]) -> list[dict]:
    # format retrieved chunks as context block
    # return [system, ...history, user_with_context]
```

### Context prompt shape

```
[system: answer only from context]
[assistant/user history turns (trimmed if needed)]
[user: Context:
  --- Chunk 1 ---
  <text>
  --- Chunk 2 ---
  <text>
  ...
  Question: <question>]
```

### `model.py`

Identical pattern to `writing-assistant/model.py` — `load_model(path)` and `stream_response(model, tokenizer, messages)` yielding `(text, chunk)` tuples. Copy and adapt.

### `main.py`

```python
# argparse with mutually exclusive group: --ingest FILE | --chat | --reset
# --ingest: load PDF with pypdf, extract text, call ingest.chunk_text + ingest.save_index
# --chat:   load model + index, REPL loop
# --reset:  prompt "Delete index and metadata? [y/N]", delete files if confirmed
```

## REPL loop (--chat)

```
Load model (mlx-lm)
Load index + metadata (FAISS)
history = []

loop:
  prompt "> "
  if input == "/exit": break
  if input == "/clear": history = []; continue

  chunks = retrieve(question, ...)
  messages = build_messages(history, question, chunks)
  stream response token by token
  append (question, response) to history
  trim history to MAX_HISTORY_TURNS
```

## Data layout on disk

```
doc-qa/
├── index.faiss       # FAISS index (binary, written by ingest)
├── metadata.json     # [{text, source, chunk_index}, ...]
```

Both files are gitignored — they're derived artifacts.

## Dependencies

```
mlx-lm>=0.19.0
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
pypdf>=4.0.0
numpy>=1.26.0
pytest>=8.0.0
ruff>=0.4.0
```

## New files

| File | Purpose |
|---|---|
| `config.py` | Constants |
| `ingest.py` | Chunking, embedding, index persistence |
| `retriever.py` | Index loading, query embedding, top-k search |
| `prompts.py` | RAG-aware message builder |
| `model.py` | mlx-lm wrapper (adapted from writing-assistant) |
| `main.py` | CLI entry point |
| `Makefile` | `install`, `ingest`, `chat`, `reset`, `test`, `lint`, `format` |
| `requirements.txt` | Dependencies |
| `tests/test_chunker.py` | Unit tests for `chunk_text` |
| `tests/test_retriever.py` | Unit tests for `retrieve` with mocked FAISS |
| `tests/test_prompts.py` | Unit tests for `build_messages` |
| `tests/test_ingest.py` | Unit tests for `embed_chunks`, `save_index` with mocked sentence-transformers |
| `CLAUDE.md` | Architecture, setup, commands, key design decisions |
| `README.md` | User-facing docs |
| `.gitignore` | Exclude `index.faiss`, `metadata.json`, `.venv` |

## Makefile targets

```makefile
make install
make ingest FILE=path/to/doc.pdf
make chat
make reset
make test
make lint
make format
```

## Implementation order

1. `requirements.txt` + `make install`
2. `config.py`
3. Tests for `ingest.py` (mocked sentence-transformers + FAISS) → fail
4. `ingest.py` → tests pass
5. Tests for `retriever.py` (mocked FAISS + metadata) → fail
6. `retriever.py` → tests pass
7. Tests for `prompts.py` → fail
8. `prompts.py` → tests pass
9. `model.py` (adapt from writing-assistant)
10. `main.py` — wire everything together
11. `Makefile`, `CLAUDE.md`, `README.md`, `.gitignore`
12. Manual smoke test: ingest a real PDF, ask questions in `--chat`
13. Commit

## Verification

- `make test` — all unit tests pass (no model or FAISS index required — dependencies mocked)
- `make lint` — ruff clean
- Manual smoke tests:
  - `make ingest FILE=<pdf>` → index and metadata files created
  - `make chat` → REPL starts, question answered with content from document
  - Follow-up question ("tell me more") resolved via history
  - `/clear` resets history mid-session
  - `/exit` exits cleanly
  - `make reset` → confirmation prompt; index deleted on `y`, cancelled on `N`
  - `make chat` after reset → clear error: no index found, run ingest first

## Out of scope (v1)

- Multiple documents in one index
- Re-ingesting without full reset
- Hybrid search (keyword + semantic)
- Answer citations (which chunk the answer came from)
- Web UI
