# doc-qa

A CLI tool for asking questions about documents using Retrieval-Augmented Generation (RAG). Ingest one or more PDFs to build a local vector index, then ask questions in a REPL session — the model answers using only content from your documents, with citations showing which files each answer came from.

## Setup

```bash
cd hf/doc-qa
make install
```

## Quick start

```bash
# Ingest a single document
make ingest FILE=path/to/document.pdf

# Or ingest a whole directory of PDFs (recursive)
make ingest-dir DIR=path/to/pdfs/

# Ask questions
make chat
```

## Make targets

| Target | Description |
|---|---|
| `make install` | Create `.venv` and install all dependencies |
| `make ingest FILE=...` | Chunk, embed, and index a single PDF |
| `make ingest-dir DIR=...` | Ingest all PDFs in a directory (recursive) |
| `make chat` | Start a Q&A REPL session |
| `make reset` | Delete all indexes (with confirmation) |
| `make test` | Run the test suite |
| `make lint` | Check code with ruff |
| `make format` | Auto-format code with ruff |

`MODEL` can be overridden:

```bash
make ingest FILE=doc.pdf MODEL=mlx-community/gemma-3-4b-it-4bit
make ingest-dir DIR=pdfs/ MODEL=mlx-community/gemma-3-4b-it-4bit
make chat MODEL=mlx-community/gemma-3-4b-it-4bit
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `QA_MODEL_PATH` | `mlx-community/gemma-3-4b-it-4bit` | HuggingFace repo ID or local MLX model directory |
| `QA_EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model for embedding |
| `QA_STORE_DIR` | `doc_store` | Directory where per-document indexes are stored |
| `QA_TOP_K` | `4` | Number of chunks to retrieve per question |
| `QA_MIN_SCORE` | `0.3` | Minimum cosine similarity score for a chunk to be included |

## Chat commands

| Command | Description |
|---|---|
| `/exit` | Quit the session |
| `/clear` | Reset conversation history |

**Keyboard shortcuts** (macOS / Linux):

| Key | Action |
|---|---|
| `↑` / `↓` | Scroll through question history |
| `←` / `→` | Move cursor within the current line |
| `Ctrl+A` | Jump to start of line |
| `Ctrl+E` | Jump to end of line |
| `Ctrl+R` | Reverse search through history |

History navigation is available on macOS and Linux via the `readline` module. On Windows, plain input is used and arrow keys are not supported.

If no chunks pass the relevance threshold for a question, the model is not called and a message is shown instead:

```
No relevant content found for that question. Try rephrasing, or check that
the relevant document has been ingested.
```

Lower `QA_MIN_SCORE` to retrieve more chunks (more permissive), or raise it to restrict answers to high-confidence matches only.

## How it works

RAG (Retrieval-Augmented Generation) solves the context window problem: rather than feeding entire documents to the model, only the most relevant passages are retrieved and included with each question.

**Ingest phase** (run once per document):
1. Extract text from the PDF with `pypdf`
2. Split into overlapping paragraph-aware chunks (~800 chars with ~150 char overlap)
3. Embed each chunk with `sentence-transformers` (`all-MiniLM-L6-v2`) into a 384-dimensional vector
4. Save a per-document FAISS index and metadata JSON to `doc_store/`

**Query phase** (every question):
1. **Contextualize** — if conversation history exists, the local model rewrites the question into a self-contained query (e.g. "Tell me more" → "Tell me more about the Valles Caldera and Bandelier Tuff"). The rewritten query is used only for retrieval; the original question is preserved for history and generation.
2. Embed the (possibly rewritten) query with the embedding model
3. Search all loaded indexes in parallel, merge results, return top-k by cosine similarity
4. Build a prompt: system instruction + conversation history + retrieved chunks + original question
5. Stream the answer token-by-token from the local Gemma model
6. Print a `Sources:` line listing every document that contributed a retrieved chunk

## Multi-document behaviour

Each PDF gets its own index files inside `doc_store/`. At query time all indexes are searched and results merged before ranking — the model can draw from multiple documents in a single answer.

Re-ingesting a directory prompts per file for any PDF that already has an index:
```
'report.pdf' is already indexed. Overwrite? [y/N]
```

`--reset` wipes the entire `doc_store/` directory.

## Citations

After each answer the source filenames are printed:

```
Sources: report.pdf, guidelines.pdf
```

Only documents that contributed at least one retrieved chunk to that turn are listed.

## Index store

Per-document indexes live in `doc_store/` (gitignored). Run `make ingest` or `make ingest-dir` to rebuild from source documents.

## Tests

```bash
make test
```

43 tests covering `ingest.py`, `retriever.py`, `prompts.py`, and `main.py`. All tests run without a model or real index — dependencies are mocked.

## Model compatibility

`mlx-lm` loads **text-only** models. Vision-language models require `mlx-vlm`. Use a text-only model from `mlx-community` on HuggingFace.

Tested and working: `mlx-community/gemma-3-4b-it-4bit`

## Known limitations

**Response truncation** — small quantized models (4B parameters, 4-bit) have a limited practical generation length. Requests for long-form output (essays, detailed summaries) may be cut off mid-sentence before reaching the requested length. This is a model capacity constraint, not a bug. To get longer responses:

- Use a larger model via `QA_MODEL_PATH` (e.g. a 12B or 27B variant from `mlx-community`)
- Break large requests into smaller, focused questions instead of asking for long essays in one turn
