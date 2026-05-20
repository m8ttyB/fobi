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

## Chat commands

| Command | Description |
|---|---|
| `/exit` | Quit the session |
| `/clear` | Reset conversation history |

## How it works

RAG (Retrieval-Augmented Generation) solves the context window problem: rather than feeding entire documents to the model, only the most relevant passages are retrieved and included with each question.

**Ingest phase** (run once per document):
1. Extract text from the PDF with `pypdf`
2. Split into overlapping paragraph-aware chunks (~800 chars with ~150 char overlap)
3. Embed each chunk with `sentence-transformers` (`all-MiniLM-L6-v2`) into a 384-dimensional vector
4. Save a per-document FAISS index and metadata JSON to `doc_store/`

**Query phase** (every question):
1. Embed the question with the same embedding model
2. Search all loaded indexes in parallel, merge results, return top-k by cosine similarity
3. Build a prompt: system instruction + conversation history + retrieved chunks + question
4. Stream the answer token-by-token from the local Gemma model
5. Print a `Sources:` line listing every document that contributed a retrieved chunk

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

36 tests covering `ingest.py`, `retriever.py`, `prompts.py`, and `main.py`. All tests run without a model or real index — dependencies are mocked.

## Model compatibility

`mlx-lm` loads **text-only** models. Vision-language models require `mlx-vlm`. Use a text-only model from `mlx-community` on HuggingFace.

Tested and working: `mlx-community/gemma-3-4b-it-4bit`
