# doc-qa

A CLI tool for asking questions about a document using Retrieval-Augmented Generation (RAG). Ingest a PDF once to build a local vector index, then ask questions in a REPL session — the model answers using only content from your document.

## Setup

```bash
cd hf/doc-qa
make install
```

## Quick start

```bash
# 1. Ingest a document (run once per document)
make ingest FILE=path/to/document.pdf

# 2. Ask questions
make chat
```

## Make targets

| Target | Description |
|---|---|
| `make install` | Create `.venv` and install all dependencies |
| `make ingest FILE=...` | Chunk, embed, and index a PDF document |
| `make chat` | Start a Q&A REPL session |
| `make reset` | Delete the index and metadata (with confirmation) |
| `make test` | Run the test suite |
| `make lint` | Check code with ruff |
| `make format` | Auto-format code with ruff |

`MODEL` can be overridden:

```bash
make ingest FILE=doc.pdf MODEL=mlx-community/gemma-3-4b-it-4bit
make chat MODEL=mlx-community/gemma-3-4b-it-4bit
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `QA_MODEL_PATH` | `mlx-community/gemma-3-4b-it-4bit` | HuggingFace repo ID or local MLX model directory |
| `QA_EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model for embedding |
| `QA_INDEX_PATH` | `index.faiss` | Path to FAISS index file |
| `QA_METADATA_PATH` | `metadata.json` | Path to chunk metadata file |
| `QA_TOP_K` | `4` | Number of chunks to retrieve per question |

## Chat commands

| Command | Description |
|---|---|
| `/exit` | Quit the session |
| `/clear` | Reset conversation history |

## How it works

RAG (Retrieval-Augmented Generation) solves the context window problem: rather than feeding the entire document to the model, only the most relevant passages are retrieved and included with each question.

**Ingest phase** (run once):
1. Extract text from the PDF with `pypdf`
2. Split into overlapping paragraph-aware chunks (~800 chars with ~150 char overlap)
3. Embed each chunk with `sentence-transformers` (`all-MiniLM-L6-v2`) into a 384-dimensional vector
4. Save the vectors to a FAISS index and the chunk text to a metadata JSON file

**Query phase** (every question):
1. Embed the question with the same model
2. Find the top-4 most similar chunks via cosine similarity search in FAISS
3. Build a prompt: system instruction + conversation history + retrieved chunks + question
4. Stream the answer token-by-token from the local Gemma model

Conversation history accumulates across turns in the REPL, enabling follow-up questions. History is trimmed oldest-first when context fills — retrieved chunks are never sacrificed for history length.

## Index files

`index.faiss` and `metadata.json` are written to the current directory after ingest. They are gitignored — run `make ingest` again to rebuild from the source document.

## Tests

```bash
make test
```

~23 tests covering `ingest.py`, `retriever.py`, and `prompts.py`. All tests run without a model or index — dependencies are mocked.

## Model compatibility

`mlx-lm` loads **text-only** models. Vision-language models require `mlx-vlm`. Use a text-only model from `mlx-community` on HuggingFace.

Tested and working: `mlx-community/gemma-3-4b-it-4bit`
