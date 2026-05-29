# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Environment setup

```bash
make install
export QA_MODEL_PATH="mlx-community/gemma-3-4b-it-4bit"  # downloads ~2.5 GB on first run
```

## Running

```bash
make ingest FILE=path/to/document.pdf   # chunk, embed, save index (run once per document)
make chat                               # start Q&A REPL
make reset                              # delete index and metadata (with confirmation)

# Override model:
make ingest FILE=doc.pdf MODEL=mlx-community/gemma-3-4b-it-4bit
```

## Tests

```bash
make test                                        # all tests (no model required — dependencies mocked)
.venv/bin/pytest tests/test_ingest.py -v        # chunker and embedding tests
.venv/bin/pytest tests/test_retriever.py -v     # FAISS retrieval tests
.venv/bin/pytest tests/test_prompts.py -v       # prompt builder tests
```

## Linting

```bash
make lint    # check
make format  # auto-fix
```

## Architecture

Six modules wired together in `main.py`:

- **`contextualizer.py`** — `contextualize_query(question, history, model, tokenizer)` rewrites follow-up questions into self-contained retrieval queries using the local model. Returns the original question unchanged on the first turn (empty history) or if the model returns an empty string. Used only for retrieval — the original question is preserved for history and generation.
- **`config.py`** — env-overridable constants only (`MODEL_PATH`, `EMBED_MODEL`, `STORE_DIR`, `TOP_K`, `MIN_SCORE`, `MAX_CHUNK_CHARS`, `OVERLAP_CHARS`, `MAX_HISTORY_TURNS`). No logic.
- **`ingest.py`** — `chunk_text(text)` splits a document into overlapping paragraph-aware chunks. `embed_chunks(chunks, model_name)` encodes them with `sentence-transformers` and normalizes to unit length. `save_index(embeddings, chunks, source, index_path, metadata_path)` writes a FAISS `IndexFlatIP` and chunk metadata JSON. `index_name_for(pdf_path, base_dir)` derives a collision-safe filename stem for nested paths. `ingest_directory(dir_path, store_dir, embed_model_name)` recursively finds all PDFs, prompts on already-indexed files, and ingests each one.
- **`retriever.py`** — `load_index(index_path, metadata_path)` reads a single FAISS index pair. `load_all_indexes(store_dir)` loads every `.faiss`/`.json` pair in the store directory. `retrieve(query, embed_model, index, metadata, top_k)` searches a single index. `retrieve_multi(query, embed_model, indexes, top_k)` searches all indexes, merges by score, returns top-k.
- **`prompts.py`** — `build_messages(history, question, chunks)` formats retrieved chunks as a context block and returns a messages list ready for `apply_chat_template`.
- **`model.py`** — thin wrapper over `mlx_lm`. `load_model(path)` returns `(model, tokenizer)`. `stream_response(model, tokenizer, messages)` yields `(chunk.text, chunk)` tuples. `generate(model, tokenizer, messages)` returns the full response as a string — used by `contextualizer.py` for the rewrite pass.
- **`main.py`** — argparse CLI with four mutually exclusive flags: `--ingest FILE`, `--ingest-dir DIR`, `--chat`, `--reset`. Wires all modules together. After each generated response, prints a `Sources:` line listing unique document filenames from the retrieved chunks. Imports `readline` at startup (when available) to enable up/down-arrow history navigation in the chat REPL; degrades gracefully to plain `input()` on Windows.

## Key design decisions

**Why FAISS `IndexFlatIP` with normalized vectors?** Inner product on unit-normalized vectors equals cosine similarity. `IndexFlatIP` is exact (no approximation), simple, and fast enough for document-scale indexes. Normalized embeddings ensure the direction of the vector encodes meaning, not its magnitude.

**Why `sentence-transformers` for embedding?** The generative model (Gemma) is trained for text generation, not semantic similarity. `sentence-transformers` models (e.g. `all-MiniLM-L6-v2`) are trained specifically for embedding — they map semantically similar text to nearby vectors. The query and chunks must use the same embedding model to be comparable.

**Why are the embedding model and generative model separate?** Embedding and generation are different tasks requiring different training objectives. Reusing the generative model for embedding would give poor retrieval quality.

**Why paragraph-aware chunking with overlap?** Paragraph boundaries represent natural semantic units — the author grouped related ideas together. Overlap ensures that ideas straddling a boundary appear in at least one chunk in full context. Purely character-count-based chunking would split mid-sentence.

**Why threshold filtering happens before top_k capping?** Applying `min_score` first means a query with no strong matches returns an empty list rather than forcing k low-quality chunks into the prompt. If top_k were applied first, the threshold would silently reduce k on every query — the user would never know whether 2 chunks or 4 were retrieved.

**Why history trimming defers to retrieval?** Retrieved chunks are a fixed cost per turn (top-k chunks, same size every time). History grows unboundedly. When the context budget tightens, trimming history preserves retrieval quality — the model always has fresh, relevant document context. Old history turns are dropped first.

**Why `index.faiss` and `metadata.json` are gitignored.** They are derived artifacts built from the ingested document. Committing them would bloat the repo and they can always be recreated with `make ingest`.
