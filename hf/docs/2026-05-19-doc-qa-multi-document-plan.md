# doc-qa — Multi-Document Ingestion Plan

## Context

`doc-qa` v1 supports a single document: one `index.faiss` and one `metadata.json` live in the working directory. Ingesting a second document silently overwrites the first.

This feature extends ingestion to accept a directory of PDFs, building a per-document index for each one and searching across all indexes at query time.

## Design decisions (from clarifying questions)

| Decision | Choice | Rationale |
|---|---|---|
| Index strategy | Per-document indexes | Each PDF gets its own `index.faiss` + `metadata.json` in a dedicated store directory. Enables removing a single document without rebuilding everything. |
| Re-ingest behavior | Ask per document | When a PDF already has an index, prompt skip or overwrite. Prevents accidental data loss while keeping the workflow deliberate. |
| Citations | Yes — cite all sources used | After each answer, show the source PDF filename(s) retrieved chunks came from. If multiple documents contributed, all are listed. |
| Reset | Reset all | `--reset` wipes the entire index store. No per-document targeting in v1. |
| Directory depth | Recursive | `--ingest-dir` walks subdirectories and picks up all `.pdf` files found. |

## CLI changes

```bash
# New flag — ingest all PDFs in a directory (recursively)
python main.py --ingest-dir path/to/pdfs/

# Existing flags — unchanged
python main.py --ingest path/to/document.pdf
python main.py --chat
python main.py --reset
python main.py --help
```

`--ingest` (single file) continues to work exactly as before.

## Index store layout

All per-document indexes live in a single store directory (default: `./doc_store/`):

```
doc_store/
├── my_report.faiss
├── my_report.json
├── another_doc.faiss
├── another_doc.json
└── subdir__nested_doc.faiss     # subdirectory separator: __
    subdir__nested_doc.json
```

Filename collisions from different subdirectories are resolved by prefixing with the relative path, using `__` as the separator (e.g. `reports/q1.pdf` → `reports__q1`).

The store directory path is env-overridable via `QA_STORE_DIR` (default: `doc_store`). The existing `QA_INDEX_PATH` and `QA_METADATA_PATH` constants are retired.

## Architecture changes

### `config.py`

Remove `INDEX_PATH` and `METADATA_PATH`. Add:

```python
STORE_DIR = os.environ.get("QA_STORE_DIR", "doc_store")
```

### `ingest.py`

Add two functions alongside the existing ones:

```python
def index_name_for(pdf_path: str, base_dir: str) -> str:
    """Derive a collision-safe index filename stem from a PDF path."""
    # relative path from base_dir, separators replaced with __
    # e.g. base_dir=pdfs/, pdf=pdfs/reports/q1.pdf → reports__q1

def ingest_directory(dir_path: str, store_dir: str) -> None:
    """Recursively find all PDFs in dir_path and ingest each one.
    Prompts skip/overwrite for any PDF that already has an index."""
```

`save_index` gains two new parameters: `index_path` and `metadata_path` (explicit paths rather than reading from config). The existing single-file `--ingest` path passes its own paths; `--ingest-dir` passes per-document paths derived from `index_name_for`.

### `retriever.py`

Replace `load_index()` with:

```python
def load_all_indexes(store_dir: str) -> list[tuple[faiss.Index, list[dict]]]:
    """Load every .faiss / .json pair in store_dir."""

def retrieve_multi(
    query: str,
    embed_model,
    indexes: list[tuple[faiss.Index, list[dict]]],
    top_k: int,
) -> list[dict]:
    """Search all indexes, merge results, return top_k by score."""
```

`retrieve` (single index) is kept for backward compatibility with the existing single-`--ingest` flow.

### `prompts.py`

No interface change. `build_messages` already receives a `chunks` list — citations are added by `main.py` after generation completes, not inside the prompt builder.

### `main.py`

- `cmd_ingest` — unchanged except it now calls `save_index` with explicit paths derived from the store directory.
- `cmd_ingest_dir` — new. Walks the directory recursively, calls `ingest_directory`.
- `cmd_chat` — calls `load_all_indexes` instead of `load_index`. After each streamed response, prints a `Sources:` line listing unique filenames from the retrieved chunks.
- `cmd_reset` — deletes the entire `STORE_DIR` directory (with confirmation) rather than two individual files.

### Citation format

After each generated response:

```
Sources: my_report.pdf, another_doc.pdf
```

Only files that contributed at least one retrieved chunk to that turn are listed. If all chunks came from one document, only that file appears.

## New files

None — all changes are to existing modules.

## Modified files

| File | Change |
|---|---|
| `config.py` | Replace `INDEX_PATH`, `METADATA_PATH` with `STORE_DIR` |
| `ingest.py` | Add `index_name_for`, `ingest_directory`; update `save_index` signature |
| `retriever.py` | Add `load_all_indexes`, `retrieve_multi`; keep `load_index`, `retrieve` |
| `prompts.py` | No change |
| `main.py` | Add `cmd_ingest_dir`; update `cmd_chat`, `cmd_reset`, `cmd_ingest` |
| `Makefile` | Add `ingest-dir` target |
| `README.md` | Document `--ingest-dir`, citation output, `STORE_DIR` env var |
| `CLAUDE.md` | Update architecture section |

## Updated Makefile targets

```makefile
make ingest FILE=path/to/doc.pdf     # single file (unchanged)
make ingest-dir DIR=path/to/pdfs/    # directory
make chat
make reset
```

## Test plan

New tests alongside existing suite:

| Test file | What's covered |
|---|---|
| `tests/test_ingest.py` | `index_name_for` path-to-stem conversion; collision avoidance for nested paths |
| `tests/test_retriever.py` | `load_all_indexes` with multiple mocked index pairs; `retrieve_multi` merges and ranks correctly; top_k respected across merged results |
| `tests/test_main.py` | `cmd_ingest_dir` skips already-indexed files when user answers N; re-indexes when user answers Y |

## Implementation order

1. Update `config.py` — replace constants
2. Update `ingest.py` — add `index_name_for`, `ingest_directory`, update `save_index` signature
3. Write / update tests for `ingest.py` → pass
4. Update `retriever.py` — add `load_all_indexes`, `retrieve_multi`
5. Write / update tests for `retriever.py` → pass
6. Update `main.py` — wire new commands, add citation output, update reset
7. Write `tests/test_main.py` for `cmd_ingest_dir` prompting behavior
8. Update `Makefile`, `README.md`, `CLAUDE.md`
9. Smoke test: ingest `pdfs/` directory, ask questions that span multiple documents, verify citations
10. Commit

## Smoke test checklist

- `make ingest-dir DIR=pdfs/` — all PDFs indexed, per-document files appear in `doc_store/`
- Re-run same command — prompted per already-indexed file; skip leaves file untouched, overwrite rebuilds it
- `make chat` — ask a question answered by one document; `Sources:` shows one file
- Ask a question that spans two documents; `Sources:` shows both files
- Ask an out-of-scope question; model says it can't find the answer
- `make reset` — confirmation prompt; `doc_store/` removed on `y`
- `make chat` after reset — clear error: no indexes found, run ingest first

## Out of scope (v1 of this feature)

- Per-document reset (`--reset-doc`)
- Filtering chat to a subset of documents
- Incremental re-index (detecting changed content within a file)
- Non-PDF formats
