# Data Extractor V2 — Chunked Extraction with Model-Based Merge

## Context

V1 truncates input to `MAX_CHARS` (default 20,000 chars) and runs a single extraction pass. This is lossy — entities mentioned only in the later portion of a long document are silently missed.

V2 processes the full document by splitting it into overlapping chunks, extracting entities from each chunk independently, then running a model-based merge pass that deduplicates and combines the partial results into a single canonical `ExtractedDocument`.

Both strategies remain available and selectable via a `--strategy` flag. The default stays `truncate` (V1) so existing usage is unchanged.

---

## Goal

```
python main.py document.pdf --strategy chunked
```

Produces the same output format as V1 — a validated `ExtractedDocument` displayed as a rich table — but draws from the full document rather than only the first `MAX_CHARS` characters.

---

## Why model-based merge over programmatic deduplication

Fuzzy string matching can resolve surface variation ("Einstein" vs "Albert Einstein") but cannot resolve semantic ambiguity:
- "Dr. Smith" in chunk 2 and "John Smith" in chunk 8 — same person or two different Smiths?
- "the President" in chunk 1 and "Roosevelt" in chunk 4 — same referent?
- "the lab" in chunk 3 and "Los Alamos" in chunk 9 — same place, different names?

A fuzzy matcher has no context to answer these. The model does. The merge input is also much smaller than the original document — structured JSON objects rather than raw text — so context pressure is low.

---

## Architecture

### Pipeline

```
Document text (full, no truncation)
  ↓
chunker.chunk_document(text)  →  list[str]
  ↓
for each chunk: extractor.extract(chunk, model, tokenizer)  →  list[ExtractedDocument]
  ↓
merger.merge(partials, model, tokenizer)  →  ExtractedDocument
  ↓
display(result)
```

### New files

| File | Responsibility |
|---|---|
| `chunker.py` | `chunk_document(text, chunk_chars, overlap_chars) -> list[str]` — splits text into overlapping character-boundary chunks |
| `merger.py` | `merge(partials, model, tokenizer) -> ExtractedDocument` — model-based deduplication and merge pass with Pydantic validation and retry |
| `tests/test_chunker.py` | Unit tests for chunk sizing, overlap, and edge cases |
| `tests/test_merger.py` | Unit tests with model mocked — valid merge, retry on malformed output, exhaustion |

### Modified files

| File | Change |
|---|---|
| `main.py` | Add `--strategy {truncate,chunked}` argument (default: `truncate`). Add `cmd_chunked()` function. Print strategy label in output header. |
| `config.py` | Add `CHUNK_CHARS` (default 4000) and `OVERLAP_CHARS` (default 400) constants. |
| `Makefile` | Add `STRATEGY` variable so `make run FILE=... STRATEGY=chunked` works. |
| `README.md` | Document `--strategy` flag, chunked pipeline, and comparison guidance. |
| `CLAUDE.md` | Document `chunker.py` and `merger.py` modules. |

`extractor.py`, `schema.py`, and `model.py` are **unchanged** — V2 reuses the same per-chunk extraction logic as V1.

---

## Module detail

### `chunker.py`

```python
def chunk_document(
    text: str,
    chunk_chars: int = config.CHUNK_CHARS,
    overlap_chars: int = config.OVERLAP_CHARS,
) -> list[str]:
```

Splits `text` into overlapping chunks of approximately `chunk_chars` characters. Each chunk after the first begins `overlap_chars` characters before the end of the previous chunk, ensuring that entities straddling a boundary appear in full context in at least one chunk.

Chunk boundaries snap to the nearest paragraph break (`\n\n`) within a tolerance window to avoid splitting mid-sentence. Falls back to a hard character cut if no paragraph break is found within the window.

Returns a list of at least one chunk. If the document is shorter than `chunk_chars`, returns `[text]` unchanged.

**Why 4,000 char chunks?** Large enough to contain complete entities with surrounding context; small enough that a 20,000 char document produces ~5–6 chunks (manageable merge input) and each chunk fits comfortably within the model's context window alongside the extraction prompt.

### `merger.py`

```python
def merge(
    partials: list[ExtractedDocument],
    model,
    tokenizer,
    max_retries: int = config.MAX_RETRIES,
) -> ExtractedDocument:
```

Serializes `partials` to a compact JSON array and asks the model to:
1. Merge all `people` lists, deduplicating by identity (same person, different name forms)
2. Merge all `places` lists, deduplicating similarly
3. Merge all `dates` lists, keeping unique events
4. Choose the best `title` (prefer non-null, prefer earlier chunks)
5. Write a unified `topic` and `summary` covering the full document

Returns a validated `ExtractedDocument`. Uses the same fence-stripping, `json.loads`, `model_validate`, and retry-with-error-appended pattern as `extractor.py`.

Raises `ExtractionError` (same exception class, already imported from `extractor`) if all retries are exhausted.

**Edge cases:**
- `partials` is empty → raise `ValueError` immediately, no model call
- `partials` has one element → return it directly, no model call (no merge needed)

### `config.py` additions

```python
CHUNK_CHARS   = int(os.getenv("EXTRACTOR_CHUNK_CHARS", "4000"))
OVERLAP_CHARS = int(os.getenv("EXTRACTOR_OVERLAP_CHARS", "400"))
```

### `main.py` additions

New `--strategy` argument:

```
--strategy {truncate,chunked}   extraction strategy (default: truncate)
```

New `cmd_chunked(text, model, tokenizer)` function:

```
1. chunk_document(text)  →  print "N chunks"
2. for each chunk: extract()  →  print progress "Chunk k/N..."
3. merge(partials)  →  print "Merging..."
4. return result
```

Existing `cmd_truncate()` is a thin rename of the current single-pass logic. `main()` dispatches based on `args.strategy`.

---

## Merge prompt design

The merge system prompt must:
1. Explain that the inputs are partial extractions from consecutive chunks of the same document
2. Show the output schema (same as the extraction prompt)
3. Instruct the model to deduplicate by identity, not just by exact string match
4. Specify how to handle conflicts (prefer the most specific/detailed version of a field)
5. Forbid preamble and fences — same rules as the extraction prompt

User message: the serialized partial results as a JSON array, labelled "Partial extractions to merge:".

---

## CLI comparison workflow

```bash
# V1 — truncation
python main.py long_document.pdf --strategy truncate

# V2 — chunked
python main.py long_document.pdf --strategy chunked
```

Both print the same rich table format. The output header will note which strategy was used and (for chunked) how many chunks were processed.

---

## New environment variables

| Variable | Default | Description |
|---|---|---|
| `EXTRACTOR_CHUNK_CHARS` | `4000` | Target chunk size in characters |
| `EXTRACTOR_OVERLAP_CHARS` | `400` | Overlap between consecutive chunks |

---

## Tests

### `test_chunker.py`

- Short document (< chunk_chars) → single chunk, no split
- Document exactly chunk_chars → single chunk
- Long document → multiple chunks, each ≤ chunk_chars + tolerance
- Overlap: last `overlap_chars` of chunk N appear at the start of chunk N+1
- Paragraph-boundary snapping: split at `\n\n` when near a boundary
- Chunk with no paragraph breaks → hard character cut
- Empty string → returns `[""]` or raises, documented behaviour

### `test_merger.py`

- Single partial → returned directly, no model call
- Empty partials → `ValueError` raised
- Two partials → model called, result validated
- Model returns fenced JSON → stripped and parsed correctly
- Model returns invalid JSON → retries with error appended
- Model returns invalid schema → retries with validation error
- All retries exhausted → `ExtractionError` raised
- Merge result contains deduplicated people from both partials

---

## Implementation order

1. Add `CHUNK_CHARS` and `OVERLAP_CHARS` to `config.py`
2. Write `tests/test_chunker.py` (TDD)
3. Implement `chunker.py`; tests pass
4. Write `tests/test_merger.py` (TDD)
5. Implement `merger.py`; tests pass
6. Update `main.py` — add `--strategy` flag and `cmd_chunked()`
7. Update `Makefile` with `STRATEGY` variable
8. Update `README.md` and `CLAUDE.md`
9. Manual smoke test: run both strategies on the same long document, compare outputs
10. Commit

---

## Verification

- `make test` — all previous 40 tests still pass + new chunker and merger tests
- `make lint` — clean
- Manual smoke tests:
  - Short document (< 4000 chars) with `--strategy chunked` → 1 chunk, no merge, same output as truncate
  - Long document with `--strategy truncate` → truncation warning, partial extraction
  - Same long document with `--strategy chunked` → no truncation warning, richer extraction
  - Document with repeated entity names across chunks → single deduplicated entry in merge output
  - All retries exhausted in merge → graceful error, non-zero exit

---

## Out of scope (deliberate)

- Parallel chunk extraction (sequential is simpler and sufficient for the learning goal)
- Caching partial results between runs
- User-visible chunk boundaries or per-chunk output
- PDF-specific chunking (page boundaries) — character chunking is simpler and model-agnostic
