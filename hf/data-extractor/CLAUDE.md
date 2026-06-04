# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Environment setup

```bash
make install
export EXTRACTOR_MODEL_PATH="mlx-community/gemma-4-12B-4bit"  # downloads ~7 GB on first run
```

## Running

```bash
# V1 — truncate (default)
make run FILE=path/to/document.txt

# V2 — chunked (full document)
make run FILE=path/to/document.pdf STRATEGY=chunked

# Override model:
make run FILE=doc.txt MODEL=mlx-community/gemma-3-4b-it-4bit STRATEGY=chunked
```

## Tests

```bash
make test                                         # all tests (no model required — dependencies mocked)
.venv/bin/pytest tests/test_schema.py -v         # Pydantic schema validation tests
.venv/bin/pytest tests/test_extractor.py -v      # extraction logic, fence stripping, retry loop
.venv/bin/pytest tests/test_chunker.py -v        # chunk sizing, overlap, paragraph snapping
.venv/bin/pytest tests/test_merger.py -v         # merge logic, deduplication prompt, retry
.venv/bin/pytest tests/test_main.py -v           # CLI arg parsing, strategy dispatch, file loading
```

## Linting

```bash
make lint    # check
make format  # auto-fix
```

## Architecture

Seven modules wired together in `main.py`:

- **`config.py`** — env-overridable constants only (`MODEL_PATH`, `MAX_CHARS`, `MAX_RETRIES`, `CHUNK_CHARS`, `OVERLAP_CHARS`). No logic.
- **`schema.py`** — Pydantic models: `Person`, `Place`, `Date`, `ExtractedDocument`. No logic. `topic` and `summary` are required; all list fields default to empty; `date` is a raw string to preserve approximate dates like "mid-1930s".
- **`extractor.py`** — `extract(text, model, tokenizer, max_retries)` builds a prompt with the full JSON schema, calls `generate()`, strips markdown fences with `_strip_fences()`, parses JSON, validates with Pydantic, and retries with the error appended on failure. Raises `ExtractionError` after all retries are exhausted. Used by both V1 and V2.
- **`chunker.py`** — `chunk_document(text, chunk_chars, overlap_chars)` splits text into overlapping chunks snapping to paragraph boundaries (`\n\n`) within a tolerance window. Falls back to hard character cuts when no paragraph break is found. Returns `[text]` unchanged if the document fits in one chunk.
- **`merger.py`** — `merge(partials, model, tokenizer, max_retries)` serializes a list of `ExtractedDocument` objects to JSON and asks the model to deduplicate and combine them into a single canonical result. Uses the same fence-stripping, parse, validate, and retry-with-error pattern as `extractor.py`. Short-circuits: returns the single element directly if `len(partials) == 1`; raises `ValueError` if empty.
- **`model.py`** — thin wrapper over `mlx_lm`. `load_model(path)` returns `(model, tokenizer)`. `generate(model, tokenizer, messages)` returns the full response as a string with `max_tokens=2048` — no streaming, since the whole JSON object must arrive before parsing.
- **`main.py`** — argparse CLI with positional `FILE`, optional `--model`, and `--strategy {truncate,chunked}` (default: `truncate`). `load_text()` handles `.txt` and `.pdf`. `truncate()` caps at `MAX_CHARS`. `cmd_truncate()` is V1; `cmd_chunked()` orchestrates chunk → extract loop → merge for V2. `display()` renders a rich table with a strategy label and truncation warning.

## Key design decisions

**Why one complete JSON object rather than field-by-field extraction?** A single extraction pass preserves cross-field context — a name appearing near a date is evidence for both `people` and `dates`. Field-by-field would multiply model calls and lose that context.

**Why raw string for `date`?** `datetime` would reject "mid-1930s" or "early July". Preserving the author's phrasing is more useful than machine-readability for a summary tool.

**Why retry with error appended rather than a fresh prompt?** Appending the validation error gives the model specific feedback. A fresh prompt repeats the same mistake. This mirrors production patterns for structured LLM output.

**Why model-based merge rather than fuzzy string matching?** Fuzzy matching resolves surface variation ("Einstein" vs "Albert Einstein") but cannot resolve semantic ambiguity ("Dr. Smith" vs "John Smith", "the lab" vs "Los Alamos"). The model has context; a fuzzy matcher does not.

**Why no streaming?** The entire JSON object must arrive before parsing. Streaming adds complexity with no benefit.

**Why paragraph-boundary snapping in the chunker?** Paragraph breaks represent natural semantic units. Splitting mid-sentence means a person or event straddling the cut may be described incompletely in both chunks. Overlap ensures the boundary region appears in full context in at least one chunk.

**Why `max_tokens=2048` in `generate()`?** The default `mlx_lm.generate` limit is 100 tokens, which is far too short for a full JSON object. 2048 gives enough headroom for a complete extraction response without risking context overflow.
