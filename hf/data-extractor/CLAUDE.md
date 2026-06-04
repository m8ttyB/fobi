# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Environment setup

```bash
make install
export EXTRACTOR_MODEL_PATH="mlx-community/gemma-3-4b-it-4bit"  # downloads ~2.5 GB on first run
```

## Running

```bash
make run FILE=path/to/document.txt
make run FILE=path/to/document.pdf

# Override model:
make run FILE=doc.txt MODEL=mlx-community/gemma-3-4b-it-4bit
```

## Tests

```bash
make test                                         # all tests (no model required — dependencies mocked)
.venv/bin/pytest tests/test_schema.py -v         # Pydantic schema validation tests
.venv/bin/pytest tests/test_extractor.py -v      # extraction logic, fence stripping, retry loop
.venv/bin/pytest tests/test_main.py -v           # CLI arg parsing, file loading, truncation, display
```

## Linting

```bash
make lint    # check
make format  # auto-fix
```

## Architecture

Five modules wired together in `main.py`:

- **`config.py`** — env-overridable constants only (`MODEL_PATH`, `MAX_CHARS`, `MAX_RETRIES`). No logic.
- **`schema.py`** — Pydantic models: `Person`, `Place`, `Date`, `ExtractedDocument`. No logic. `topic` and `summary` are required; all list fields default to empty; `date` is a raw string to preserve approximate dates like "mid-1930s".
- **`extractor.py`** — `extract(text, model, tokenizer, max_retries)` builds a prompt with the full JSON schema, calls `generate()`, strips markdown fences with `_strip_fences()`, parses JSON, validates with Pydantic, and retries with the error appended on failure. Raises `ExtractionError` after all retries are exhausted.
- **`model.py`** — thin wrapper over `mlx_lm`. `load_model(path)` returns `(model, tokenizer)`. `generate(model, tokenizer, messages)` returns the full response as a string — no streaming, since the whole JSON object must arrive before parsing.
- **`main.py`** — argparse CLI with a positional `FILE` argument and optional `--model` flag. `load_text()` handles `.txt` and `.pdf`. `truncate()` caps at `MAX_CHARS` and returns a `(text, truncated)` tuple. `display()` renders the result as a rich table with labeled rows for each field.

## Key design decisions

**Why one complete JSON object rather than field-by-field extraction?** A single extraction pass preserves cross-field context — a name appearing near a date is evidence for both `people` and `dates`. Field-by-field would multiply model calls and lose that context.

**Why raw string for `date`?** `datetime` would reject "mid-1930s" or "early July". Preserving the author's phrasing is more useful than machine-readability for a summary tool.

**Why retry with error appended rather than a fresh prompt?** Appending the validation error gives the model specific feedback. A fresh prompt repeats the same mistake. This mirrors production patterns for structured LLM output.

**Why no streaming?** The entire JSON object must arrive before parsing. Streaming adds complexity with no benefit.

**Why truncate for v1?** Keeps the focus on structured output, validation, and retry. V2 will add chunked extraction once v1 patterns are solid.
