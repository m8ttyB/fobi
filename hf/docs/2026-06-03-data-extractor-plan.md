# Data Extractor — Implementation Plan

## Purpose

Extract structured entities from unstructured text (emails, meeting notes, articles, PDFs) and return a validated JSON object. The model is given the full document text and asked to return a single complete JSON object matching a predefined schema. If the response is malformed or fails Pydantic validation, the error is appended to the prompt and the model is retried up to N times.

This project teaches:
- Prompting for structured JSON output reliably
- Pydantic validation — both flat fields and nested models
- Retry logic when the model doesn't follow format
- LLM failure modes (preamble, markdown fences, missing fields, wrong types)
- Graceful degradation when extraction partially fails

---

## Schema

```python
class Person(BaseModel):
    name: str
    role: str | None      # "physicist", "senator", "author"
    context: str | None   # "discussed relativity with Bohr"

class Place(BaseModel):
    name: str
    context: str | None   # "site of the 1945 test"

class Date(BaseModel):
    date: str             # raw string — preserves "mid-1930s", "early July"
    event: str | None     # "Trinity nuclear test"

class ExtractedDocument(BaseModel):
    title: str | None
    topic: str            # required
    people: list[Person]
    places: list[Place]
    dates: list[Date]
    summary: str          # required
```

`topic` and `summary` are required — they anchor every extraction. All list fields default to empty. `date` is kept as a raw string (not `datetime`) to preserve the author's phrasing ("mid-1930s", "early July") without forcing a machine-readable format that would fail on approximate dates.

---

## Architecture

Four modules wired together in `main.py`:

| Module | Responsibility |
|---|---|
| `config.py` | Env-overridable constants — `MODEL_PATH`, `MAX_CHARS`, `MAX_RETRIES` |
| `schema.py` | Pydantic models — `Person`, `Place`, `Date`, `ExtractedDocument` |
| `extractor.py` | `extract(text, model, tokenizer) -> ExtractedDocument` — prompt construction, JSON parsing, Pydantic validation, retry loop |
| `model.py` | `load_model(path) -> (model, tokenizer)`, `generate(model, tokenizer, messages) -> str` — thin wrapper over `mlx_lm` |
| `main.py` | CLI entry point — load file, truncate if needed, load model, run extraction, display with `rich` |

---

## Module detail

### `config.py`

```python
MODEL_PATH  = os.getenv("EXTRACTOR_MODEL_PATH", "mlx-community/gemma-3-4b-it-4bit")
MAX_CHARS   = int(os.getenv("EXTRACTOR_MAX_CHARS", "20000"))
MAX_RETRIES = int(os.getenv("EXTRACTOR_MAX_RETRIES", "3"))
```

### `schema.py`

Pydantic models only. No logic. Imports `pydantic.BaseModel` and `pydantic.Field` for optional `description=` metadata.

### `extractor.py`

`extract(text, model, tokenizer) -> ExtractedDocument`

1. Build the initial prompt — system message describing the task and the exact JSON schema, user message containing the document text.
2. Call `model.generate()` to get a full response string (not streaming — the whole response is needed before parsing).
3. Strip markdown code fences if present (```` ```json ... ``` ````).
4. `json.loads()` — catch `json.JSONDecodeError`.
5. `ExtractedDocument.model_validate(parsed)` — catch `pydantic.ValidationError`.
6. On any parse or validation failure: append the error message to the conversation and retry. Repeat up to `MAX_RETRIES` times.
7. If all retries exhausted: raise `ExtractionError` with the last error attached.

Helper: `_strip_fences(text: str) -> str` — strips ` ```json ` / ` ``` ` wrappers. Models frequently wrap JSON in markdown fences even when instructed not to.

### `model.py`

Identical pattern to `doc-qa/model.py`:
- `load_model(path) -> (model, tokenizer)`
- `generate(model, tokenizer, messages) -> str` — returns full response as string

No streaming needed — the whole JSON object must arrive before it can be parsed.

### `main.py`

```
usage: python main.py FILE [--model MODEL_PATH]

positional arguments:
  FILE          Path to a .txt or .pdf file to extract from

optional arguments:
  --model       Override model path (default: EXTRACTOR_MODEL_PATH env var)
  -h, --help    Show this help message and exit
```

Flow:
1. Parse args with `argparse`.
2. Read file — `.pdf` via `pypdf`, `.txt` via `open()`. Error and exit on unknown extension.
3. Truncate to `MAX_CHARS` — print a visible warning with `rich` if truncated, showing original and truncated length.
4. Load model.
5. Call `extractor.extract(text, model, tokenizer)`.
6. Display result with `rich` — pretty-printed JSON with field labels, or a structured error message if extraction failed after all retries.

---

## Prompt design

The system prompt must:
1. State the task clearly — "Extract structured information from the document below."
2. Show the exact JSON schema the model must return — inline, not by reference.
3. Explicitly forbid preamble and markdown fences — "Return only the JSON object. No explanation. No markdown code fences."
4. Specify handling for missing information — "Use null for fields where information is not present. Use an empty list [] if no items are found."

The user message is just the document text.

On retry, the failed response and the error message are appended:
```
Your previous response could not be parsed. Error: [validation error details]
Return only the valid JSON object, correcting the error above.
```

---

## Error taxonomy

| Exception | When raised | Shown to user |
|---|---|---|
| `ExtractionError` | All retries exhausted | "Extraction failed after N attempts: [last error]" |
| `FileNotFoundError` | Input file missing | "File not found: [path]" |
| `ValueError` | Unsupported file extension | "Unsupported file type: [ext]. Use .txt or .pdf" |
| `pypdf` errors | Corrupt or unreadable PDF | "Could not read PDF: [error]" |

---

## Display

Use `rich` to render the output. A successful extraction prints each top-level field as a labeled section:

```
 Title    The Oppenheimer Story
 Topic    Manhattan Project and nuclear weapons development
 People   J. Robert Oppenheimer — physicist, directed the bomb design
          Niels Bohr — physicist, consulted on fission theory
 Places   Los Alamos, NM — primary weapons design facility
          Trinity Site, NM — site of the first nuclear test
 Dates    July 16, 1945 — Trinity nuclear test
          August 6, 1945 — bomb dropped on Hiroshima
 Summary  A detailed account of ...
```

A truncation warning appears before the output:
```
 Warning  Document truncated: 87,432 → 20,000 chars. Results cover only the first portion.
```

An extraction failure prints the error clearly and exits with a non-zero code.

---

## File layout

```
data-extractor/
├── config.py
├── schema.py
├── extractor.py
├── model.py
├── main.py
├── requirements.txt
├── Makefile
├── README.md
├── CLAUDE.md
└── tests/
    ├── test_schema.py        # Pydantic validation — valid, partial, invalid inputs
    ├── test_extractor.py     # extract() with model mocked — success, retry, exhaustion
    └── test_main.py          # CLI arg parsing and file loading (model mocked)
```

---

## Makefile targets

| Target | Command |
|---|---|
| `make install` | Create `.venv`, install dependencies |
| `make run FILE=...` | Run extraction on a file |
| `make test` | Run all tests |
| `make lint` | `ruff check .` |
| `make format` | `ruff format .` |

---

## Requirements

```
mlx-lm
pypdf
pydantic
rich
```

No `sentence-transformers` or `faiss` — this project does not use vector search.

---

## Key design decisions

**Why one complete JSON object rather than field-by-field extraction?** A single extraction pass preserves cross-field context — a name appearing near a date in the text is evidence for both `people` and `dates`. Field-by-field would multiply model calls by the number of fields and lose that context.

**Why raw string for `date` rather than `datetime`?** Real documents contain approximate dates — "mid-1930s", "early July", "sometime after the war". A `datetime` type would reject these and force lossy normalization. Preserving the author's phrasing is more useful than machine-readability for a summary extraction tool.

**Why retry with error appended rather than a fresh prompt?** Appending the validation error gives the model specific feedback about what it got wrong. A fresh prompt would repeat the same mistake. This mirrors the pattern used in production LLM pipelines for structured output.

**Why not streaming?** The entire JSON object must arrive before it can be parsed. Streaming token-by-token is useful for prose generation (the user sees progress); for structured output it adds complexity with no benefit.

**Why truncate rather than chunk-and-merge for v1?** Truncation keeps the focus on structured output, validation, and retry — the core teaching goals. Chunk-and-merge adds a merge pass that would dominate the implementation. V2 will add chunked extraction once the v1 patterns are solid.

---

## Implementation order

1. `requirements.txt` and `Makefile` — install deps
2. `config.py` — constants
3. `schema.py` — Pydantic models; write `test_schema.py` first (TDD)
4. `model.py` — `load_model` and `generate`
5. `extractor.py` — prompt, parse, validate, retry; write `test_extractor.py` first (TDD)
6. `main.py` — CLI, file loading, truncation, display; write `test_main.py` first (TDD)
7. `README.md` and `CLAUDE.md`
8. Manual smoke test against a real document
9. Commit

---

## Verification

- `make test` — all tests passing without a real model (mocked)
- `make lint` — clean
- Manual smoke tests:
  - Short plain text → clean JSON output, no truncation warning
  - Long PDF → truncation warning + extraction from first 20K chars
  - Intentionally bad document (random bytes) → graceful error, non-zero exit
  - Retry path — observable via log output when model returns prose instead of JSON
