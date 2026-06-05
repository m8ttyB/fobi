# data-extractor

A CLI tool for extracting structured entities from unstructured text. Give it a plain text file or PDF and it extracts title, topic, people, places, dates, and a summary — validated against a Pydantic schema and displayed as a formatted table.

Two extraction strategies are available for direct comparison:

| Strategy | How it works | Best for |
|---|---|---|
| `truncate` (V1, default) | Caps input at `MAX_CHARS`, single model call | Short documents, fast results |
| `chunked` (V2) | Splits full document, extracts per chunk, model-based merge | Long documents, maximum coverage |

## Setup

```bash
cd hf/data-extractor
make install
```

## Quick start

```bash
# V1 — truncate (default)
make run FILE=path/to/document.txt

# V2 — chunked (full document)
make run FILE=path/to/document.pdf STRATEGY=chunked

# Compare both on the same document
make run FILE=doc.pdf STRATEGY=truncate
make run FILE=doc.pdf STRATEGY=chunked
```

## Make targets

| Target | Description |
|---|---|
| `make install` | Create `.venv` and install all dependencies |
| `make run FILE=...` | Run extraction (default strategy: truncate) |
| `make run FILE=... STRATEGY=chunked` | Run chunked extraction |
| `make test` | Run the test suite |
| `make lint` | Check code with ruff |
| `make format` | Auto-format code with ruff |

`MODEL` and `STRATEGY` can be overridden:

```bash
make run FILE=doc.txt MODEL=mlx-community/gemma-3-4b-it-4bit STRATEGY=chunked
```

## CLI flags

```
python main.py FILE [--model MODEL_PATH] [--strategy {truncate,chunked}]
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `EXTRACTOR_MODEL_PATH` | `mlx-community/gemma-4-12B-4bit` | HuggingFace repo ID or local MLX model directory |
| `EXTRACTOR_MAX_CHARS` | `100000` | Maximum characters for truncate strategy |
| `EXTRACTOR_MAX_RETRIES` | `3` | Number of extraction attempts before giving up |
| `EXTRACTOR_CHUNK_CHARS` | `16000` | Target chunk size for chunked strategy |
| `EXTRACTOR_OVERLAP_CHARS` | `800` | Overlap between consecutive chunks |

## Output schema

Each extracted document produces:

| Field | Type | Notes |
|---|---|---|
| `title` | `string \| null` | Article or document title if present |
| `topic` | `string` | Required — one-line description of the subject matter |
| `people` | `list[Person]` | Each with `name`, optional `role`, optional `context` |
| `places` | `list[Place]` | Each with `name`, optional `context` |
| `dates` | `list[Date]` | Each with `date` (raw string) and optional `event` |
| `summary` | `string` | Required — short prose summary |

Dates are preserved as written in the source text ("mid-1930s", "early July") rather than being normalized to a machine-readable format.

## How it works

### V1 — truncate

1. Load the file — `.txt` read directly, `.pdf` extracted with `pypdf`
2. Truncate to `MAX_CHARS` with a visible warning if the document exceeds that limit
3. Build a prompt with the full JSON schema and strict formatting rules
4. Call the local model and parse the response
5. Validate with Pydantic — on failure, append the error and retry
6. Display the result as a formatted table with `rich`

### V2 — chunked

1. Load the full file with no truncation
2. Split into overlapping chunks (~4,000 chars with ~400 char overlap, snapping to paragraph boundaries)
3. Run the same extraction prompt on each chunk independently
4. Collect all partial `ExtractedDocument` objects
5. Pass them to a model-based merge pass — the model deduplicates people, places, and dates by identity (not just exact string match), then writes a unified topic and summary
6. Validate the merged result with Pydantic and display

## Retry behaviour

Models frequently wrap JSON in markdown code fences or add explanatory preamble. The extractor strips fences automatically before parsing. If the response still can't be parsed or fails schema validation, the error is appended to the conversation and the model is asked to correct it. After `MAX_RETRIES` failures the tool exits with a non-zero code.

This applies to both per-chunk extraction and the merge pass.

## Chunking

Chunk boundaries snap to the nearest paragraph break (`\n\n`) within a tolerance window to avoid splitting mid-sentence. Each chunk overlaps with the previous by `OVERLAP_CHARS` characters so entities straddling a boundary appear in full context in at least one chunk.

## Known limitations

**Small models may hallucinate** — a 4B 4-bit model has limited instruction-following reliability, particularly in the merge pass where it must reason about identity across multiple partial results. Larger models (12B, 27B) from `mlx-community` will produce more accurate extractions and deduplication. Schema errors are caught by Pydantic; factual errors are not.

**Chunked strategy is slower** — extraction and merging happen sequentially: extract chunk 1, merge with chunk 2, merge with chunk 3, and so on. An N-chunk document requires N extractions and N-1 merge calls.

**Context window and the merge pass** — merging all N partials in one model call was the original V2 design, but it fails on longer documents. When 7 or more chunks each produce 10–20 extracted entities, the combined JSON sent to the merge prompt can exceed the model's context window (~8,192 tokens for Gemma 3 4B). The model returns an empty string when its input exceeds this limit, causing an "Invalid JSON" parse failure. The fix is pairwise sequential merging: each merge call sees exactly 2 inputs (the accumulated result so far + the next chunk's partial), keeping input size bounded regardless of document length.

## Tests

```bash
make test
```

66 tests covering `schema.py`, `extractor.py`, `chunker.py`, `merger.py`, and `main.py`. All run without a real model — dependencies are mocked.

## Model compatibility

`mlx-lm` loads **text-only** models. Vision-language models require `mlx-vlm`. Use a text-only model from `mlx-community` on HuggingFace.

Tested and working: `mlx-community/gemma-4-12B-4bit`
