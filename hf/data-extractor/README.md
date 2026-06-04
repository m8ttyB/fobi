# data-extractor

A CLI tool for extracting structured entities from unstructured text. Give it a plain text file or PDF and it extracts title, topic, people, places, dates, and a summary — validated against a Pydantic schema and displayed as a formatted table.

## Setup

```bash
cd hf/data-extractor
make install
```

## Quick start

```bash
make run FILE=path/to/document.txt
make run FILE=path/to/document.pdf
```

## Make targets

| Target | Description |
|---|---|
| `make install` | Create `.venv` and install all dependencies |
| `make run FILE=...` | Run extraction on a file |
| `make test` | Run the test suite |
| `make lint` | Check code with ruff |
| `make format` | Auto-format code with ruff |

`MODEL` can be overridden:

```bash
make run FILE=doc.txt MODEL=mlx-community/gemma-3-4b-it-4bit
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `EXTRACTOR_MODEL_PATH` | `mlx-community/gemma-3-4b-it-4bit` | HuggingFace repo ID or local MLX model directory |
| `EXTRACTOR_MAX_CHARS` | `20000` | Maximum characters of document text sent to the model |
| `EXTRACTOR_MAX_RETRIES` | `3` | Number of extraction attempts before giving up |

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

1. Load the file — `.txt` read directly, `.pdf` extracted with `pypdf`
2. Truncate to `MAX_CHARS` with a visible warning if the document exceeds that limit
3. Build a prompt with the full JSON schema and strict formatting rules
4. Call the local model and parse the response
5. Validate with Pydantic — on failure, append the error to the conversation and retry
6. Display the result as a formatted table with `rich`

## Retry behaviour

Models frequently wrap JSON in markdown code fences or add explanatory preamble. The extractor strips fences automatically before parsing. If the response still can't be parsed or fails schema validation, the error is appended to the conversation and the model is asked to correct it. After `MAX_RETRIES` failures the tool exits with a non-zero code.

## Truncation

Documents longer than `MAX_CHARS` (default 20,000 characters) are truncated before sending to the model. A warning is printed showing the original and truncated length:

```
Warning: Document truncated: 87,432 → 20,000 chars. Results cover only the first portion.
```

Raise `EXTRACTOR_MAX_CHARS` to process more of long documents (at the cost of longer generation time and increased risk of hitting the model's context limit).

## Known limitations

**Truncation is lossy** — entities mentioned only in the later portion of a long document will not be extracted. V2 will add a chunked extraction mode that processes the full document in segments and merges results.

**Small models may miss entities** — a 4B 4-bit model has limited instruction-following reliability. Larger models (12B, 27B) from `mlx-community` will produce more complete and accurate extractions.

## Tests

```bash
make test
```

40 tests covering `schema.py`, `extractor.py`, and `main.py`. All run without a real model — dependencies are mocked.

## Model compatibility

`mlx-lm` loads **text-only** models. Vision-language models require `mlx-vlm`. Use a text-only model from `mlx-community` on HuggingFace.

Tested and working: `mlx-community/gemma-3-4b-it-4bit`
