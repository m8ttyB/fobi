# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Environment setup

```bash
make install
export WA_MODEL_PATH="mlx-community/gemma-3-4b-it-4bit"  # downloads ~2.5 GB on first run
```

`WA_MODEL_PATH` accepts a HuggingFace repo ID or a local directory of MLX-converted weights. `WA_HOST` and `WA_PORT` override the server bind address and port (defaults: `127.0.0.1`, `8000`).

## Running

```bash
make run
# or with overrides:
make run MODEL=mlx-community/gemma-3-4b-it-4bit PORT=9000
```

Server starts at `http://127.0.0.1:8000`. The model is loaded once at startup — expect a delay on first run while weights download (~2.5 GB).

## Tests

```bash
# All tests (no model required — mlx_lm is mocked)
make test

# Single file
.venv/bin/pytest tests/test_prompts.py -v

# Single test
.venv/bin/pytest tests/test_generate.py::test_generate_streams_tokens -v
```

## Linting

```bash
make lint    # check
make format  # auto-fix
```

## Architecture

Five modules wired together in `main.py`:

- **`config.py`** — env-overridable constants only (`MODEL_PATH`, `HOST`, `PORT`). No logic.
- **`fetcher.py`** — `fetch_and_extract(url) -> ExtractedContent`. Uses `httpx` (sync) for the HTTP request and `trafilatura` for article extraction. Defines a typed exception hierarchy (`InvalidURLError`, `FetchTimeoutError`, `FetchHTTPError`, `UnsupportedContentTypeError`, `ExtractionEmptyError`) that `main.py` maps to HTTP status codes. Enforces: 10s timeout, 5 MB max body, `text/html` content-type only, 20K-char truncation cap.
- **`prompts.py`** — `build_messages(text, mode)` maps a mode string to a system prompt and returns a messages list ready for `apply_chat_template`. Single source of truth for mode definitions. Raises `ValueError` on unknown modes.
- **`model.py`** — thin wrapper over `mlx_lm`. `load_model(path)` returns `(model, tokenizer)`. `stream_response(model, tokenizer, messages)` builds the prompt via `tokenizer.apply_chat_template` and yields `(chunk.text, chunk)` tuples. No web logic here.
- **`main.py`** — FastAPI app. Model is loaded once at startup via the `lifespan` context manager and stored on `app.state`. `POST /fetch` calls `fetch_and_extract` via `asyncio.to_thread` (it's blocking) and maps typed exceptions to HTTP statuses. `POST /generate` bridges the blocking MLX generator to an async SSE stream using `asyncio.Queue` + `loop.run_in_executor`. A `None` sentinel in the queue signals end-of-stream.
- **`static/index.html`** — single-page UI with a two-stage URL flow. Detects `^https?://` on click; calls `/fetch` first, populates an editable preview, then calls `/generate` on the second click. Uses `fetch` + `ReadableStream` rather than `EventSource` because `EventSource` only supports GET requests.

## Key design decisions

**Why `asyncio.Queue` + `run_in_executor`?** `mlx_lm.stream_generate` is a blocking synchronous generator. Calling it directly inside an `async` endpoint would freeze FastAPI's event loop for the entire duration of generation. The queue bridges the blocking thread world and the async world: the worker thread pushes tokens via `loop.call_soon_threadsafe`, and the async endpoint awaits them one at a time.

**Why `fetch` instead of `EventSource` on the frontend?** The browser's `EventSource` API only supports GET requests. We need POST to send `{text, mode}` in the request body. `fetch` with `ReadableStream` gives equivalent streaming behaviour with full control over the request.

**Why no conversation history?** Each request is a stateless single-shot transform. There is no multi-turn context to preserve, so the history dict pattern from `cli-chat` is intentionally absent.

**Why `asyncio.to_thread` for `/fetch` but `run_in_executor` + `Queue` for `/generate`?** `fetch_and_extract` is blocking but returns a single value — `to_thread` is the clean one-liner for that pattern. `stream_generate` is a blocking *iterator* that produces values incrementally, so it needs a queue to hand each token back to the async endpoint as it arrives; `to_thread` can't do that.
