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

Four modules wired together in `main.py`:

- **`config.py`** — env-overridable constants only (`MODEL_PATH`, `HOST`, `PORT`). No logic.
- **`prompts.py`** — `build_messages(text, mode)` maps a mode string to a system prompt and returns a messages list ready for `apply_chat_template`. Single source of truth for mode definitions. Raises `ValueError` on unknown modes.
- **`model.py`** — thin wrapper over `mlx_lm`. `load_model(path)` returns `(model, tokenizer)`. `stream_response(model, tokenizer, messages)` builds the prompt via `tokenizer.apply_chat_template` and yields `(chunk.text, chunk)` tuples. No web logic here.
- **`main.py`** — FastAPI app. Model is loaded once at startup via the `lifespan` context manager and stored on `app.state`. `POST /generate` bridges the blocking MLX generator to an async SSE stream using `asyncio.Queue` + `loop.run_in_executor`. A `None` sentinel in the queue signals end-of-stream. Errors from the worker thread are forwarded as `event: error` SSE messages.
- **`static/index.html`** — single-page UI. Uses `fetch` + `ReadableStream` rather than `EventSource` because `EventSource` only supports GET requests. SSE messages are parsed manually from the stream buffer.

## Key design decisions

**Why `asyncio.Queue` + `run_in_executor`?** `mlx_lm.stream_generate` is a blocking synchronous generator. Calling it directly inside an `async` endpoint would freeze FastAPI's event loop for the entire duration of generation. The queue bridges the blocking thread world and the async world: the worker thread pushes tokens via `loop.call_soon_threadsafe`, and the async endpoint awaits them one at a time.

**Why `fetch` instead of `EventSource` on the frontend?** The browser's `EventSource` API only supports GET requests. We need POST to send `{text, mode}` in the request body. `fetch` with `ReadableStream` gives equivalent streaming behaviour with full control over the request.

**Why no conversation history?** Each request is a stateless single-shot transform. There is no multi-turn context to preserve, so the history dict pattern from `cli-chat` is intentionally absent.
