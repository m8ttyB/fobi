# Writing Assistant

A web-based writing assistant that streams responses from a local MLX model. Paste text, choose a mode, and see the result appear token-by-token in the browser. Part of the `hf/` LLM learning project series.

## Setup

```bash
cd hf/writing-assistant
make install
```

## Make targets

| Target | Description |
|---|---|
| `make install` | Create `.venv` and install all dependencies |
| `make run` | Start the dev server at `http://127.0.0.1:8000` |
| `make test` | Run the test suite |
| `make lint` | Check code with ruff |
| `make format` | Auto-format code with ruff |

`MODEL`, `HOST`, and `PORT` can be overridden at the command line:

```bash
make run                                        # defaults
make run MODEL=mlx-community/gemma-3-4b-it-4bit PORT=9000
make run MODEL=/path/to/local/mlx-model
```

Then open `http://127.0.0.1:8000` in a browser.

The model is downloaded automatically from HuggingFace on first run (~2.5 GB) and cached at `~/.cache/huggingface/hub/`. Subsequent starts load from cache.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `WA_MODEL_PATH` | `mlx-community/gemma-3-4b-it-4bit` | HuggingFace repo ID or local MLX model directory |
| `WA_HOST` | `127.0.0.1` | Server bind address |
| `WA_PORT` | `8000` | Server port |

## URL input

The input field accepts either pasted text or a URL. If the value starts with `http://` or `https://`, clicking Run fetches the page, extracts the main article content, and shows it in an editable preview before generation runs.

**Flow:**
1. Paste a URL → click **Run** → article text appears in the textarea
2. Review or edit the extracted text (and pick a mode)
3. Click **Run** again → result streams as normal

A **Start over** link appears during preview to clear everything and start fresh.

If the extracted article exceeds 20,000 characters it is truncated and a warning is shown. The character budget corresponds to roughly 5,000 tokens — within the context window of the default model.

**Supported:** HTML pages (articles, blog posts, documentation).  
**Not supported (v1):** PDFs, JavaScript-rendered pages, paywalled content.

### Fetch error messages

| Error | Cause |
|---|---|
| Invalid URL | Input doesn't start with `http://` or `https://` |
| Request timed out | Server waited 10s with no response |
| Could not fetch the URL | HTTP 4xx/5xx from the target server |
| Unsupported content type | URL points to a non-HTML resource (e.g. PDF) |
| No article content found | Page loaded but trafilatura found no extractable article body |

All errors appear inline below the input field; the URL is preserved so you can fix it and retry.

## Writing modes

| Mode | What it does |
|---|---|
| Rewrite | Improves clarity, flow, and conciseness while preserving meaning |
| Summarize | Condenses the text into a concise paragraph covering the key points |
| Make Formal | Rewrites in a professional tone for business or academic contexts |
| Make Casual | Rewrites in a relaxed, conversational tone |

## How streaming works

Responses stream token-by-token using [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) (SSE) over a persistent HTTP connection. The model runs in a background thread so the server stays responsive between tokens. The browser receives each token as it is produced and appends it to the output area in real time.

The `mlx-lm` generator is synchronous and blocking. To avoid freezing FastAPI's async event loop, the generator runs in a thread pool and pushes tokens into an `asyncio.Queue`. The async endpoint drains that queue and formats each token as an SSE message (`data: <token>\n\n`). A named `event: done` message signals completion.

## Model compatibility

`mlx-lm` loads **text-only** models. Vision-language models (e.g. Gemma 4 multimodal) require `mlx-vlm` and will fail with a "parameters not in model" error. Use a text-only model from `mlx-community` on HuggingFace.

Tested and working: `mlx-community/gemma-3-4b-it-4bit` (Gemma 3 4B, 4-bit quantized, Apple Silicon).

## Tests

```bash
make test
```

~24 tests covering `prompts.py`, `fetcher.py`, `POST /fetch`, and `POST /generate` (all with mocked dependencies). `static/index.html` is verified manually.
