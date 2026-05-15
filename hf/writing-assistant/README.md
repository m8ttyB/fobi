# Writing Assistant

A web-based writing assistant that streams responses from a local MLX model. Paste text, choose a mode, and see the result appear token-by-token in the browser. Part of the `hf/` LLM learning project series.

## Setup

```bash
cd hf/writing-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
export WA_MODEL_PATH="mlx-community/gemma-3-4b-it-4bit"
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Then open `http://127.0.0.1:8000` in a browser.

The model is downloaded automatically from HuggingFace on first run (~2.5 GB) and cached at `~/.cache/huggingface/hub/`. Subsequent starts load from cache.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `WA_MODEL_PATH` | `mlx-community/gemma-3-4b-it-4bit` | HuggingFace repo ID or local MLX model directory |
| `WA_HOST` | `127.0.0.1` | Server bind address |
| `WA_PORT` | `8000` | Server port |

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
source .venv/bin/activate
pytest tests/ -v
```

9 tests covering `prompts.py` and the `POST /generate` endpoint (with mocked model). `static/index.html` is verified manually.
