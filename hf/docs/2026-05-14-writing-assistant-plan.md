# Writing Assistant — Implementation Plan

**Goal:** Build a web-based writing assistant with a FastAPI backend and a single-page HTML frontend. Users paste text, choose a rewrite mode, and receive the model's response streamed token-by-token via Server-Sent Events.

**Architecture:** Four focused modules (`config`, `prompts`, `model`, `main`) plus a single static HTML file. The model is loaded once at startup and stored on `app.state`. Each generation request runs `mlx-lm`'s blocking `stream_generate` in a thread pool, pushing tokens into an `asyncio.Queue` that the async endpoint drains and formats as SSE messages.

**Tech Stack:** Python 3.11+, `mlx-lm` (inference), `fastapi` + `uvicorn` (server), `httpx` (test client), `pytest` + `pytest-anyio` (async tests), vanilla HTML/JS (frontend)

**Modes supported:** `rewrite`, `summarize`, `make_formal`, `make_casual`

---

## File Map

| File | Purpose |
|---|---|
| `writing-assistant/config.py` | Env-overridable constants: `MODEL_PATH`, `HOST`, `PORT` |
| `writing-assistant/prompts.py` | `build_messages(text, mode)` — maps mode names to system prompts and returns a messages list |
| `writing-assistant/model.py` | `load_model(path)`, `stream_response(model, tokenizer, messages)` — MLX inference, no web logic |
| `writing-assistant/main.py` | FastAPI app — lifespan model loading, `GET /`, `POST /generate` SSE endpoint |
| `writing-assistant/static/index.html` | Single-page UI — textarea, mode selector, output area, EventSource JS |
| `writing-assistant/requirements.txt` | Pinned dependencies |
| `writing-assistant/tests/__init__.py` | Empty |
| `writing-assistant/tests/conftest.py` | `sys.path` setup for imports |
| `writing-assistant/tests/test_prompts.py` | Unit tests for `prompts.py` |
| `writing-assistant/tests/test_generate.py` | Integration tests for `POST /generate` with mocked model |

---

## Task 1: Project Setup

**Files:**
- Create: `writing-assistant/requirements.txt`
- Create: `writing-assistant/tests/__init__.py`
- Create: `writing-assistant/tests/conftest.py`
- Create: `writing-assistant/static/` directory

- [ ] **Step 1: Create the directory structure**

```bash
mkdir -p /Users/m8ttyb/workspace/fobi/hf/writing-assistant/tests
mkdir -p /Users/m8ttyb/workspace/fobi/hf/writing-assistant/static
touch /Users/m8ttyb/workspace/fobi/hf/writing-assistant/tests/__init__.py
```

- [ ] **Step 2: Create requirements.txt**

```
mlx-lm>=0.19.0
fastapi>=0.111.0
uvicorn>=0.30.0
httpx>=0.27.0
pytest>=8.0.0
pytest-anyio>=0.0.0
anyio>=4.0.0
```

- [ ] **Step 3: Create conftest.py**

```python
# writing-assistant/tests/conftest.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
```

- [ ] **Step 4: Create and activate a virtual environment, install dependencies**

```bash
cd /Users/m8ttyb/workspace/fobi/hf/writing-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- [ ] **Step 5: Verify pytest discovers zero tests**

```bash
pytest tests/ -v
```

Expected: `no tests ran`.

- [ ] **Step 6: Commit**

```bash
git add writing-assistant/
git commit -m "chore: scaffold writing-assistant project structure"
```

---

## Task 2: config.py

**Files:**
- Create: `writing-assistant/config.py`

- [ ] **Step 1: Create config.py**

```python
# writing-assistant/config.py
import os
from pathlib import Path

MODEL_PATH = os.environ.get(
    "WA_MODEL_PATH",
    "mlx-community/gemma-3-4b-it-4bit",
)
HOST = os.environ.get("WA_HOST", "127.0.0.1")
PORT = int(os.environ.get("WA_PORT", "8000"))
```

- [ ] **Step 2: Smoke-test**

```bash
python -c "import config; print(config.MODEL_PATH, config.HOST, config.PORT)"
```

Expected: prints model path, host, and port without errors.

---

## Task 3: prompts.py

**Files:**
- Create: `writing-assistant/prompts.py`
- Create: `writing-assistant/tests/test_prompts.py`

This module is the only place that knows about modes. `main.py` passes a mode string here and gets back a fully-formed messages list ready for `tokenizer.apply_chat_template`.

- [ ] **Step 1: Write the failing tests**

```python
# writing-assistant/tests/test_prompts.py
import pytest
import prompts


MODES = ["rewrite", "summarize", "make_formal", "make_casual"]


def test_build_messages_returns_list_for_all_modes():
    for mode in MODES:
        result = prompts.build_messages("Some text.", mode)
        assert isinstance(result, list)


def test_build_messages_has_system_and_user_roles():
    messages = prompts.build_messages("Hello world.", "rewrite")
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert "user" in roles


def test_build_messages_user_content_contains_input_text():
    text = "The quick brown fox."
    messages = prompts.build_messages(text, "summarize")
    user_message = next(m for m in messages if m["role"] == "user")
    assert text in user_message["content"]


def test_build_messages_raises_on_unknown_mode():
    with pytest.raises(ValueError):
        prompts.build_messages("Some text.", "unknown_mode")


def test_each_mode_has_distinct_system_prompt():
    system_prompts = set()
    for mode in MODES:
        messages = prompts.build_messages("text", mode)
        system = next(m for m in messages if m["role"] == "system")
        system_prompts.add(system["content"])
    assert len(system_prompts) == len(MODES)
```

- [ ] **Step 2: Run tests — verify they all fail**

```bash
pytest tests/test_prompts.py -v
```

Expected: all failures — `ModuleNotFoundError: No module named 'prompts'`

- [ ] **Step 3: Implement prompts.py**

```python
# writing-assistant/prompts.py

_SYSTEM_PROMPTS = {
    "rewrite": (
        "You are a writing assistant. Rewrite the user's text to improve clarity, "
        "flow, and conciseness while preserving the original meaning. "
        "Return only the rewritten text, no commentary."
    ),
    "summarize": (
        "You are a writing assistant. Summarize the user's text into a concise "
        "paragraph that captures the key points. "
        "Return only the summary, no commentary."
    ),
    "make_formal": (
        "You are a writing assistant. Rewrite the user's text in a formal, "
        "professional tone suitable for business or academic contexts. "
        "Return only the rewritten text, no commentary."
    ),
    "make_casual": (
        "You are a writing assistant. Rewrite the user's text in a casual, "
        "conversational tone. Return only the rewritten text, no commentary."
    ),
}


def build_messages(text: str, mode: str) -> list[dict]:
    if mode not in _SYSTEM_PROMPTS:
        raise ValueError(f"Unknown mode {mode!r}. Valid modes: {list(_SYSTEM_PROMPTS)}")
    return [
        {"role": "system", "content": _SYSTEM_PROMPTS[mode]},
        {"role": "user", "content": text},
    ]
```

- [ ] **Step 4: Run tests — verify they all pass**

```bash
pytest tests/test_prompts.py -v
```

Expected: 5 passed.

---

## Task 4: model.py

**Files:**
- Create: `writing-assistant/model.py`

This is a direct adaptation of `cli-chat/model.py`. The only change: `stream_response` accepts a pre-built `messages` list instead of a history dict, since the writing assistant has no conversation memory.

```python
# writing-assistant/model.py
from typing import Iterator
from mlx_lm import load, stream_generate


def load_model(model_path: str):
    return load(model_path)


def stream_response(model, tokenizer, messages: list[dict]) -> Iterator[tuple[str, object]]:
    """Yield (token_text, raw_chunk) pairs for the given messages."""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    for chunk in stream_generate(model, tokenizer, prompt, max_tokens=1024):
        yield chunk.text, chunk
```

No tests are required here — the pattern is identical to `cli-chat/model.py` which is already tested. The difference (messages list vs history dict) is tested indirectly through `test_generate.py`.

---

## Task 5: main.py — FastAPI App

**Files:**
- Create: `writing-assistant/main.py`
- Create: `writing-assistant/tests/test_generate.py`

Two things to understand before reading this code:

**Lifespan:** FastAPI's `lifespan` context manager is the modern way to run startup/shutdown logic. The model is loaded once here and stored on `app.state` so every request can access it without reloading.

**The queue pattern:** `stream_generate` is blocking. We can't call it directly in an async function without freezing the event loop. Instead: the async endpoint creates a `queue`, hands the blocking call to `loop.run_in_executor` (a thread pool), and the worker thread puts each token into the queue. The async endpoint awaits tokens from the queue one at a time, yielding each as an SSE message. A `None` sentinel signals the end of the stream.

- [ ] **Step 1: Write the failing tests**

```python
# writing-assistant/tests/test_generate.py
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def make_chunk(text):
    chunk = MagicMock()
    chunk.text = text
    return chunk


@pytest.fixture
def client():
    with patch("main.load_model") as mock_load:
        mock_load.return_value = (MagicMock(), MagicMock())
        from main import app
        with TestClient(app) as c:
            yield c


def test_generate_streams_tokens(client):
    chunks = [make_chunk("The"), make_chunk(" cat")]

    with patch("main.stream_response", return_value=iter([(c.text, c) for c in chunks])):
        with client.stream("POST", "/generate", json={"text": "A cat sat.", "mode": "rewrite"}) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            body = r.read().decode()

    assert "data: The\n\n" in body
    assert "data:  cat\n\n" in body
    assert "event: done" in body


def test_generate_rejects_unknown_mode(client):
    response = client.post("/generate", json={"text": "Hello", "mode": "explode"})
    assert response.status_code == 422


def test_generate_rejects_empty_text(client):
    response = client.post("/generate", json={"text": "", "mode": "rewrite"})
    assert response.status_code == 422


def test_root_serves_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
```

- [ ] **Step 2: Run tests — verify they all fail**

```bash
pytest tests/test_generate.py -v
```

Expected: all failures — `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Implement main.py**

```python
# writing-assistant/main.py
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, field_validator

import config
from model import load_model, stream_response
from prompts import build_messages


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.tokenizer = load_model(config.MODEL_PATH)
    yield


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    text: str
    mode: Literal["rewrite", "summarize", "make_formal", "make_casual"]

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


@app.get("/", response_class=HTMLResponse)
async def root():
    return Path("static/index.html").read_text()


@app.post("/generate")
async def generate(request: GenerateRequest):
    messages = build_messages(request.text, request.mode)
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def run_model():
        try:
            for text, chunk in stream_response(
                app.state.model, app.state.tokenizer, messages
            ):
                loop.call_soon_threadsafe(queue.put_nowait, text)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"error": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, run_model)

    async def event_stream():
        while True:
            token = await queue.get()
            if token is None:
                yield "event: done\ndata: \n\n"
                break
            if isinstance(token, dict) and "error" in token:
                yield f"event: error\ndata: {token['error']}\n\n"
                break
            yield f"data: {token}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

- [ ] **Step 4: Run tests — verify they all pass**

```bash
pytest tests/test_generate.py -v
```

Expected: 4 passed.

---

## Task 6: static/index.html — Frontend

**Files:**
- Create: `writing-assistant/static/index.html`

No tests for this — verified manually in the smoke test.

The JavaScript uses `fetch` with `ReadableStream` rather than `EventSource` because `EventSource` only supports `GET` requests and we need to `POST` a body. We parse the SSE format manually — it's simple enough to do inline.

```html
<!-- writing-assistant/static/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Writing Assistant</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; }
    h1 { font-size: 1.25rem; margin-bottom: 1.5rem; }
    label { display: block; font-size: 0.875rem; font-weight: 500; margin-bottom: 0.25rem; }
    textarea { width: 100%; box-sizing: border-box; padding: 0.5rem; font-size: 0.95rem; border: 1px solid #ccc; border-radius: 4px; resize: vertical; }
    select { padding: 0.4rem 0.6rem; font-size: 0.95rem; border: 1px solid #ccc; border-radius: 4px; }
    button { padding: 0.45rem 1.1rem; font-size: 0.95rem; background: #2563eb; color: white; border: none; border-radius: 4px; cursor: pointer; }
    button:disabled { background: #93c5fd; cursor: not-allowed; }
    .controls { display: flex; gap: 0.75rem; align-items: center; margin: 0.75rem 0; }
    #output { margin-top: 1.5rem; padding: 1rem; background: #f8f8f8; border-radius: 4px; min-height: 80px; font-size: 0.95rem; white-space: pre-wrap; line-height: 1.6; }
    #error { color: #dc2626; margin-top: 0.5rem; font-size: 0.875rem; }
  </style>
</head>
<body>
  <h1>Writing Assistant</h1>

  <label for="input">Your text</label>
  <textarea id="input" rows="6" placeholder="Paste your text here..."></textarea>

  <div class="controls">
    <select id="mode">
      <option value="rewrite">Rewrite</option>
      <option value="summarize">Summarize</option>
      <option value="make_formal">Make Formal</option>
      <option value="make_casual">Make Casual</option>
    </select>
    <button id="run">Run</button>
  </div>

  <div id="error"></div>
  <div id="output"></div>

  <script>
    const runBtn = document.getElementById("run");
    const output = document.getElementById("output");
    const errorEl = document.getElementById("error");

    runBtn.addEventListener("click", async () => {
      const text = document.getElementById("input").value.trim();
      const mode = document.getElementById("mode").value;
      if (!text) return;

      runBtn.disabled = true;
      output.textContent = "";
      errorEl.textContent = "";

      try {
        const response = await fetch("/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, mode }),
        });

        if (!response.ok) {
          errorEl.textContent = `Error: ${response.status} ${response.statusText}`;
          return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const messages = buffer.split("\n\n");
          buffer = messages.pop(); // keep incomplete trailing chunk

          for (const message of messages) {
            if (message.startsWith("event: done")) break;
            if (message.startsWith("event: error")) {
              const data = message.split("\n").find(l => l.startsWith("data: "));
              errorEl.textContent = data ? data.slice(6) : "Unknown error";
              break;
            }
            if (message.startsWith("data: ")) {
              output.textContent += message.slice(6);
            }
          }
        }
      } finally {
        runBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
```

---

## Task 7: Smoke Test

Manual end-to-end verification with the real model.

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 2: Start the server**

```bash
cd /Users/m8ttyb/workspace/fobi/hf/writing-assistant
source .venv/bin/activate
export WA_MODEL_PATH="mlx-community/gemma-3-4b-it-4bit"
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Expected: model loads, server starts, `Uvicorn running on http://127.0.0.1:8000`.

- [ ] **Step 3: Open the UI**

Navigate to `http://127.0.0.1:8000` in a browser.

Expected: the writing assistant page loads.

- [ ] **Step 4: Test each mode**

Paste a paragraph of text and run each mode in turn. For each:
- Tokens should stream into the output area one-by-one
- The Run button should be disabled during generation and re-enable when done

- [ ] **Step 5: Test empty input**

Click Run with an empty textarea.

Expected: nothing happens (the JS guard prevents submission).

- [ ] **Step 6: Commit**

```bash
git add writing-assistant/
git commit -m "feat: add writing-assistant with FastAPI SSE backend and web UI"
```
