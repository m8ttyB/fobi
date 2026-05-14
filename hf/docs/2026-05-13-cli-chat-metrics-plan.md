# CLI Chat Metrics — Implementation Plan

**Goal:** Add per-turn performance metrics (tokens, TPS, TTFT) to `cli-chat` using an A/B approach — two backends selectable via the `CHAT_METRICS` env var — so we can compare what `mlx-lm` exposes natively against manual instrumentation and pick a winner.

**Decisions recorded:**
- Interrupted generation: show partial stats (tokens collected so far)
- Token count (manual backend): approximate — count chunks yielded
- Scope: per-turn only, no session totals
- Display: dim status line on a new line after each response

**Backend comparison:**

| | `mlx` backend | `manual` backend |
|---|---|---|
| TPS | Accurate (measured in kernel) | Noisy (includes Python + print overhead) |
| Token count | Exact (from `GenerationResponse`) | Approximate (chunks yielded) |
| TTFT | Not available | Available |
| Partial stats on cancel | Not available (final chunk never arrives) | Available |

---

## File Map

| File | Change |
|---|---|
| `cli-chat/metrics.py` | New — `MetricsCollector` class, `format_stats()`, `print_stats()` |
| `cli-chat/model.py` | Modify — yield `(text, chunk)` tuples instead of bare `text` |
| `cli-chat/chat.py` | Modify — wire in `MetricsCollector`, call `print_stats()` after each turn |
| `cli-chat/config.py` | Modify — add `METRICS_BACKEND` env var constant |
| `cli-chat/tests/test_metrics.py` | New — unit tests for both backends |
| `cli-chat/tests/test_model.py` | Modify — update assertions for new yield shape |

---

## Task 1: Add `METRICS_BACKEND` to config.py

**Files:** `cli-chat/config.py`

- [ ] **Step 1: Add the env var constant**

```python
METRICS_BACKEND = os.environ.get("CHAT_METRICS", "manual")  # "manual" | "mlx"
```

- [ ] **Step 2: Smoke-test**

```bash
cd /Users/m8ttyb/workspace/fobi/hf/cli-chat
source .venv/bin/activate
python -c "import config; print(config.METRICS_BACKEND)"
```

Expected: prints `manual`.

```bash
CHAT_METRICS=mlx python -c "import config; print(config.METRICS_BACKEND)"
```

Expected: prints `mlx`.

---

## Task 2: Create metrics.py

**Files:** `cli-chat/metrics.py`, `cli-chat/tests/test_metrics.py`

This module owns all metrics logic. `chat.py` creates a `MetricsCollector`, passes it into the generation loop, then calls `print_stats()` at the end. Neither `model.py` nor `chat.py` knows about the two backends — that's entirely `MetricsCollector`'s concern.

- [ ] **Step 1: Write the failing tests**

```python
# cli-chat/tests/test_metrics.py
import time
from unittest.mock import MagicMock
import metrics


def make_chunk(text, generation_tps=45.0, generation_tokens=10, prompt_tokens=20):
    chunk = MagicMock()
    chunk.text = text
    chunk.generation_tps = generation_tps
    chunk.generation_tokens = generation_tokens
    chunk.prompt_tokens = prompt_tokens
    return chunk


# --- manual backend ---

def test_manual_records_token_count():
    c = metrics.MetricsCollector(backend="manual")
    c.start()
    c.record(make_chunk("Hello"))
    c.record(make_chunk(" world"))
    stats = c.finish()
    assert stats["tokens"] == 2


def test_manual_reports_ttft():
    c = metrics.MetricsCollector(backend="manual")
    c.start()
    time.sleep(0.05)
    c.record(make_chunk("Hi"))
    stats = c.finish()
    assert stats["ttft"] >= 0.05


def test_manual_partial_stats_on_cancel():
    c = metrics.MetricsCollector(backend="manual")
    c.start()
    c.record(make_chunk("Hello"))
    c.record(make_chunk(" world"))
    stats = c.finish(cancelled=True)
    assert stats["tokens"] == 2
    assert stats["cancelled"] is True


def test_manual_tps_is_nonzero_after_tokens():
    c = metrics.MetricsCollector(backend="manual")
    c.start()
    c.record(make_chunk("a"))
    c.record(make_chunk("b"))
    c.record(make_chunk("c"))
    stats = c.finish()
    assert stats["tps"] > 0


# --- mlx backend ---

def test_mlx_uses_chunk_tps():
    c = metrics.MetricsCollector(backend="mlx")
    c.start()
    c.record(make_chunk("Hello", generation_tps=47.5))
    c.record(make_chunk(" world", generation_tps=47.5))
    stats = c.finish()
    assert stats["tps"] == 47.5


def test_mlx_uses_chunk_token_count():
    c = metrics.MetricsCollector(backend="mlx")
    c.start()
    c.record(make_chunk("Hi", generation_tokens=5))
    c.record(make_chunk("!", generation_tokens=6))
    stats = c.finish()
    assert stats["tokens"] == 6  # last chunk's cumulative count


def test_mlx_has_no_ttft():
    c = metrics.MetricsCollector(backend="mlx")
    c.start()
    c.record(make_chunk("Hi"))
    stats = c.finish()
    assert stats.get("ttft") is None


def test_mlx_cancelled_returns_none_stats():
    c = metrics.MetricsCollector(backend="mlx")
    c.start()
    c.record(make_chunk("Hi"))
    stats = c.finish(cancelled=True)
    assert stats["tps"] is None
    assert stats["tokens"] is None


# --- format_stats ---

def test_format_stats_manual_complete():
    stats = {"tokens": 42, "tps": 38.1, "ttft": 1.23, "cancelled": False}
    line = metrics.format_stats(stats, backend="manual")
    assert "42" in line
    assert "38.1" in line
    assert "1.23" in line


def test_format_stats_manual_cancelled():
    stats = {"tokens": 10, "tps": 20.0, "ttft": 0.5, "cancelled": True}
    line = metrics.format_stats(stats, backend="manual")
    assert "cancelled" in line.lower() or "cancel" in line.lower()


def test_format_stats_mlx_no_ttft():
    stats = {"tokens": 30, "tps": 45.0, "ttft": None, "cancelled": False}
    line = metrics.format_stats(stats, backend="mlx")
    assert "TTFT" not in line


def test_format_stats_mlx_cancelled():
    stats = {"tokens": None, "tps": None, "ttft": None, "cancelled": True}
    line = metrics.format_stats(stats, backend="mlx")
    assert "cancelled" in line.lower() or "cancel" in line.lower()
```

- [ ] **Step 2: Run tests to verify they all fail**

```bash
pytest tests/test_metrics.py -v
```

Expected: all failures — `ModuleNotFoundError: No module named 'metrics'`

- [ ] **Step 3: Implement metrics.py**

```python
# cli-chat/metrics.py
import time


class MetricsCollector:
    def __init__(self, backend: str):
        self._backend = backend
        self._start_time: float = 0.0
        self._first_token_time: float | None = None
        self._token_count: int = 0
        self._last_chunk = None

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._first_token_time = None
        self._token_count = 0
        self._last_chunk = None

    def record(self, chunk) -> None:
        if self._first_token_time is None:
            self._first_token_time = time.perf_counter()
        self._token_count += 1
        self._last_chunk = chunk

    def finish(self, cancelled: bool = False) -> dict:
        elapsed = time.perf_counter() - self._start_time

        if self._backend == "mlx":
            if cancelled or self._last_chunk is None:
                return {"tokens": None, "tps": None, "ttft": None, "cancelled": cancelled}
            return {
                "tokens": self._last_chunk.generation_tokens,
                "tps": round(self._last_chunk.generation_tps, 1),
                "ttft": None,
                "cancelled": False,
            }

        # manual backend
        ttft = round(self._first_token_time - self._start_time, 2) if self._first_token_time else None
        tps = round(self._token_count / elapsed, 1) if elapsed > 0 and self._token_count > 0 else 0.0
        return {
            "tokens": self._token_count,
            "tps": tps,
            "ttft": ttft,
            "cancelled": cancelled,
        }


def format_stats(stats: dict, backend: str) -> str:
    if stats.get("cancelled") and stats.get("tokens") is None:
        return "(generation cancelled — no metrics)"

    parts = []

    tokens = stats.get("tokens")
    if tokens is not None:
        parts.append(f"{tokens} tokens")

    tps = stats.get("tps")
    if tps is not None:
        parts.append(f"{tps} tok/s")

    ttft = stats.get("ttft")
    if ttft is not None:
        parts.append(f"TTFT {ttft}s")

    base = " · ".join(parts)

    if stats.get("cancelled"):
        return f"{base} (cancelled)"
    return base


def print_stats(stats: dict, backend: str, console) -> None:
    line = format_stats(stats, backend)
    console.print(f"[dim]{line}[/]")
```

- [ ] **Step 4: Run tests to verify they all pass**

```bash
pytest tests/test_metrics.py -v
```

Expected: all passed.

---

## Task 3: Update model.py to yield chunks

`stream_response` currently yields bare `chunk.text` strings. We need to yield `(text, chunk)` tuples so `chat.py` can pass the raw chunk to `MetricsCollector.record()`.

**Files:** `cli-chat/model.py`, `cli-chat/tests/test_model.py`

- [ ] **Step 1: Update model.py**

Change the `stream_response` signature and yield shape:

```python
# cli-chat/model.py
from typing import Iterator
from mlx_lm import load, stream_generate


def load_model(model_path: str):
    return load(model_path)


def stream_response(model, tokenizer, history: dict) -> Iterator[tuple[str, object]]:
    """Yield (token_text, raw_chunk) pairs for the next assistant turn."""
    messages = []
    if history.get("system"):
        messages.append({"role": "system", "content": history["system"]})
    messages.extend(history["messages"])

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    for chunk in stream_generate(model, tokenizer, prompt, max_tokens=1024):
        yield chunk.text, chunk
```

- [ ] **Step 2: Update test_model.py for the new yield shape**

Update `test_stream_response_yields_tokens` and any assertion that unpacks the yielded values:

```python
def test_stream_response_yields_tokens():
    mock_model = MagicMock()
    mock_tokenizer = make_mock_tokenizer()
    history = {
        "system": "Be helpful.",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    chunks = [make_stream_chunk("Hi"), make_stream_chunk("!")]

    with patch("model.stream_generate", return_value=iter(chunks)):
        results = list(model.stream_response(mock_model, mock_tokenizer, history))

    texts = [text for text, _ in results]
    assert texts == ["Hi", "!"]
```

- [ ] **Step 3: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass (test_history, test_commands, test_model, test_metrics).

---

## Task 4: Wire metrics into chat.py

**Files:** `cli-chat/chat.py`

- [ ] **Step 1: Update the generation loop**

Import `MetricsCollector` and `print_stats`. Create a collector per turn, call `record()` on each chunk, call `print_stats()` after generation ends (including on cancel).

Key changes to the `while True` loop:

```python
# At top of chat.py, add:
import config
from metrics import MetricsCollector, print_stats

# Inside the loop, replace the generation block:
collector = MetricsCollector(backend=config.METRICS_BACKEND)
collector.start()

full_response = ""
try:
    for text, chunk in stream_response(model, tokenizer, h):
        full_response += text
        print(text, end="", flush=True)
        collector.record(chunk)
except KeyboardInterrupt:
    print()
    h["messages"].pop()
    stats = collector.finish(cancelled=True)
    print_stats(stats, config.METRICS_BACKEND, console)
    continue
except Exception as e:
    print()
    h["messages"].pop()
    console.print(f"[red]Generation error: {e}[/]")
    continue

print()
stats = collector.finish()
print_stats(stats, config.METRICS_BACKEND, console)
hist.append(h, "assistant", full_response)
```

- [ ] **Step 2: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

---

## Task 5: Smoke Test — Manual Backend

- [ ] **Step 1: Run with default (manual) backend**

```bash
cd /Users/m8ttyb/workspace/fobi/hf/cli-chat
source .venv/bin/activate
python chat.py
```

- [ ] **Step 2: Send a message, verify metrics line appears**

Type: `What is the capital of France?`

Expected output after response:
```
[dim]18 tokens · 42.3 tok/s · TTFT 0.91s[/]
```

- [ ] **Step 3: Test Ctrl+C mid-stream**

Start a long generation (e.g. `Explain the history of the Roman Empire`). Press Ctrl+C mid-stream.

Expected: partial stats appear, e.g. `7 tokens · 39.1 tok/s · TTFT 0.88s (cancelled)`

---

## Task 6: Smoke Test — MLX Backend

- [ ] **Step 1: Switch to mlx backend**

```bash
CHAT_METRICS=mlx python chat.py
```

- [ ] **Step 2: Send the same message, compare output**

Type: `What is the capital of France?`

Expected: stats line appears without TTFT field:
```
[dim]18 tokens · 47.1 tok/s[/]
```

- [ ] **Step 3: Test Ctrl+C mid-stream**

Expected: `(generation cancelled — no metrics)` — no partial data because the final chunk never arrived.

- [ ] **Step 4: Note observations for A/B decision**

Compare across both backends:
- Do the token counts agree?
- Is TPS meaningfully different between the two?
- Does TTFT from the manual backend feel accurate?
- Which display is more useful in practice?

---

## Task 7: Pick a Winner and Clean Up

Once you've tested both backends and formed an opinion:

- [ ] **Step 1: Delete the losing backend's code path from `metrics.py`**
- [ ] **Step 2: Remove the `METRICS_BACKEND` branching from `MetricsCollector`**
- [ ] **Step 3: Remove `METRICS_BACKEND` from `config.py`**
- [ ] **Step 4: Update `CLAUDE.md` to remove any reference to the A/B flag**
- [ ] **Step 5: Run full test suite one final time**
- [ ] **Step 6: Commit**

```bash
git add cli-chat/
git commit -m "feat: add per-turn metrics display to cli-chat"
```
