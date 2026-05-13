# CLI Chat with Memory — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a terminal chat REPL backed by a local MLX/Gemma 4 model, with persistent conversation history and slash commands.

**Architecture:** Five focused modules (`config`, `history`, `commands`, `model`, `chat`) each with one responsibility, wired together in the REPL loop in `chat.py`. History is a plain dict serialized to JSON on every turn. Inference is streamed token-by-token via `mlx_lm.stream_generate`.

**Tech Stack:** Python 3.11+, `mlx-lm` (inference), `rich` (terminal UI), `pytest` (tests)

---

## File Map

| File | Purpose |
|---|---|
| `cli-chat/config.py` | Model path, history path, default system prompt — all env-overridable |
| `cli-chat/history.py` | `load`, `save`, `append` — pure functions over the history dict |
| `cli-chat/commands.py` | `handle(line, history, path)` — parse and dispatch slash commands |
| `cli-chat/model.py` | `load_model`, `stream_response` — MLX inference, no UI logic |
| `cli-chat/chat.py` | REPL loop — wires all modules together |
| `cli-chat/requirements.txt` | Pinned dependencies |
| `cli-chat/tests/conftest.py` | `sys.path` setup for imports |
| `cli-chat/tests/test_history.py` | Unit tests for `history.py` |
| `cli-chat/tests/test_commands.py` | Unit tests for `commands.py` |
| `cli-chat/tests/test_model.py` | Unit tests for `model.py` with mocked MLX |

---

## Task 1: Project Setup

**Files:**
- Create: `cli-chat/requirements.txt`
- Create: `cli-chat/tests/conftest.py`
- Create: `cli-chat/tests/__init__.py`

- [ ] **Step 1: Create the directory structure**

```bash
mkdir -p /Users/m8ttyb/workspace/midden-lab/hf/cli-chat/tests
touch /Users/m8ttyb/workspace/midden-lab/hf/cli-chat/tests/__init__.py
```

- [ ] **Step 2: Create requirements.txt**

```
# cli-chat/requirements.txt
mlx-lm>=0.19.0
rich>=13.0.0
pytest>=8.0.0
```

- [ ] **Step 3: Create conftest.py so tests can import project modules**

```python
# cli-chat/tests/conftest.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
```

- [ ] **Step 4: Create and activate a virtual environment, install dependencies**

```bash
cd /Users/m8ttyb/workspace/midden-lab/hf/cli-chat
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: pip installs `mlx-lm`, `rich`, `pytest` without errors.

- [ ] **Step 5: Verify pytest discovers zero tests (correct baseline)**

```bash
cd /Users/m8ttyb/workspace/midden-lab/hf/cli-chat
pytest tests/ -v
```

Expected output: `no tests ran` or `0 passed`.

- [ ] **Step 6: Commit**

```bash
cd /Users/m8ttyb/workspace/midden-lab/hf
git init
git add cli-chat/requirements.txt cli-chat/tests/
git commit -m "chore: scaffold cli-chat project structure"
```

---

## Task 2: config.py

**Files:**
- Create: `cli-chat/config.py`

- [ ] **Step 1: Create config.py**

```python
# cli-chat/config.py
import os
from pathlib import Path

MODEL_PATH = os.environ.get(
    "CHAT_MODEL_PATH",
    str(Path.home() / "models" / "gemma-4"),
)
HISTORY_PATH = os.environ.get(
    "CHAT_HISTORY_PATH",
    str(Path.home() / ".cli-chat-history.json"),
)
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
```

- [ ] **Step 2: Smoke-test config imports cleanly**

```bash
cd /Users/m8ttyb/workspace/midden-lab/hf/cli-chat
python -c "import config; print(config.MODEL_PATH, config.HISTORY_PATH)"
```

Expected: prints two paths without errors.

- [ ] **Step 3: Set CHAT_MODEL_PATH to your actual converted Gemma 4 path**

Find the directory where `mlx_model.safetensors` lives (the output of your HuggingFace conversion). Export it for the rest of the session:

```bash
export CHAT_MODEL_PATH="/path/to/your/gemma-4-mlx"
```

Replace `/path/to/your/gemma-4-mlx` with the actual path. Verify:

```bash
ls $CHAT_MODEL_PATH
```

Expected: shows `config.json`, `*.safetensors`, `tokenizer.json`, etc.

- [ ] **Step 4: Commit**

```bash
cd /Users/m8ttyb/workspace/midden-lab/hf
git add cli-chat/config.py
git commit -m "feat: add config module with env-overridable paths"
```

---

## Task 3: history.py

**Files:**
- Create: `cli-chat/history.py`
- Create: `cli-chat/tests/test_history.py`

- [ ] **Step 1: Write the failing tests**

```python
# cli-chat/tests/test_history.py
import json
import pytest
from pathlib import Path
import history


def test_load_returns_empty_when_file_missing(tmp_path):
    result = history.load(str(tmp_path / "nonexistent.json"))
    assert result == {"system": "", "messages": []}


def test_load_returns_history_from_file(tmp_path):
    data = {"system": "You are a pirate.", "messages": [{"role": "user", "content": "Ahoy"}]}
    p = tmp_path / "history.json"
    p.write_text(json.dumps(data))
    result = history.load(str(p))
    assert result == data


def test_load_returns_none_on_malformed_json(tmp_path):
    p = tmp_path / "history.json"
    p.write_text("not json {{{{")
    result = history.load(str(p))
    assert result is None


def test_save_writes_json_to_disk(tmp_path):
    p = tmp_path / "history.json"
    data = {"system": "Be brief.", "messages": []}
    history.save(str(p), data)
    assert json.loads(p.read_text()) == data


def test_append_adds_user_message(tmp_path):
    h = {"system": "", "messages": []}
    history.append(h, "user", "Hello")
    assert h["messages"] == [{"role": "user", "content": "Hello"}]


def test_append_adds_multiple_messages_in_order(tmp_path):
    h = {"system": "", "messages": []}
    history.append(h, "user", "Hello")
    history.append(h, "assistant", "Hi!")
    assert len(h["messages"]) == 2
    assert h["messages"][1] == {"role": "assistant", "content": "Hi!"}
```

- [ ] **Step 2: Run tests to verify they all fail**

```bash
cd /Users/m8ttyb/workspace/midden-lab/hf/cli-chat
pytest tests/test_history.py -v
```

Expected: 6 failures — `ModuleNotFoundError: No module named 'history'`

- [ ] **Step 3: Implement history.py**

```python
# cli-chat/history.py
import json
from pathlib import Path


def load(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return {"system": "", "messages": []}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def save(path: str, history: dict) -> None:
    Path(path).write_text(json.dumps(history, indent=2))


def append(history: dict, role: str, content: str) -> None:
    history["messages"].append({"role": role, "content": content})
```

- [ ] **Step 4: Run tests to verify they all pass**

```bash
pytest tests/test_history.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/m8ttyb/workspace/midden-lab/hf
git add cli-chat/history.py cli-chat/tests/test_history.py
git commit -m "feat: add history module with load/save/append"
```

---

## Task 4: commands.py

**Files:**
- Create: `cli-chat/commands.py`
- Create: `cli-chat/tests/test_commands.py`

- [ ] **Step 1: Write the failing tests**

```python
# cli-chat/tests/test_commands.py
import json
import pytest
from unittest.mock import patch
import commands


def make_history(system="Be helpful.", messages=None):
    return {"system": system, "messages": messages or []}


def test_returns_false_for_plain_text(tmp_path):
    h = make_history()
    result = commands.handle("Hello there", h, str(tmp_path / "h.json"))
    assert result is False


def test_quit_calls_sys_exit(tmp_path):
    h = make_history()
    with pytest.raises(SystemExit):
        commands.handle("/quit", h, str(tmp_path / "h.json"))


def test_clear_empties_messages_and_saves(tmp_path):
    p = tmp_path / "h.json"
    h = make_history(messages=[{"role": "user", "content": "Hi"}])
    commands.handle("/clear", h, str(p))
    assert h["messages"] == []
    assert json.loads(p.read_text())["messages"] == []


def test_system_sets_prompt_clears_messages_and_saves(tmp_path):
    p = tmp_path / "h.json"
    h = make_history(messages=[{"role": "user", "content": "Hi"}])
    commands.handle('/system "You are a pirate."', h, str(p))
    assert h["system"] == "You are a pirate."
    assert h["messages"] == []
    saved = json.loads(p.read_text())
    assert saved["system"] == "You are a pirate."


def test_system_with_single_quotes(tmp_path):
    p = tmp_path / "h.json"
    h = make_history()
    commands.handle("/system 'Be terse.'", h, str(p))
    assert h["system"] == "Be terse."


def test_system_without_quotes(tmp_path):
    p = tmp_path / "h.json"
    h = make_history()
    commands.handle("/system Be concise.", h, str(p))
    assert h["system"] == "Be concise."


def test_unknown_command_returns_true(tmp_path):
    h = make_history()
    result = commands.handle("/bogus", h, str(tmp_path / "h.json"))
    assert result is True


def test_clear_returns_true(tmp_path):
    h = make_history()
    result = commands.handle("/clear", h, str(tmp_path / "h.json"))
    assert result is True


def test_history_command_returns_true(tmp_path):
    h = make_history()
    result = commands.handle("/history", h, str(tmp_path / "h.json"))
    assert result is True
```

- [ ] **Step 2: Run tests to verify they all fail**

```bash
cd /Users/m8ttyb/workspace/midden-lab/hf/cli-chat
pytest tests/test_commands.py -v
```

Expected: 9 failures — `ModuleNotFoundError: No module named 'commands'`

- [ ] **Step 3: Implement commands.py**

```python
# cli-chat/commands.py
import sys
import history as hist
from rich.console import Console
from rich.panel import Panel

console = Console()


def handle(line: str, history: dict, path: str) -> bool:
    if not line.startswith("/"):
        return False

    parts = line.strip().split(None, 1)
    cmd = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/quit":
        console.print("[dim]Goodbye![/]")
        sys.exit(0)

    elif cmd == "/clear":
        history["messages"].clear()
        hist.save(path, history)
        console.print("[dim]History cleared.[/]")

    elif cmd == "/history":
        _show_history(history)

    elif cmd == "/system":
        prompt = args.strip().strip('"').strip("'")
        if not prompt:
            console.print("[red]Usage: /system \"your prompt here\"[/]")
        else:
            history["system"] = prompt
            history["messages"].clear()
            hist.save(path, history)
            console.print("[dim]System prompt updated. History cleared.[/]")

    elif cmd == "/help":
        console.print(
            "[dim]Commands:[/]\n"
            "  /clear           — clear conversation history\n"
            "  /history         — show conversation history\n"
            "  /system \"...\"    — set system prompt and clear history\n"
            "  /quit            — exit\n"
            "  /help            — show this message"
        )

    else:
        console.print(f"[red]Unknown command: {cmd}. Type /help for commands.[/]")

    return True


def _show_history(history: dict) -> None:
    if history.get("system"):
        console.print(Panel(history["system"], title="system", border_style="dim"))
    if not history["messages"]:
        console.print("[dim]No messages yet.[/]")
        return
    for msg in history["messages"]:
        style = "green" if msg["role"] == "user" else "blue"
        console.print(Panel(msg["content"], title=msg["role"], border_style=style))
```

- [ ] **Step 4: Run tests to verify they all pass**

```bash
pytest tests/test_commands.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/m8ttyb/workspace/midden-lab/hf
git add cli-chat/commands.py cli-chat/tests/test_commands.py
git commit -m "feat: add commands module with slash command dispatch"
```

---

## Task 5: model.py

**Files:**
- Create: `cli-chat/model.py`
- Create: `cli-chat/tests/test_model.py`

- [ ] **Step 1: Write the failing tests**

These tests mock `mlx_lm` so they run without the actual model loaded.

```python
# cli-chat/tests/test_model.py
from unittest.mock import patch, MagicMock
import model


def make_mock_tokenizer(prompt_output="<prompt>"):
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = prompt_output
    return tokenizer


def make_stream_chunk(text):
    chunk = MagicMock()
    chunk.text = text
    return chunk


def test_stream_response_yields_tokens():
    mock_model = MagicMock()
    mock_tokenizer = make_mock_tokenizer()
    history = {
        "system": "Be helpful.",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    chunks = [make_stream_chunk("Hi"), make_stream_chunk("!")]

    with patch("model.stream_generate", return_value=iter(chunks)):
        tokens = list(model.stream_response(mock_model, mock_tokenizer, history))

    assert tokens == ["Hi", "!"]


def test_stream_response_includes_system_in_messages():
    mock_model = MagicMock()
    mock_tokenizer = make_mock_tokenizer()
    history = {
        "system": "You are a pirate.",
        "messages": [{"role": "user", "content": "Ahoy"}],
    }

    with patch("model.stream_generate", return_value=iter([])):
        list(model.stream_response(mock_model, mock_tokenizer, history))

    call_args = mock_tokenizer.apply_chat_template.call_args
    messages_passed = call_args[0][0]
    assert messages_passed[0] == {"role": "system", "content": "You are a pirate."}
    assert messages_passed[1] == {"role": "user", "content": "Ahoy"}


def test_stream_response_skips_system_when_empty():
    mock_model = MagicMock()
    mock_tokenizer = make_mock_tokenizer()
    history = {
        "system": "",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    with patch("model.stream_generate", return_value=iter([])):
        list(model.stream_response(mock_model, mock_tokenizer, history))

    call_args = mock_tokenizer.apply_chat_template.call_args
    messages_passed = call_args[0][0]
    assert messages_passed[0]["role"] == "user"


def test_load_model_calls_mlx_load():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("model.load", return_value=(mock_model, mock_tokenizer)) as mock_load:
        m, t = model.load_model("/some/path")

    mock_load.assert_called_once_with("/some/path")
    assert m is mock_model
    assert t is mock_tokenizer
```

- [ ] **Step 2: Run tests to verify they all fail**

```bash
cd /Users/m8ttyb/workspace/midden-lab/hf/cli-chat
pytest tests/test_model.py -v
```

Expected: 4 failures — `ModuleNotFoundError: No module named 'model'`

- [ ] **Step 3: Implement model.py**

```python
# cli-chat/model.py
from typing import Iterator
from mlx_lm import load, stream_generate


def load_model(model_path: str):
    """Return (model, tokenizer) loaded from model_path."""
    return load(model_path)


def stream_response(model, tokenizer, history: dict) -> Iterator[str]:
    """Yield string tokens one at a time for the next assistant turn."""
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
        yield chunk.text
```

- [ ] **Step 4: Run tests to verify they all pass**

```bash
pytest tests/test_model.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/m8ttyb/workspace/midden-lab/hf
git add cli-chat/model.py cli-chat/tests/test_model.py
git commit -m "feat: add model module wrapping mlx-lm load and stream"
```

---

## Task 6: chat.py — REPL Loop

**Files:**
- Create: `cli-chat/chat.py`

The REPL loop is wiring — it calls the other modules. It's not unit-tested; you'll verify it with a manual smoke test in Task 7.

- [ ] **Step 1: Create chat.py**

```python
# cli-chat/chat.py
import sys
from rich.console import Console

import config
import history as hist
import commands
from model import load_model, stream_response

console = Console()


def main() -> None:
    # Load history
    h = hist.load(config.HISTORY_PATH)
    if h is None:
        console.print("[yellow]Warning: history file was corrupted — starting fresh.[/]")
        h = {"system": config.DEFAULT_SYSTEM_PROMPT, "messages": []}
    if not h.get("system"):
        h["system"] = config.DEFAULT_SYSTEM_PROMPT

    # Load model
    with console.status("[bold green]Loading Gemma 4...[/]", spinner="dots"):
        try:
            model, tokenizer = load_model(config.MODEL_PATH)
        except Exception as e:
            console.print(f"[red]Failed to load model at {config.MODEL_PATH!r}:[/] {e}")
            sys.exit(1)

    console.print("[dim]Model loaded. Type /help for commands, Ctrl+C to exit.[/]\n")

    while True:
        try:
            user_input = console.input("[bold green]you > [/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/]")
            break

        if not user_input:
            continue

        if commands.handle(user_input, h, config.HISTORY_PATH):
            continue

        hist.append(h, "user", user_input)
        console.print("[dim]gemma >[/] ", end="")

        full_response = ""
        try:
            for token in stream_response(model, tokenizer, h):
                full_response += token
                print(token, end="", flush=True)
        except KeyboardInterrupt:
            # Interrupted mid-generation: discard the incomplete user turn
            print()
            h["messages"].pop()
            console.print("[dim](generation cancelled)[/]")
            continue

        print()
        hist.append(h, "assistant", full_response)
        hist.save(config.HISTORY_PATH, h)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite to confirm nothing is broken**

```bash
cd /Users/m8ttyb/workspace/midden-lab/hf/cli-chat
pytest tests/ -v
```

Expected: 19 passed, 0 failed.

- [ ] **Step 3: Commit**

```bash
cd /Users/m8ttyb/workspace/midden-lab/hf
git add cli-chat/chat.py
git commit -m "feat: add REPL loop in chat.py"
```

---

## Task 7: Smoke Test

Manual verification that the full app works end-to-end with your real Gemma 4 model.

- [ ] **Step 1: Confirm CHAT_MODEL_PATH is set to your converted model**

```bash
echo $CHAT_MODEL_PATH
ls $CHAT_MODEL_PATH
```

Expected: lists `config.json`, `*.safetensors`, `tokenizer.json`, and related files.

- [ ] **Step 2: Run the app**

```bash
cd /Users/m8ttyb/workspace/midden-lab/hf/cli-chat
source .venv/bin/activate
python chat.py
```

Expected: spinner appears ("Loading Gemma 4..."), then the `you >` prompt.

- [ ] **Step 3: Send a message and verify streaming**

Type: `What is the capital of France?`

Expected: `gemma >` label appears, then tokens stream in one-by-one, landing on "Paris" (or similar). No crash.

- [ ] **Step 4: Test /history**

Type: `/history`

Expected: Rich panels showing your user message and the assistant's response.

- [ ] **Step 5: Test /system**

Type: `/system "You are a pirate. Respond only in pirate speak."`

Expected: confirms system prompt updated, history cleared.

Send a message: `Hello`

Expected: pirate-flavored response streamed to terminal.

- [ ] **Step 6: Test /clear**

Type: `/clear`

Expected: "History cleared." message. Type `/history` to confirm it's empty.

- [ ] **Step 7: Test Ctrl+C mid-stream**

Start a message that will generate a long response (e.g., `Explain quantum computing in detail`). Press `Ctrl+C` while it's streaming.

Expected: generation stops, "(generation cancelled)" appears, `you >` prompt returns. App does not exit.

- [ ] **Step 8: Test session persistence**

Exit with `Ctrl+C` at the `you >` prompt. Restart:

```bash
python chat.py
```

Type: `/history`

Expected: previous conversation messages appear (from before you ran `/clear`).

- [ ] **Step 9: Final commit**

```bash
cd /Users/m8ttyb/workspace/midden-lab/hf
git add .
git commit -m "chore: complete cli-chat smoke test — all features verified"
```
