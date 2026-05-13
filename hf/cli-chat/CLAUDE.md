# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export CHAT_MODEL_PATH="mlx-community/gemma-3-4b-it-4bit"  # downloads ~2.5 GB on first run
```

`CHAT_MODEL_PATH` accepts a HuggingFace repo ID or a local directory of MLX-converted weights. `CHAT_HISTORY_PATH` overrides the default history file (`~/.cli-chat-history.json`).

## Running

```bash
python chat.py
```

## Tests

```bash
# All tests
pytest tests/ -v

# Single test file
pytest tests/test_history.py -v

# Single test
pytest tests/test_history.py::test_load_returns_none_on_malformed_json -v
```

Tests for `model.py` mock `mlx_lm` entirely — no model required to run the suite.

## Architecture

Five modules, each with one responsibility, wired together in `chat.py`:

- **`config.py`** — env-overridable constants only (`MODEL_PATH`, `HISTORY_PATH`, `DEFAULT_SYSTEM_PROMPT`). No logic.
- **`history.py`** — pure functions over the history dict: `load`, `save`, `append`. The history dict shape is always `{"system": str, "messages": list[{"role": str, "content": str}]}`. `load` returns `None` on malformed JSON (distinct from a missing file, which returns an empty history).
- **`model.py`** — thin wrapper over `mlx_lm`. `load_model(path)` returns `(model, tokenizer)`. `stream_response(model, tokenizer, history)` builds the prompt via `tokenizer.apply_chat_template`, calls `mlx_lm.stream_generate`, and yields `chunk.text` tokens. No UI logic here.
- **`commands.py`** — `handle(line, history, path) -> bool` returns `False` for non-commands, `True` for all slash commands (consumed). Saves history to disk after `/clear` and `/system`.
- **`chat.py`** — REPL loop. On `KeyboardInterrupt` mid-stream, pops the incomplete user message from history before continuing. On any other generation exception, same cleanup then continues. Wraps `hist.save()` in a try/except to avoid crashing on disk errors.
