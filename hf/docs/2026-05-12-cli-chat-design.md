# CLI Chat with Memory — Design Spec

**Project:** `hf/cli-chat/`
**Date:** 2026-05-12
**Stack:** Python, `mlx-lm`, `rich`
**Goal:** Learn the core LLM application loop — inference, streaming, conversation history — using a locally hosted MLX model on Apple Silicon.

**Verified working model:** `mlx-community/gemma-3-4b-it-4bit` (Gemma 3 4B, 4-bit quantized, ~2.5 GB). Set via `CHAT_MODEL_PATH` env var — `mlx-lm` accepts HuggingFace repo IDs directly and downloads on first run.

---

## Overview

A conversational REPL in the terminal. The user types messages, the model streams responses token-by-token, and the full conversation persists to disk between sessions. Slash commands allow runtime control of history and the system prompt.

---

## File Layout

```
hf/
├── docs/
│   ├── follow-along-projects.md
│   └── 2026-05-12-cli-chat-design.md
├── cli-chat/
│   ├── chat.py          # entry point — REPL loop
│   ├── model.py         # MLX model loading + streaming inference
│   ├── history.py       # load, save, append, clear conversation history
│   ├── commands.py      # parse and dispatch slash commands
│   ├── config.py        # model path, history file path, defaults
│   └── requirements.txt
├── doc-qa/              # Project 2 (future)
├── writing-assistant/   # Project 3 (future)
└── data-extractor/      # Project 4 (future)
```

---

## Data Flow

1. `chat.py` starts → `config.py` resolves paths → `history.py` loads JSON from disk → `model.py` loads MLX model (Rich spinner shown during load)
2. User types input → `commands.py` checks for leading `/` → if a command, dispatch and loop back; if not, treat as chat message
3. User message appended to in-memory history → full history + system prompt passed to `model.py` → tokens streamed back via `Rich Live`
4. Completed response appended to history → `history.py` saves JSON to disk immediately

---

## History File Schema

Stored at a path defined in `config.py` (e.g. `~/.cli-chat-history.json`).

```json
{
  "system": "You are a helpful assistant.",
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

---

## Component Responsibilities

### `config.py`
- Model path, history file path, default system prompt
- Single source of truth for all paths and defaults
- No logic — constants and path resolution only

### `model.py`
- `load_model(model_path)` — returns a loaded MLX model + tokenizer
- `stream_response(model, tokenizer, messages)` — generator that yields string tokens one at a time
- No UI logic; pure inference

### `history.py`
- `load(path)` — reads JSON; returns empty history if file doesn't exist
- `save(path, history)` — writes JSON atomically
- `append(history, role, content)` — mutates history in place
- History is a plain dict: `{"system": str, "messages": list}`

### `commands.py`
- `handle(line, history, path)` — returns `True` if line was a command (consumed), `False` if not
- Supported commands:
  - `/quit` — exit the process
  - `/clear` — empty messages, keep system prompt, save to disk
  - `/history` — print all turns using Rich Panels (user=green, assistant=blue)
  - `/system "..."` — set new system prompt, clear messages, save to disk

### `chat.py`
- Loads config, history, model
- REPL loop: read input → `commands.handle()` → if not a command: append user message, stream response with `Rich Live`, append assistant response, save history
- Catches `KeyboardInterrupt` mid-stream: print newline, return to prompt (do not exit)
- Catches `KeyboardInterrupt` at input prompt: exit with goodbye message

---

## Streaming & UI

- **Model loading:** Rich spinner with status text showing the model path from config
- **User prompt:** styled `[bold green]you >[/]`
- **Assistant label:** dim prefix `[dim]gemma >[/]`
- **Streaming:** `Rich Live` context re-renders after each token — response builds in place
- **`/history`:** each turn rendered in a `Rich Panel`, user turns green, assistant turns blue

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Model path doesn't exist | Clear error at startup, exit immediately |
| History JSON malformed | Warn user, start with fresh empty history |
| `Ctrl+C` mid-stream | Catch `KeyboardInterrupt`, discard incomplete user turn, return to prompt |
| `Ctrl+C` at input prompt | Exit cleanly with goodbye message |
| Generation error (non-KeyboardInterrupt) | Discard incomplete user turn, print error, return to prompt |
| History save failure (disk full, permissions) | Print yellow warning, continue session |

No retry logic or inference fallbacks — MLX either works or it doesn't.

---

## Dependencies

```
mlx-lm>=0.31.3       # MLX inference + tokenization for Apple Silicon
huggingface_hub>=1.14.0  # model download from HuggingFace Hub (used by mlx-lm)
rich>=13.0.0          # terminal UI: spinner, streaming output, styled panels
pytest>=8.0.0         # test runner
```

**Model compatibility note:** `mlx-lm` supports text-only models. Gemma 4 (`gemma-4-E4B`) is a vision-language model requiring `mlx-vlm` — it will fail with "parameters not in model" if loaded via `mlx-lm`. Use a text-only model such as `mlx-community/gemma-3-4b-it-4bit`.

`CHAT_MODEL_PATH` accepts either a local directory path or a HuggingFace repo ID (e.g. `mlx-community/gemma-3-4b-it-4bit`). When a repo ID is given, `mlx-lm` downloads and caches the model on first run.

---

## What You'll Learn

- The MLX inference loop: loading a model, building a prompt from message history, streaming tokens
- How chat history is structured and passed to a model (the messages array format)
- How to stream LLM output to a terminal in real time
- Persisting and loading state between sessions
- Structuring a small Python app with single-responsibility modules
