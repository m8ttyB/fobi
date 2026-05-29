# doc-qa — Readline History Navigation Plan

## Context

The `cmd_chat` REPL uses Python's built-in `input()` for user input. By default, `input()` does not support up-arrow history navigation — each prompt starts blank. Adding readline support lets the user scroll through prior questions in the current session without retyping them.

## How readline works

Python's `readline` module hooks directly into `input()` — no changes to the input call itself are needed. Importing the module at startup is sufficient to enable:

- **Up/down arrow** — scroll through session history
- **Left/right arrow** — cursor movement within the current line
- **Ctrl+A / Ctrl+E** — jump to start/end of line
- **Ctrl+R** — reverse search through history

`readline` is part of the Python standard library on macOS and Linux. On Windows it is not available; the implementation must degrade gracefully (plain `input()` with no history) rather than crashing.

## Design decisions

- **Scope:** In-session only — history is not persisted to disk between runs.
- **Filtering:** All input is added to history, including `/exit` and `/clear`.
- **History length:** Capped at `MAX_HISTORY_TURNS * 2` entries (same constant already used for conversation history) — keeps the implementation tied to one config value.
- **Graceful degradation:** `import readline` is wrapped in a `try/except ImportError` so the tool works on Windows without modification.

## Changes

### `main.py`

Add a module-level readline initialisation block:

```python
try:
    import readline
    readline.set_history_length(config.MAX_HISTORY_TURNS * 2)
except ImportError:
    pass  # Windows — plain input() used, no history navigation
```

That is the entire implementation. No changes to `cmd_chat`, no new modules, no new config constants. `input()` picks up readline automatically once the module is imported.

Add a module docstring update noting readline support and its graceful degradation on Windows.

### `tests/test_main.py`

Add one test confirming that importing `main` does not raise even when `readline` is unavailable:

```python
def test_readline_import_failure_does_not_crash():
    import sys
    with patch.dict(sys.modules, {"readline": None}):
        import importlib
        import main as m
        importlib.reload(m)  # re-executes module-level code with readline absent
```

### `README.md`

Add a note under **Chat commands** documenting the keyboard shortcuts available in the REPL.

### `CLAUDE.md`

Note the readline initialisation in the `main.py` architecture description.

## Files changed

| File | Change |
|---|---|
| `main.py` | Add readline initialisation block with graceful degradation; update module docstring |
| `tests/test_main.py` | Add test for readline import failure |
| `README.md` | Document readline keyboard shortcuts under Chat commands |
| `CLAUDE.md` | Note readline in main.py architecture section |

## Implementation order

1. Add readline initialisation to `main.py` and update module docstring
2. Write readline import failure test → confirm it passes
3. Update `README.md` — add keyboard shortcuts table under Chat commands
4. Update `CLAUDE.md` — note readline in main.py description
5. Full test suite + lint clean
6. Smoke test: launch `make chat`, ask a question, press up-arrow and confirm the question reappears
7. Commit

## Smoke test

```bash
make chat
> What is Bandelier Tuff?   # ask a question
> [up arrow]                # should repopulate "What is Bandelier Tuff?"
> [enter]                   # re-run the same question
```

## Out of scope

- Persistent history across sessions
- Filtering slash commands from history
- Custom readline keybindings
- Tab completion of slash commands
