# CLI Chat with Memory

A terminal REPL that streams responses from a local MLX model, with persistent conversation history and slash commands. Part of the `hf/` LLM learning project series.

## Setup

```bash
cd hf/cli-chat
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Model

Set the model via environment variable before running. `mlx-lm` accepts a HuggingFace repo ID directly — it downloads and caches the model on first run (~2.5 GB):

```bash
export CHAT_MODEL_PATH="mlx-community/gemma-3-4b-it-4bit"
python chat.py
```

You can also point to a local directory of MLX-converted weights:

```bash
export CHAT_MODEL_PATH="/path/to/local/mlx-model"
python chat.py
```

### Model compatibility

`mlx-lm` loads **text-only** models. If you get a "parameters not in model" error, the model is likely a vision-language model (e.g. Gemma 4 E4B), which requires `mlx-vlm` instead. Use a text-only model like the one above.

Tested and working: `mlx-community/gemma-3-4b-it-4bit` (Gemma 3 4B, 4-bit quantized, Apple Silicon).

## Commands

| Command | Effect |
|---|---|
| `/help` | Show available commands |
| `/history` | Display conversation history in Rich panels |
| `/clear` | Clear conversation history (keeps system prompt) |
| `/system "..."` | Set a new system prompt and clear history |
| `/quit` | Exit |
| `Ctrl+C` (mid-stream) | Cancel current generation, return to prompt |
| `Ctrl+C` (at prompt) | Exit |

## History

Conversation history is saved to `~/.cli-chat-history.json` after every turn and loaded on startup. Override the path:

```bash
export CHAT_HISTORY_PATH="/path/to/history.json"
```

## Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

19 tests covering `history.py`, `commands.py`, and `model.py`. `chat.py` is verified manually.
