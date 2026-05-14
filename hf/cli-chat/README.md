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

## Metrics

After each assistant response a dim status line shows performance stats:

```
42 tokens · 38.3 tok/s · TTFT 0.91s
```

On a cancelled generation (Ctrl+C mid-stream), partial stats are shown for what was collected before interruption.

### Backends

Two backends are available, selectable via `CHAT_METRICS`:

```bash
CHAT_METRICS=manual python chat.py   # default
CHAT_METRICS=mlx python chat.py
```

| | `manual` (default) | `mlx` |
|---|---|---|
| Token count | Chunks yielded by `stream_generate` | `chunk.generation_tokens` (cumulative, from last chunk) |
| TPS | Computed from wall-clock time | `chunk.generation_tps` (measured inside the MLX kernel) |
| TTFT | Time from prompt submission to first chunk | Not available |
| On cancel | Shows partial stats | Shows `(generation cancelled — no metrics)` |

### Assumptions and accuracy caveats

**`manual` backend**
- *Token count is approximate.* `stream_generate` does not guarantee one chunk per token — it may batch multiple tokens into a single yield. The count is chunks received, not vocabulary tokens.
- *TPS is noisy.* Wall-clock time includes Python overhead, `print()` latency, and terminal rendering. It is slower than what the model actually spent generating.
- *TTFT includes prompt processing.* The timer starts before `stream_generate` is called, so TTFT covers both prompt tokenization and the first forward pass — there is no way to separate the two from outside the kernel.

**`mlx` backend**
- *TPS is accurate.* Measured inside the MLX kernel, unaffected by Python or terminal overhead. This is the more trustworthy throughput figure.
- *No partial stats on cancel.* The aggregate fields (`generation_tps`, `generation_tokens`) are only reliable on the final chunk. If generation is interrupted, the last chunk never arrives and no stats are shown.
- *No TTFT.* MLX does not expose prompt processing time separately.

## Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

31 tests covering `history.py`, `commands.py`, `model.py`, and `metrics.py`. `chat.py` is verified manually.
