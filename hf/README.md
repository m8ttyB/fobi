# Local LLM Projects

Hands-on projects for building LLM-embedded applications using locally hosted models on Apple Silicon via [MLX](https://github.com/ml-explore/mlx) and HuggingFace.

Each project is an independent Python environment built and explored alongside Claude Code — the goal is to understand not just how each pattern works, but why it's built that way.

## Prerequisites

- Apple Silicon Mac (M1 or later)
- Python 3.11+
- ~3 GB free disk space for the default model

Models are downloaded automatically from HuggingFace on first run and cached at `~/.cache/huggingface/hub/`. Use `huggingface-cli scan-cache` to inspect disk usage and `huggingface-cli delete-cache` for cleanup.

## Projects

| Directory | Status | What it teaches |
|---|---|---|
| `cli-chat/` | Complete | Inference loop, chat history, streaming tokens, REPL design |
| `writing-assistant/` | Planned | FastAPI backend, SSE streaming over HTTP, web UI |
| `doc-qa/` | Planned | RAG — chunking, vector search, grounding answers in a document |
| `data-extractor/` | Planned | Structured JSON output, pydantic validation, retry on malformat |

**Recommended build order:** `cli-chat` → `writing-assistant` → `doc-qa` → `data-extractor`

Each project has its own `CLAUDE.md` with setup instructions, commands, and architecture notes.

## Model

All projects default to `mlx-community/gemma-3-4b-it-4bit` (~2.5 GB). Override per-project via an environment variable (see each project's `CLAUDE.md`). Any text-only model from `mlx-community` on HuggingFace should work — avoid vision/multimodal models, which require `mlx-vlm` instead of `mlx-lm`.
