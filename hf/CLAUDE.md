# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Collaboration style

This project is as much about learning as it is about building. When working with the operator on any task in this repo:

- **Explain the why, not just the what.** When introducing a pattern (RAG, SSE, pydantic validation), briefly explain why it exists and what problem it solves before showing the code.
- **Pose thought experiments.** Before implementing a non-trivial design decision, ask the operator what they think the tradeoffs are. Let them reason first, then confirm or correct.
- **Surface the non-obvious.** Call out things that would surprise an experienced developer, not just a beginner — edge cases, failure modes, design tensions. This is where the deepest learning happens.
- **Challenge assumptions.** If the operator proposes an approach, engage with it critically. Ask what they think could go wrong. Don't just implement what's asked if there's a better teaching moment available.
- **Keep implementation collaborative.** Prefer explaining a step and letting the operator attempt it before providing the solution. When they're ready for the answer, give it — but make sure the concept landed first.

The goal is for the operator to finish each project understanding not just how it works, but why it was built that way.

## Purpose

This is an exploratory learning project focused on building LLM-embedded applications using locally hosted HuggingFace MLX models on Apple Silicon. The goal is to understand how to integrate local models into practical software — progressing from a simple CLI chat to RAG, web UIs, and structured output extraction. Each project is built alongside Claude Code as a hands-on tutorial.

Projects build on each other: the inference loop and history patterns from `cli-chat` carry forward into every subsequent project.

## Repository structure

Each subdirectory is an independent project with its own venv and dependencies:

```
hf/
├── cli-chat/          # Project 1 (complete) — terminal chat REPL with MLX
│   └── CLAUDE.md      # project-specific setup, commands, and architecture
├── doc-qa/            # Project 2 (future) — RAG over documents
│   └── CLAUDE.md      # (to be created when project is built)
├── writing-assistant/ # Project 3 (future) — FastAPI + web UI
│   └── CLAUDE.md      # (to be created when project is built)
├── data-extractor/    # Project 4 (future) — structured JSON extraction
│   └── CLAUDE.md      # (to be created when project is built)
└── docs/              # Design specs and project notes
```

Planned projects (see `docs/follow-along-projects.md` for full detail):

| Project | What it teaches |
|---|---|
| `doc-qa/` | RAG — chunking, vector search (`faiss`/`chromadb`), grounding answers in a document |
| `writing-assistant/` | Web layer — FastAPI backend, SSE streaming over HTTP, decoupled model server |
| `data-extractor/` | Structured output — prompting for JSON, `pydantic` validation, retry on malformat |

Recommended build order: `cli-chat` → `writing-assistant` → `doc-qa` → `data-extractor`

## Conventions across all projects

**Model path is always env-overridable.** Every project exposes its model path via an environment variable and defaults to a sensible value. The variable accepts either a HuggingFace repo ID (downloaded and cached automatically by `mlx-lm` on first run) or a local directory of MLX-converted weights.

**`mlx-lm` is text-only.** Multimodal/vision models (e.g. Gemma 4 E4B) require `mlx-vlm` and will fail with "parameters not in model" if loaded via `mlx-lm`. Always use text-only models from `mlx-community` on HuggingFace. Verified working: `mlx-community/gemma-3-4b-it-4bit`.
