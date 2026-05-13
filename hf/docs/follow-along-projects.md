# Follow-Along LLM Projects

These projects build on each other progressively. Each one is designed to be coded alongside Claude Code, using a locally hosted MLX model (Gemma 4) on Apple Silicon.

---

## Project 2 — Document Q&A (RAG)

**What it does:** Load a PDF or plain text file, ask questions about it, and get answers grounded in that document's content.

**What you learn:**
- Chunking text into retrievable segments
- Retrieval-augmented generation (RAG) — injecting relevant context into prompts
- How to prevent the model from hallucinating outside its source material
- Vector similarity search (using a lightweight local library like `faiss` or `chromadb`)

**Suggested stack:** Python, `mlx-lm`, `pypdf` or `pdfminer`, `faiss-cpu` or `chromadb`

**Natural next step after:** Project 1 (you'll reuse the inference loop and conversation history patterns)

---

## Project 3 — Writing Assistant Web App

**What it does:** A minimal FastAPI backend + single HTML page. Paste in a paragraph, choose a mode (rewrite, summarize, make formal, make casual), click a button, and get the result streamed back in the browser.

**What you learn:**
- Wrapping an MLX model in a FastAPI server
- Streaming model output over HTTP using Server-Sent Events (SSE)
- Building a minimal web UI that talks to a locally running model
- Decoupling the model server from the UI layer

**Suggested stack:** Python, `mlx-lm`, `fastapi`, `uvicorn`, vanilla HTML/JS (no build step)

**Natural next step after:** Project 1 (adds the web layer on top of the inference patterns you learned)

---

## Project 4 — Structured Data Extractor

**What it does:** Give it unstructured text (an email, a meeting note, a recipe) and it extracts structured fields into JSON — e.g., action items, dates, names, ingredients.

**What you learn:**
- Prompting for structured / JSON output reliably
- Validating model output with `pydantic`
- Retry logic when the model doesn't follow format
- LLM failure modes and how to handle them gracefully

**Suggested stack:** Python, `mlx-lm`, `pydantic`, `rich` (for display)

**Natural next step after:** Projects 1 and 3 (combines inference patterns with structured output and validation)

---

## Recommended Order

```
Project 1 (CLI Chat)  →  Project 3 (Web App)  →  Project 2 (RAG)  →  Project 4 (Extractor)
```

Projects 2 and 4 can be done in either order after 3. RAG (2) is more architecturally complex; Structured Output (4) is more about prompt engineering discipline.
