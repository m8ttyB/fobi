# doc-qa — Query Contextualization Plan

## Context

The chat REPL supports multi-turn conversation via accumulating history, but the retrieval step is stateless — it only embeds the raw text of the current question. Follow-up queries like "Tell me more", "Go deeper", or "What about that?" produce semantically vague embeddings that score below the relevance threshold, returning no chunks even when relevant content exists in the index.

The fix is query contextualization: before embedding the user's question, rewrite it into a self-contained query using recent conversation history. This is a standard production RAG pattern.

## How it works

```
User types:   "Tell me more"
History:      Q: "Tell me about Bandelier geology"
              A: "The area is shaped by the Valles Caldera..."

         ↓ contextualize_query()

Rewritten:    "Tell me more about the geology of Bandelier,
               including the Valles Caldera and Bandelier Tuff"

         ↓ embed + retrieve

FAISS returns relevant geological chunks → model generates answer
```

The rewrite is done by the local Gemma model using a short, focused prompt. The rewritten query is used **only for retrieval** — the original question is still what goes into the conversation history and the final generation prompt.

## Design decisions

- **Approach:** LLM-based rewriting — most accurate, produces a semantically rich query that retrieves the right chunks even for vague follow-ups.
- **When to apply:** Always when history is non-empty. Applying only to "short" queries requires a fragile heuristic; always rewriting is simpler and more predictable. When history is empty the question is returned unchanged (no rewrite needed for the first turn).
- **Model:** Same local Gemma instance already loaded in `cmd_chat` — no second model needed.
- **Latency:** One short non-streaming generation pass before each retrieval. The rewrite prompt is kept minimal to keep this fast.
- **Rewrite prompt:** Instructs the model to return only the rewritten question with no preamble or explanation, so the output can be used directly as the embedding input.
- **Fallback:** If the rewrite returns empty or fails, fall back to the original question — retrieval continues as before.

## New module: `contextualizer.py`

```python
REWRITE_PROMPT = """Given the conversation history below, rewrite the follow-up question
as a fully self-contained question that can be understood without the history.
Return only the rewritten question, no explanation.

Conversation history:
{history}

Follow-up question: {question}

Rewritten question:"""

def contextualize_query(
    question: str,
    history: list[dict],
    model,
    tokenizer,
) -> str:
    """Rewrite a follow-up question into a standalone query using conversation history.

    Returns the original question unchanged if history is empty (first turn)
    or if the model returns an empty string.
    """
```

## Changes

### `contextualizer.py` (new)

Single public function `contextualize_query(question, history, model, tokenizer) -> str`. Uses a non-streaming generation call (collects the full rewrite before proceeding). Defined rewrite prompt kept in this module as the single source of truth.

### `model.py`

Add `generate(model, tokenizer, messages) -> str` alongside the existing `stream_response` — a non-streaming variant that returns the full generated text as a string. Used by `contextualizer.py` for the rewrite pass.

### `main.py`

In `cmd_chat`, after reading the user's question and before calling `retrieve_multi`, call `contextualize_query` and use the returned string as the retrieval query. The original question is still used for history and the generation prompt.

```python
retrieval_query = contextualize_query(question, history, model, tokenizer)
chunks = retrieve_multi(retrieval_query, embed_model, indexes, top_k=config.TOP_K, min_score=config.MIN_SCORE)
```

Update `cmd_chat` docstring to note contextualization.

### `tests/test_contextualizer.py` (new)

| Test | What it covers |
|---|---|
| `test_empty_history_returns_original` | No history → original question returned unchanged |
| `test_nonempty_history_calls_model` | History present → model is called |
| `test_empty_model_output_falls_back` | Model returns empty string → original question returned |
| `test_rewritten_query_returned` | Model returns rewrite → rewrite is returned |

### `tests/test_generate.py` (new or extended)

Add test for `generate()` in `model.py` — confirms it returns a string and calls the model once.

### `README.md`

Add a note under **How it works** explaining query contextualization and when it fires.

### `CLAUDE.md`

Add `contextualizer.py` to the architecture section. Note `generate()` in the `model.py` description.

## Files changed

| File | Change |
|---|---|
| `contextualizer.py` | New module — `contextualize_query` with docstrings |
| `model.py` | Add `generate()` non-streaming variant with docstring |
| `main.py` | Wire `contextualize_query` into `cmd_chat`; update docstring |
| `tests/test_contextualizer.py` | New — 4 tests, all mocked |
| `README.md` | Document contextualization under How it works |
| `CLAUDE.md` | Add `contextualizer.py` to architecture; update `model.py` description |

## Implementation order

1. Add `generate()` to `model.py` with docstring
2. Write tests for `generate()` → pass
3. Write tests for `contextualizer.py` → fail
4. Implement `contextualizer.py` with docstrings → tests pass
5. Update `cmd_chat` in `main.py` to call `contextualize_query`; update docstring
6. Update `README.md`
7. Update `CLAUDE.md`
8. Full test suite + lint clean
9. Smoke test: ask a question, then "Tell me more" — confirm retrieval succeeds and cites sources
10. Commit

## Smoke test

```bash
make chat
> Tell me about the geology of Bandelier
[answer streams, Sources: Bandelier_Tuff.pdf]
> Tell me more
[answer streams with additional geological detail — no "No relevant content" message]
> What formed the caldera?
[answer streams — specific follow-up resolved via rewrite]
```

## Out of scope

- Caching rewritten queries
- Showing the user the rewritten query (debug mode)
- Applying contextualization to the generation prompt (history already handles that)
