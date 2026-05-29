# doc-qa — Relevance Score Thresholding Plan

## Context

`retrieve_multi` currently returns exactly `TOP_K` chunks regardless of how well they match the query. A question with no strong match in the index still retrieves 4 chunks — the model receives low-quality context and tends to either hallucinate or produce vague answers.

Relevance score thresholding filters out chunks whose cosine similarity falls below a minimum value before they reach the prompt. This is a small, targeted change with meaningful impact on answer quality for out-of-scope or weakly-matched queries.

## How cosine scores work in this codebase

All embeddings are L2-normalized, so FAISS `IndexFlatIP` returns inner product scores that equal cosine similarity. Scores range from 0.0 to 1.0:

| Score range | Meaning |
|---|---|
| 0.8 – 1.0 | Strong semantic match |
| 0.5 – 0.8 | Moderate relevance |
| 0.3 – 0.5 | Weak / tangential match |
| < 0.3 | Likely noise |

A default threshold of `0.3` is conservative — it removes clear noise while keeping borderline matches that may still be useful. Users can tighten it via env var.

## Design decisions

- **Default threshold:** `0.3` — filters noise without being overly aggressive.
- **Fallback when no chunks pass:** Return an empty list and let `build_messages` handle it gracefully (it already handles empty chunks). The chat loop prints a message telling the user no relevant content was found rather than generating a hallucinated answer.
- **Configuration:** `QA_MIN_SCORE` env var in `config.py`, consistent with existing pattern.
- **Scope:** Applied in `retrieve_multi` only — `retrieve` (single-index, kept for backward compatibility) also gets the parameter for consistency.
- **No CLI flag:** Threshold is a tuning knob, not a per-session decision. Env var is the right interface.

## Changes

### `config.py`

Add:
```python
MIN_SCORE = float(os.environ.get("QA_MIN_SCORE", "0.3"))
```

### `retriever.py`

Add `min_score: float` parameter to both `retrieve` and `retrieve_multi`:

```python
def retrieve(query, embed_model, index, metadata, top_k, min_score=0.0) -> list[dict]:
    ...
    results = [r for r in results if r["score"] >= min_score]
    return results

def retrieve_multi(query, embed_model, indexes, top_k, min_score=0.0) -> list[dict]:
    ...
    all_results = [r for r in all_results if r["score"] >= min_score]
    all_results.sort(...)
    return all_results[:top_k]
```

Default is `0.0` (no filtering) to keep the function signatures backward-compatible. The actual default threshold lives in `config.py` and is passed in by `main.py`.

### `main.py`

Pass `min_score=config.MIN_SCORE` to `retrieve_multi` in `cmd_chat`.

When the returned chunks list is empty, print an informative message instead of sending an empty context to the model:

```
No relevant content found for that question. Try rephrasing, or check that
the relevant document has been ingested.
```

### `tests/test_retriever.py`

Add tests for:
- Chunks below threshold are excluded
- All chunks below threshold → empty list returned
- `min_score=0.0` returns all results (backward-compatible default)
- Threshold applied before top_k cap (so top_k of 4 with 2 passing threshold returns 2, not 4)

### `tests/test_main.py`

Add test for `cmd_chat` no-match path: when `retrieve_multi` returns empty, the model is not called and an informative message is printed.

## Files changed

| File | Change |
|---|---|
| `config.py` | Add `MIN_SCORE` — include module docstring update noting the new constant |
| `retriever.py` | Add `min_score` param to `retrieve` and `retrieve_multi` — update docstrings for both functions |
| `main.py` | Pass `min_score` to `retrieve_multi`; handle empty results gracefully — update `cmd_chat` docstring |
| `tests/test_retriever.py` | Add threshold tests |
| `tests/test_main.py` | Add no-match path test |
| `CLAUDE.md` | Note `MIN_SCORE` config and threshold behaviour |
| `README.md` | Document `QA_MIN_SCORE` env var and no-match behaviour |

## Implementation order

1. Add `MIN_SCORE` to `config.py` (update module docstring)
2. Add threshold tests to `test_retriever.py` → fail
3. Update `retrieve` and `retrieve_multi` in `retriever.py` with `min_score` param and updated docstrings → tests pass
4. Add no-match test to `test_main.py` → fail
5. Update `cmd_chat` in `main.py` to pass `min_score`, handle empty results, update docstring → test passes
6. Update `README.md` — add `QA_MIN_SCORE` to env vars table; document no-match message under Chat commands
7. Update `CLAUDE.md` — note `MIN_SCORE` in config section; note threshold behaviour in key design decisions
8. Full test suite + lint clean
9. Smoke test: ask an out-of-scope question and confirm the no-match message appears instead of a hallucinated answer
10. Commit

## Smoke test

```bash
make chat
> What is the population of Tokyo?   # out-of-scope — should show no-match message
> What is Bandelier Tuff?             # in-scope — should answer normally with source citation
```

## Out of scope

- Dynamic threshold (per-query adjustment)
- Exposing threshold as a `--chat` CLI flag
- Logging scores for offline analysis
