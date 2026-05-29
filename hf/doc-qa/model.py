"""Thin wrapper over mlx_lm for loading and streaming from a local model.

Provides two generation modes:
- stream_response: yields tokens incrementally for interactive chat output.
- generate: collects the full response and returns it as a string, used for
  internal model calls (e.g. query rewriting) where streaming is not needed.
"""

from typing import Iterator

import mlx_lm


def load_model(path: str) -> tuple:
    """Load model weights and tokenizer from a HuggingFace repo ID or local path."""
    return mlx_lm.load(path)


def stream_response(model, tokenizer, messages: list[dict]) -> Iterator[tuple[str, object]]:
    """Yield (token_text, raw_chunk) tuples for each token produced by the model.

    Applies the tokenizer's chat template before generation so the model receives
    a correctly formatted prompt regardless of which chat format it expects.
    """
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    for chunk in mlx_lm.stream_generate(model, tokenizer, prompt=prompt):
        yield chunk.text, chunk


def generate(model, tokenizer, messages: list[dict]) -> str:
    """Return the full model response as a string without streaming.

    Uses the same chat template as stream_response. Intended for short
    internal generation tasks (e.g. query rewriting) where incremental
    output is not needed and latency should be minimised.
    """
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    return "".join(
        chunk.text for chunk in mlx_lm.stream_generate(model, tokenizer, prompt=prompt)
    )
