"""Thin wrapper over mlx_lm for loading and streaming from a local model."""

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
