# writing-assistant/model.py
from typing import Iterator
from mlx_lm import load, stream_generate


def load_model(model_path: str):
    return load(model_path)


def stream_response(model, tokenizer, messages: list[dict]) -> Iterator[tuple[str, object]]:
    """Yield (token_text, raw_chunk) pairs for the given messages."""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    for chunk in stream_generate(model, tokenizer, prompt, max_tokens=1024):
        yield chunk.text, chunk
