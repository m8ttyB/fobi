from typing import Iterator

import mlx_lm


def load_model(path: str) -> tuple:
    return mlx_lm.load(path)


def stream_response(model, tokenizer, messages: list[dict]) -> Iterator[tuple[str, object]]:
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    for chunk in mlx_lm.stream_generate(model, tokenizer, prompt=prompt):
        yield chunk.text, chunk
