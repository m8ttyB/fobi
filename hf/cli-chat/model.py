# cli-chat/model.py
from typing import Iterator
from mlx_lm import load, stream_generate


def load_model(model_path: str):
    """Return (model, tokenizer) loaded from model_path."""
    return load(model_path)


def stream_response(model, tokenizer, history: dict) -> Iterator[str]:
    """Yield string tokens one at a time for the next assistant turn."""
    messages = []
    if history.get("system"):
        messages.append({"role": "system", "content": history["system"]})
    messages.extend(history["messages"])

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    for chunk in stream_generate(model, tokenizer, prompt, max_tokens=1024):
        yield chunk.text
