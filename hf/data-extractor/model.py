from mlx_lm import load, generate as mlx_generate


def load_model(path: str):
    """Load an MLX model and tokenizer from a HuggingFace repo ID or local path."""
    return load(path)


def generate(model, tokenizer, messages: list[dict], max_tokens: int = 4096) -> str:
    """Generate a complete response string for the given messages list."""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
