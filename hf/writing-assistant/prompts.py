# writing-assistant/prompts.py

_SYSTEM_PROMPTS = {
    "rewrite": (
        "You are a writing assistant. Rewrite the user's text to improve clarity, "
        "flow, and conciseness while preserving the original meaning. "
        "Return only the rewritten text, no commentary."
    ),
    "summarize": (
        "You are a writing assistant. Summarize the user's text into a concise "
        "paragraph that captures the key points. "
        "Return only the summary, no commentary."
    ),
    "make_formal": (
        "You are a writing assistant. Rewrite the user's text in a formal, "
        "professional tone suitable for business or academic contexts. "
        "Return only the rewritten text, no commentary."
    ),
    "make_casual": (
        "You are a writing assistant. Rewrite the user's text in a casual, "
        "conversational tone. Return only the rewritten text, no commentary."
    ),
}


def build_messages(text: str, mode: str) -> list[dict]:
    if mode not in _SYSTEM_PROMPTS:
        raise ValueError(f"Unknown mode {mode!r}. Valid modes: {list(_SYSTEM_PROMPTS)}")
    return [
        {"role": "system", "content": _SYSTEM_PROMPTS[mode]},
        {"role": "user", "content": text},
    ]
