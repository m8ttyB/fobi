SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a document. "
    "Base your answer only on the provided context. "
    "If the answer is not in the context, say so clearly."
)


def build_messages(
    history: list[dict],
    question: str,
    chunks: list[dict],
) -> list[dict]:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"--- Chunk {i} ---\n{chunk['text']}")
    context_block = "\n\n".join(context_parts)

    if context_block:
        user_content = f"Context:\n{context_block}\n\nQuestion: {question}"
    else:
        user_content = f"Question: {question}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages
