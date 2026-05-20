"""Prompt construction for document-grounded Q&A.

Retrieved chunks are formatted as a numbered context block and prepended to the
user's question. The system prompt instructs the model to answer only from that
context, which reduces hallucination on out-of-scope questions.
"""

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
    """Build a messages list ready for tokenizer.apply_chat_template.

    Args:
        history: Alternating user/assistant turns from previous exchanges.
        question: The current user question.
        chunks: Retrieved document chunks, each with at least a 'text' key.

    Returns:
        [system, ...history, user_with_context] — the user turn embeds retrieved
        chunks as a context block above the question.
    """
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
