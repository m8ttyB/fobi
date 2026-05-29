"""Query contextualization for multi-turn RAG conversations.

The retrieval step is stateless — it embeds only the current question, with no
awareness of prior turns. Follow-up questions like "Tell me more" or "What about
that?" produce vague embeddings that fail to match relevant document chunks.

contextualize_query() fixes this by using the local model to rewrite a follow-up
into a self-contained question before it reaches the retriever. The rewritten
query is used only for retrieval; the original question is preserved for the
conversation history and the final generation prompt.
"""

from model import generate

_REWRITE_PROMPT = """\
Given the conversation history below, rewrite the follow-up question as a fully \
self-contained question that can be understood without the history.
Return only the rewritten question with no explanation or preamble.

Conversation history:
{history}

Follow-up question: {question}

Rewritten question:"""


def _format_history(history: list[dict]) -> str:
    """Format conversation history as a plain Q/A transcript for the rewrite prompt."""
    lines = []
    for turn in history:
        role = "Q" if turn["role"] == "user" else "A"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def contextualize_query(
    question: str,
    history: list[dict],
    model,
    tokenizer,
) -> str:
    """Rewrite a follow-up question into a standalone retrieval query.

    On the first turn (empty history) the original question is returned
    immediately without calling the model — no rewriting is needed when there
    is no prior context to incorporate.

    If the model returns an empty string the original question is used as a
    fallback so retrieval always has something to work with.

    Args:
        question: The user's current question as typed.
        history: Alternating user/assistant turns from prior exchanges.
        model: Loaded MLX model instance.
        tokenizer: Matching tokenizer instance.

    Returns:
        A self-contained query string ready for embedding and retrieval.
    """
    if not history:
        return question

    prompt = _REWRITE_PROMPT.format(
        history=_format_history(history),
        question=question,
    )
    messages = [{"role": "user", "content": prompt}]
    rewritten = generate(model, tokenizer, messages).strip()
    return rewritten if rewritten else question
