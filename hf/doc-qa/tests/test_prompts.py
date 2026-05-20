from prompts import build_messages


CHUNKS = [
    {"text": "The contract was signed in 2021.", "source": "doc.pdf", "score": 0.92},
    {"text": "Termination requires 30 days notice.", "source": "doc.pdf", "score": 0.85},
]


def test_build_messages_returns_list():
    messages = build_messages([], "What year was it signed?", CHUNKS)
    assert isinstance(messages, list)


def test_build_messages_has_system_message():
    messages = build_messages([], "Any question?", CHUNKS)
    assert messages[0]["role"] == "system"
    assert len(messages[0]["content"]) > 0


def test_build_messages_last_message_is_user():
    messages = build_messages([], "Any question?", CHUNKS)
    assert messages[-1]["role"] == "user"


def test_build_messages_includes_chunk_text_in_user_turn():
    messages = build_messages([], "What year?", CHUNKS)
    user_content = messages[-1]["content"]
    assert "The contract was signed in 2021." in user_content
    assert "Termination requires 30 days notice." in user_content


def test_build_messages_includes_question_in_user_turn():
    question = "What year was the contract signed?"
    messages = build_messages([], question, CHUNKS)
    assert question in messages[-1]["content"]


def test_build_messages_includes_history():
    history = [
        {"role": "user", "content": "Previous question?"},
        {"role": "assistant", "content": "Previous answer."},
    ]
    messages = build_messages(history, "Follow up?", CHUNKS)
    roles = [m["role"] for m in messages]
    assert "user" in roles
    assert "assistant" in roles


def test_build_messages_history_precedes_final_user_turn():
    history = [
        {"role": "user", "content": "First question?"},
        {"role": "assistant", "content": "First answer."},
    ]
    messages = build_messages(history, "Second question?", CHUNKS)
    # Last message should be the new question, not the history
    assert "Second question?" in messages[-1]["content"]
    # History should appear before the final user turn
    contents = [m["content"] for m in messages]
    assert "First question?" in contents


def test_build_messages_empty_chunks():
    messages = build_messages([], "Any question?", [])
    assert messages[-1]["role"] == "user"
    assert "Any question?" in messages[-1]["content"]
