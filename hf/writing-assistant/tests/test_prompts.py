# writing-assistant/tests/test_prompts.py
import pytest
import prompts


MODES = ["rewrite", "summarize", "make_formal", "make_casual"]


def test_build_messages_returns_list_for_all_modes():
    for mode in MODES:
        result = prompts.build_messages("Some text.", mode)
        assert isinstance(result, list)


def test_build_messages_has_system_and_user_roles():
    messages = prompts.build_messages("Hello world.", "rewrite")
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert "user" in roles


def test_build_messages_user_content_contains_input_text():
    text = "The quick brown fox."
    messages = prompts.build_messages(text, "summarize")
    user_message = next(m for m in messages if m["role"] == "user")
    assert text in user_message["content"]


def test_build_messages_raises_on_unknown_mode():
    with pytest.raises(ValueError):
        prompts.build_messages("Some text.", "unknown_mode")


def test_each_mode_has_distinct_system_prompt():
    system_prompts = set()
    for mode in MODES:
        messages = prompts.build_messages("text", mode)
        system = next(m for m in messages if m["role"] == "system")
        system_prompts.add(system["content"])
    assert len(system_prompts) == len(MODES)
