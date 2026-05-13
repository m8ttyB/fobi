# cli-chat/tests/test_commands.py
import json
import pytest
from unittest.mock import patch
import commands


def make_history(system="Be helpful.", messages=None):
    return {"system": system, "messages": messages or []}


def test_returns_false_for_plain_text(tmp_path):
    h = make_history()
    result = commands.handle("Hello there", h, str(tmp_path / "h.json"))
    assert result is False


def test_quit_calls_sys_exit(tmp_path):
    h = make_history()
    with pytest.raises(SystemExit):
        commands.handle("/quit", h, str(tmp_path / "h.json"))


def test_clear_empties_messages_and_saves(tmp_path):
    p = tmp_path / "h.json"
    h = make_history(messages=[{"role": "user", "content": "Hi"}])
    commands.handle("/clear", h, str(p))
    assert h["messages"] == []
    assert json.loads(p.read_text())["messages"] == []


def test_system_sets_prompt_clears_messages_and_saves(tmp_path):
    p = tmp_path / "h.json"
    h = make_history(messages=[{"role": "user", "content": "Hi"}])
    commands.handle('/system "You are a pirate."', h, str(p))
    assert h["system"] == "You are a pirate."
    assert h["messages"] == []
    saved = json.loads(p.read_text())
    assert saved["system"] == "You are a pirate."


def test_system_with_single_quotes(tmp_path):
    p = tmp_path / "h.json"
    h = make_history()
    commands.handle("/system 'Be terse.'", h, str(p))
    assert h["system"] == "Be terse."


def test_system_without_quotes(tmp_path):
    p = tmp_path / "h.json"
    h = make_history()
    commands.handle("/system Be concise.", h, str(p))
    assert h["system"] == "Be concise."


def test_unknown_command_returns_true(tmp_path):
    h = make_history()
    result = commands.handle("/bogus", h, str(tmp_path / "h.json"))
    assert result is True


def test_clear_returns_true(tmp_path):
    h = make_history()
    result = commands.handle("/clear", h, str(tmp_path / "h.json"))
    assert result is True


def test_history_command_returns_true(tmp_path):
    h = make_history()
    result = commands.handle("/history", h, str(tmp_path / "h.json"))
    assert result is True
