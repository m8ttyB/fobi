# cli-chat/tests/test_history.py
import json
import pytest
from pathlib import Path
import history


def test_load_returns_empty_when_file_missing(tmp_path):
    result = history.load(str(tmp_path / "nonexistent.json"))
    assert result == {"system": "", "messages": []}


def test_load_returns_history_from_file(tmp_path):
    data = {"system": "You are a pirate.", "messages": [{"role": "user", "content": "Ahoy"}]}
    p = tmp_path / "history.json"
    p.write_text(json.dumps(data))
    result = history.load(str(p))
    assert result == data


def test_load_returns_none_on_malformed_json(tmp_path):
    p = tmp_path / "history.json"
    p.write_text("not json {{{{")
    result = history.load(str(p))
    assert result is None


def test_save_writes_json_to_disk(tmp_path):
    p = tmp_path / "history.json"
    data = {"system": "Be brief.", "messages": []}
    history.save(str(p), data)
    assert json.loads(p.read_text()) == data


def test_append_adds_user_message(tmp_path):
    h = {"system": "", "messages": []}
    history.append(h, "user", "Hello")
    assert h["messages"] == [{"role": "user", "content": "Hello"}]


def test_append_adds_multiple_messages_in_order(tmp_path):
    h = {"system": "", "messages": []}
    history.append(h, "user", "Hello")
    history.append(h, "assistant", "Hi!")
    assert len(h["messages"]) == 2
    assert h["messages"][1] == {"role": "assistant", "content": "Hi!"}
