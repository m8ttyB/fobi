# cli-chat/tests/test_model.py
from unittest.mock import patch, MagicMock
import model


def make_mock_tokenizer(prompt_output="<prompt>"):
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = prompt_output
    return tokenizer


def make_stream_chunk(text):
    chunk = MagicMock()
    chunk.text = text
    return chunk


def test_stream_response_yields_tokens():
    mock_model = MagicMock()
    mock_tokenizer = make_mock_tokenizer()
    history = {
        "system": "Be helpful.",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    chunks = [make_stream_chunk("Hi"), make_stream_chunk("!")]

    with patch("model.stream_generate", return_value=iter(chunks)):
        tokens = list(model.stream_response(mock_model, mock_tokenizer, history))

    assert tokens == ["Hi", "!"]


def test_stream_response_includes_system_in_messages():
    mock_model = MagicMock()
    mock_tokenizer = make_mock_tokenizer()
    history = {
        "system": "You are a pirate.",
        "messages": [{"role": "user", "content": "Ahoy"}],
    }

    with patch("model.stream_generate", return_value=iter([])):
        list(model.stream_response(mock_model, mock_tokenizer, history))

    call_args = mock_tokenizer.apply_chat_template.call_args
    messages_passed = call_args[0][0]
    assert messages_passed[0] == {"role": "system", "content": "You are a pirate."}
    assert messages_passed[1] == {"role": "user", "content": "Ahoy"}


def test_stream_response_skips_system_when_empty():
    mock_model = MagicMock()
    mock_tokenizer = make_mock_tokenizer()
    history = {
        "system": "",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    with patch("model.stream_generate", return_value=iter([])):
        list(model.stream_response(mock_model, mock_tokenizer, history))

    call_args = mock_tokenizer.apply_chat_template.call_args
    messages_passed = call_args[0][0]
    assert messages_passed[0]["role"] == "user"


def test_load_model_calls_mlx_load():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("model.load", return_value=(mock_model, mock_tokenizer)) as mock_load:
        m, t = model.load_model("/some/path")

    mock_load.assert_called_once_with("/some/path")
    assert m is mock_model
    assert t is mock_tokenizer
