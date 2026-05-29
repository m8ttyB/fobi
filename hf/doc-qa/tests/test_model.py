"""Tests for model.py — load_model, stream_response, and generate."""

from unittest.mock import MagicMock, patch


def _make_chunks(texts):
    """Build mock stream_generate chunks from a list of token strings."""
    chunks = []
    for text in texts:
        chunk = MagicMock()
        chunk.text = text
        chunks.append(chunk)
    return chunks


def test_generate_returns_string():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "prompt"

    with patch("model.mlx_lm.stream_generate", return_value=_make_chunks(["Hello", " world"])):
        from model import generate
        result = generate(mock_model, mock_tokenizer, [{"role": "user", "content": "hi"}])

    assert isinstance(result, str)
    assert result == "Hello world"


def test_generate_calls_model_once():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "prompt"

    with patch("model.mlx_lm.stream_generate", return_value=_make_chunks(["ok"])) as mock_gen:
        from model import generate
        generate(mock_model, mock_tokenizer, [])

    mock_gen.assert_called_once()


def test_generate_empty_response():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "prompt"

    with patch("model.mlx_lm.stream_generate", return_value=_make_chunks([])):
        from model import generate
        result = generate(mock_model, mock_tokenizer, [])

    assert result == ""


def test_stream_response_yields_tuples():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "prompt"

    with patch("model.mlx_lm.stream_generate", return_value=_make_chunks(["Hi", "!"])):
        from model import stream_response
        results = list(stream_response(mock_model, mock_tokenizer, []))

    texts = [text for text, _ in results]
    assert texts == ["Hi", "!"]
