"""Tests for contextualizer.py — query rewriting using conversation history."""

from unittest.mock import MagicMock, patch


HISTORY = [
    {"role": "user", "content": "Tell me about Bandelier geology"},
    {"role": "assistant", "content": "The area is shaped by the Valles Caldera..."},
]


def test_empty_history_returns_original():
    """First turn — no history, question returned unchanged without calling model."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("contextualizer.generate") as mock_gen:
        from contextualizer import contextualize_query
        result = contextualize_query("What is Bandelier?", [], mock_model, mock_tokenizer)

    mock_gen.assert_not_called()
    assert result == "What is Bandelier?"


def test_nonempty_history_calls_model():
    """Follow-up turn — model is called to rewrite the query."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("contextualizer.generate", return_value="Tell me more about Bandelier geology") as mock_gen:
        from contextualizer import contextualize_query
        contextualize_query("Tell me more", HISTORY, mock_model, mock_tokenizer)

    mock_gen.assert_called_once()


def test_rewritten_query_returned():
    """Model output is returned as the new retrieval query."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    rewritten = "Tell me more about the geology of Bandelier, including the Valles Caldera"

    with patch("contextualizer.generate", return_value=rewritten):
        from contextualizer import contextualize_query
        result = contextualize_query("Tell me more", HISTORY, mock_model, mock_tokenizer)

    assert result == rewritten


def test_empty_model_output_falls_back_to_original():
    """If the model returns an empty string, the original question is used."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("contextualizer.generate", return_value="   "):
        from contextualizer import contextualize_query
        result = contextualize_query("Tell me more", HISTORY, mock_model, mock_tokenizer)

    assert result == "Tell me more"
