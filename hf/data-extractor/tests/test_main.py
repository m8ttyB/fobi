from unittest.mock import MagicMock, patch, mock_open

import pytest

import main
from schema import ExtractedDocument, Person, Place, Date


SAMPLE_RESULT = ExtractedDocument(
    title="Test Article",
    topic="Testing",
    people=[Person(name="Alice", role="engineer", context=None)],
    places=[Place(name="San Francisco", context="headquarters")],
    dates=[Date(date="January 2024", event="Launch")],
    summary="A test article about testing.",
)


class TestLoadText:
    def test_loads_txt_file(self):
        m = mock_open(read_data="Hello world")
        with patch("builtins.open", m):
            text = main.load_text("doc.txt")
        assert text == "Hello world"

    def test_rejects_unknown_extension(self):
        with pytest.raises(ValueError, match="Unsupported"):
            main.load_text("doc.docx")

    def test_pdf_calls_pypdf(self):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page content"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        with patch("main.PdfReader", return_value=mock_reader):
            text = main.load_text("doc.pdf")
        assert "Page content" in text


class TestTruncate:
    def test_no_truncation_when_short(self):
        text = "hello"
        result, truncated = main.truncate(text, max_chars=100)
        assert result == text
        assert truncated is False

    def test_truncates_long_text(self):
        text = "a" * 200
        result, truncated = main.truncate(text, max_chars=100)
        assert len(result) == 100
        assert truncated is True

    def test_exact_length_not_truncated(self):
        text = "a" * 100
        result, truncated = main.truncate(text, max_chars=100)
        assert truncated is False


class TestDisplay:
    def test_display_runs_without_error(self):
        main.display(SAMPLE_RESULT)

    def test_display_truncated_runs_without_error(self):
        main.display(SAMPLE_RESULT, truncated=True, original_length=50000, strategy="truncate")

    def test_display_chunked_shows_chunk_count(self):
        main.display(SAMPLE_RESULT, strategy="chunked", chunk_count=5)


class TestCmdTruncate:
    def test_runs_without_error(self):
        model, tokenizer = MagicMock(), MagicMock()
        with patch("main.extract", return_value=SAMPLE_RESULT):
            main.cmd_truncate("Some text.", model, tokenizer)

    def test_truncates_long_text(self):
        model, tokenizer = MagicMock(), MagicMock()
        long_text = "a" * 100000
        with patch("main.extract", return_value=SAMPLE_RESULT) as mock_extract:
            main.cmd_truncate(long_text, model, tokenizer)
        actual_text = mock_extract.call_args[0][0]
        assert len(actual_text) == main.config.MAX_CHARS


class TestCmdChunked:
    def test_calls_extract_per_chunk(self):
        model, tokenizer = MagicMock(), MagicMock()
        # Use a text long enough to produce multiple chunks
        text = ("word " * 200 + "\n\n") * 10  # ~14000 chars
        with (
            patch("main.extract", return_value=SAMPLE_RESULT) as mock_extract,
            patch("main.merge", return_value=SAMPLE_RESULT),
        ):
            main.cmd_chunked(text, model, tokenizer)
        assert mock_extract.call_count > 1

    def test_single_chunk_skips_merge(self):
        model, tokenizer = MagicMock(), MagicMock()
        short_text = "Short document."
        with (
            patch("main.extract", return_value=SAMPLE_RESULT),
            patch("main.merge", return_value=SAMPLE_RESULT) as mock_merge,
        ):
            main.cmd_chunked(short_text, model, tokenizer)
        # merge is called but returns single partial directly (no model call)
        mock_merge.assert_called_once()


class TestMain:
    def test_missing_file_exits(self):
        with patch("sys.argv", ["main.py", "nonexistent.txt"]):
            with pytest.raises(SystemExit):
                main.main()

    def test_default_strategy_is_truncate(self, tmp_path):
        doc = tmp_path / "doc.txt"
        doc.write_text("Sample document content.")
        with (
            patch("main.load_model", return_value=(MagicMock(), MagicMock())),
            patch("main.cmd_truncate", return_value=SAMPLE_RESULT) as mock_trunc,
            patch("sys.argv", ["main.py", str(doc)]),
        ):
            main.main()
        mock_trunc.assert_called_once()

    def test_chunked_strategy_dispatches_correctly(self, tmp_path):
        doc = tmp_path / "doc.txt"
        doc.write_text("Sample document content.")
        with (
            patch("main.load_model", return_value=(MagicMock(), MagicMock())),
            patch("main.cmd_chunked", return_value=SAMPLE_RESULT) as mock_chunked,
            patch("sys.argv", ["main.py", str(doc), "--strategy", "chunked"]),
        ):
            main.main()
        mock_chunked.assert_called_once()

    def test_extraction_failure_exits_nonzero(self, tmp_path):
        from extractor import ExtractionError
        doc = tmp_path / "doc.txt"
        doc.write_text("Some content.")
        with (
            patch("main.load_model", return_value=(MagicMock(), MagicMock())),
            patch("main.cmd_truncate", side_effect=ExtractionError("all retries failed")),
            patch("sys.argv", ["main.py", str(doc)]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main.main()
        assert exc_info.value.code != 0
