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
    def test_display_runs_without_error(self, capsys):
        main.display(SAMPLE_RESULT, truncated=False, original_length=500)
        # rich output goes to stdout — just verify no exception raised

    def test_display_truncated_runs_without_error(self, capsys):
        main.display(SAMPLE_RESULT, truncated=True, original_length=50000)


class TestMain:
    def test_missing_file_exits(self):
        with patch("sys.argv", ["main.py", "nonexistent.txt"]):
            with pytest.raises(SystemExit):
                main.main()

    def test_full_run_txt(self, tmp_path):
        doc = tmp_path / "doc.txt"
        doc.write_text("Sample document content.")
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        with (
            patch("main.load_model", return_value=(mock_model, mock_tokenizer)),
            patch("main.extract", return_value=SAMPLE_RESULT),
            patch("sys.argv", ["main.py", str(doc)]),
        ):
            main.main()  # should complete without raising

    def test_extraction_failure_exits_nonzero(self, tmp_path):
        from extractor import ExtractionError
        doc = tmp_path / "doc.txt"
        doc.write_text("Some content.")
        with (
            patch("main.load_model", return_value=(MagicMock(), MagicMock())),
            patch("main.extract", side_effect=ExtractionError("all retries failed")),
            patch("sys.argv", ["main.py", str(doc)]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main.main()
        assert exc_info.value.code != 0
