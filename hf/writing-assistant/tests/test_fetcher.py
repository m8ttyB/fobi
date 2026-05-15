# writing-assistant/tests/test_fetcher.py
import pytest
import httpx
from unittest.mock import patch, MagicMock
import fetcher


def make_response(status=200, content_type="text/html; charset=utf-8", text="<html><body><article>Hi</article></body></html>"):
    r = MagicMock(spec=httpx.Response)
    r.status_code = status
    r.headers = {"content-type": content_type}
    r.text = text
    return r


def test_invalid_url_raises():
    with pytest.raises(fetcher.InvalidURLError):
        fetcher.fetch_and_extract("not a url")


def test_invalid_scheme_raises():
    with pytest.raises(fetcher.InvalidURLError):
        fetcher.fetch_and_extract("ftp://example.com")


def test_success_returns_extracted():
    with patch("fetcher.httpx.get", return_value=make_response()):
        with patch("fetcher.trafilatura.extract", return_value="Hi there."):
            with patch("fetcher.trafilatura.extract_metadata", return_value=None):
                result = fetcher.fetch_and_extract("https://example.com/a")
    assert result.text == "Hi there."
    assert result.truncated is False
    assert result.original_length == len("Hi there.")


def test_http_error_raises():
    with patch("fetcher.httpx.get", return_value=make_response(status=404)):
        with pytest.raises(fetcher.FetchHTTPError):
            fetcher.fetch_and_extract("https://example.com/a")


def test_timeout_raises():
    with patch("fetcher.httpx.get", side_effect=httpx.TimeoutException("slow")):
        with pytest.raises(fetcher.FetchTimeoutError):
            fetcher.fetch_and_extract("https://example.com/a")


def test_non_html_content_type_raises():
    with patch("fetcher.httpx.get", return_value=make_response(content_type="application/pdf")):
        with pytest.raises(fetcher.UnsupportedContentTypeError):
            fetcher.fetch_and_extract("https://example.com/a")


def test_empty_extraction_raises():
    with patch("fetcher.httpx.get", return_value=make_response()):
        with patch("fetcher.trafilatura.extract", return_value=None):
            with pytest.raises(fetcher.ExtractionEmptyError):
                fetcher.fetch_and_extract("https://example.com/a")


def test_long_content_is_truncated():
    long_text = "x" * 50_000
    with patch("fetcher.httpx.get", return_value=make_response()):
        with patch("fetcher.trafilatura.extract", return_value=long_text):
            with patch("fetcher.trafilatura.extract_metadata", return_value=None):
                result = fetcher.fetch_and_extract("https://example.com/a")
    assert result.truncated is True
    assert result.original_length == 50_000
    assert len(result.text) == fetcher.MAX_CHARS


def test_title_extracted_from_metadata():
    meta = MagicMock()
    meta.title = "My Article"
    with patch("fetcher.httpx.get", return_value=make_response()):
        with patch("fetcher.trafilatura.extract", return_value="Body text."):
            with patch("fetcher.trafilatura.extract_metadata", return_value=meta):
                result = fetcher.fetch_and_extract("https://example.com/a")
    assert result.title == "My Article"
