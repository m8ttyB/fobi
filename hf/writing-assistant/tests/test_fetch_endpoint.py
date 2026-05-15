# writing-assistant/tests/test_fetch_endpoint.py
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import fetcher


@pytest.fixture
def client():
    with patch("main.load_model") as mock_load:
        mock_load.return_value = (MagicMock(), MagicMock())
        from main import app
        with TestClient(app) as c:
            yield c


def test_fetch_returns_extracted(client):
    fake = fetcher.ExtractedContent(text="Hi", title="Hello", truncated=False, original_length=2)
    with patch("main.fetch_and_extract", return_value=fake):
        r = client.post("/fetch", json={"url": "https://example.com/a"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "Hi"
    assert body["title"] == "Hello"
    assert body["truncated"] is False


def test_fetch_invalid_url_returns_400(client):
    with patch("main.fetch_and_extract", side_effect=fetcher.InvalidURLError("bad")):
        r = client.post("/fetch", json={"url": "not a url"})
    assert r.status_code == 400


def test_fetch_timeout_returns_504(client):
    with patch("main.fetch_and_extract", side_effect=fetcher.FetchTimeoutError("slow")):
        r = client.post("/fetch", json={"url": "https://example.com/a"})
    assert r.status_code == 504


def test_fetch_http_error_returns_502(client):
    with patch("main.fetch_and_extract", side_effect=fetcher.FetchHTTPError("HTTP 404")):
        r = client.post("/fetch", json={"url": "https://example.com/a"})
    assert r.status_code == 502


def test_fetch_unsupported_content_returns_415(client):
    with patch("main.fetch_and_extract", side_effect=fetcher.UnsupportedContentTypeError("application/pdf")):
        r = client.post("/fetch", json={"url": "https://example.com/a.pdf"})
    assert r.status_code == 415


def test_fetch_empty_extraction_returns_422(client):
    with patch("main.fetch_and_extract", side_effect=fetcher.ExtractionEmptyError("https://x.com")):
        r = client.post("/fetch", json={"url": "https://x.com"})
    assert r.status_code == 422
