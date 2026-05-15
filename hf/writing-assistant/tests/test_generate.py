# writing-assistant/tests/test_generate.py
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def make_chunk(text):
    chunk = MagicMock()
    chunk.text = text
    return chunk


@pytest.fixture
def client():
    with patch("main.load_model") as mock_load:
        mock_load.return_value = (MagicMock(), MagicMock())
        from main import app
        with TestClient(app) as c:
            yield c


def test_generate_streams_tokens(client):
    chunks = [make_chunk("The"), make_chunk(" cat")]

    with patch("main.stream_response", return_value=iter([(c.text, c) for c in chunks])):
        with client.stream("POST", "/generate", json={"text": "A cat sat.", "mode": "rewrite"}) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            body = r.read().decode()

    assert "data: The\n\n" in body
    assert "data:  cat\n\n" in body
    assert "event: done" in body


def test_generate_rejects_unknown_mode(client):
    response = client.post("/generate", json={"text": "Hello", "mode": "explode"})
    assert response.status_code == 422


def test_generate_rejects_empty_text(client):
    response = client.post("/generate", json={"text": "", "mode": "rewrite"})
    assert response.status_code == 422


def test_root_serves_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
