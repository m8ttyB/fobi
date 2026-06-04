import json
from unittest.mock import MagicMock, patch

import pytest

from extractor import ExtractionError, extract, _strip_fences
from schema import ExtractedDocument


VALID_JSON = json.dumps({
    "title": "Test Article",
    "topic": "Testing",
    "people": [{"name": "Alice", "role": "engineer", "context": None}],
    "places": [{"name": "San Francisco", "context": "headquarters"}],
    "dates": [{"date": "January 2024", "event": "Launch"}],
    "summary": "A test article about testing.",
})


def _make_model():
    return MagicMock(), MagicMock()


class TestStripFences:
    def test_no_fences(self):
        assert _strip_fences('{"a": 1}') == '{"a": 1}'

    def test_json_fence(self):
        text = "```json\n{\"a\": 1}\n```"
        assert _strip_fences(text) == '{"a": 1}'

    def test_plain_fence(self):
        text = "```\n{\"a\": 1}\n```"
        assert _strip_fences(text) == '{"a": 1}'

    def test_fence_with_preamble(self):
        text = "Sure! Here is the JSON:\n```json\n{\"a\": 1}\n```"
        assert _strip_fences(text) == '{"a": 1}'

    def test_strips_surrounding_whitespace(self):
        assert _strip_fences("  \n" + VALID_JSON + "\n  ") == VALID_JSON


class TestExtractSuccess:
    def test_returns_extracted_document(self):
        model, tokenizer = _make_model()
        with patch("extractor.generate", return_value=VALID_JSON):
            result = extract("Some document text.", model, tokenizer)
        assert isinstance(result, ExtractedDocument)
        assert result.topic == "Testing"
        assert result.people[0].name == "Alice"

    def test_strips_fences_before_parsing(self):
        fenced = f"```json\n{VALID_JSON}\n```"
        model, tokenizer = _make_model()
        with patch("extractor.generate", return_value=fenced):
            result = extract("Some text.", model, tokenizer)
        assert result.title == "Test Article"

    def test_strips_preamble_with_fences(self):
        response = f"Here is the extracted JSON:\n```json\n{VALID_JSON}\n```"
        model, tokenizer = _make_model()
        with patch("extractor.generate", return_value=response):
            result = extract("Some text.", model, tokenizer)
        assert result.summary == "A test article about testing."


class TestExtractRetry:
    def test_retries_on_invalid_json(self):
        model, tokenizer = _make_model()
        responses = ["not json at all", VALID_JSON]
        with patch("extractor.generate", side_effect=responses):
            result = extract("Some text.", model, tokenizer, max_retries=2)
        assert result.topic == "Testing"

    def test_retries_on_validation_error(self):
        missing_topic = json.dumps({
            "title": "T",
            "people": [],
            "places": [],
            "dates": [],
            "summary": "S",
        })
        model, tokenizer = _make_model()
        responses = [missing_topic, VALID_JSON]
        with patch("extractor.generate", side_effect=responses):
            result = extract("Some text.", model, tokenizer, max_retries=2)
        assert result.topic == "Testing"

    def test_retry_appends_error_to_messages(self):
        captured = []

        def fake_generate(model, tokenizer, messages):
            captured.append(list(messages))
            if len(captured) == 1:
                return "not json"
            return VALID_JSON

        model, tokenizer = _make_model()
        with patch("extractor.generate", side_effect=fake_generate):
            extract("Some text.", model, tokenizer, max_retries=2)

        # Second call should have more messages than the first
        assert len(captured[1]) > len(captured[0])
        # The appended messages should mention the error
        last_content = captured[1][-1]["content"]
        assert "failed" in last_content.lower() or "error" in last_content.lower()

    def test_exhaustion_raises_extraction_error(self):
        model, tokenizer = _make_model()
        with patch("extractor.generate", return_value="not json"):
            with pytest.raises(ExtractionError):
                extract("Some text.", model, tokenizer, max_retries=3)

    def test_exhaustion_error_includes_last_error(self):
        model, tokenizer = _make_model()
        with patch("extractor.generate", return_value="bad"):
            with pytest.raises(ExtractionError, match="JSON"):
                extract("Some text.", model, tokenizer, max_retries=1)
